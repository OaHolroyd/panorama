#include "resident_tile_cache.h"

#import <Foundation/Foundation.h>

#include "metal_tile.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

#ifndef PANORAMA_METALLIB_PATH
#define PANORAMA_METALLIB_PATH "obj/release/raytracing/panorama.metallib"
#endif

constexpr const char *kMetallibPath = PANORAMA_METALLIB_PATH;

static_assert(sizeof(ResidentTile) == 3U * sizeof(uint64_t));
static_assert(sizeof(ResidentTileHashEntry) == 3U * sizeof(uint64_t));
static_assert(sizeof(QuantizedTerrainLayout) == 3U * sizeof(uint32_t));

/// Check that a byte count fits Metal's NSUInteger buffer-length argument.
[[nodiscard]] NSUInteger checked_buffer_length(size_t count, size_t size, const char *name) {
  if (count > std::numeric_limits<size_t>::max() / size) {
    throw std::overflow_error(std::string(name) + " buffer is too large");
  }
  return static_cast<NSUInteger>(count * size);
}

/// Allocate storage with the requested Metal access and hazard-tracking mode.
[[nodiscard]] id<MTLBuffer> make_buffer(
    id<MTLDevice> device,
    NSUInteger length,
    const char *name,
    MTLResourceOptions options = MTLResourceStorageModeShared
) {
  id<MTLBuffer> buffer = [device newBufferWithLength:length options:options];
  if (buffer == nil) {
    throw std::runtime_error(std::string("Could not allocate ") + name + " Metal buffer");
  }
  return buffer;
}

/// Print one Objective-C error using the renderer's command-line convention.
void print_error(NSString *context, NSError *error) {
  std::fprintf(stderr, "%s: %s\n", context.UTF8String, error.localizedDescription.UTF8String);
}

/// Mix one unsigned 64-bit value for the resident-tile hash table.
[[nodiscard]] uint64_t mix_tile_hash(uint64_t value) {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return value;
}

/// Return the hash used by host and Metal resident-tile lookup tables.
[[nodiscard]] uint64_t tile_key_hash(TileKey key) {
  const uint64_t row = mix_tile_hash(static_cast<uint64_t>(key.row));
  const uint64_t column = mix_tile_hash(static_cast<uint64_t>(key.column));
  return mix_tile_hash(row ^ (column + 0x9e3779b97f4a7c15ULL));
}

/// Return a power-of-two hash capacity at no more than 50% load.
[[nodiscard]] uint32_t resident_hash_capacity(uint32_t slot_count) {
  const uint64_t required = 2U * static_cast<uint64_t>(slot_count);
  uint64_t capacity = 1U;
  while (capacity < required) {
    capacity <<= 1U;
  }
  if (capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::overflow_error("Resident tile hash table exceeds Metal uint range");
  }
  return static_cast<uint32_t>(capacity);
}

/// Build observer-relative metadata for a resident atlas slot.
[[nodiscard]] ResidentTile
make_resident_tile(const LoadedTile &tile, TileKey key, const RaytraceConfig &config) {
  const double x = tile.lower_left_x - config.observer.easting;
  const double y = tile.lower_left_y - config.observer.northing;
  if (x < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      x > static_cast<double>(std::numeric_limits<float>::max()) ||
      y < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      y > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::overflow_error("Resident tile origin does not fit float32");
  }
  return {static_cast<float>(x), static_cast<float>(y), key.row, key.column};
}

/// Build resident metadata directly from projected tile coordinates.
[[nodiscard]] ResidentTile make_resident_tile(
    double lower_left_x,
    double lower_left_y,
    TileKey key,
    const RaytraceConfig &config
) {
  const double x = lower_left_x - config.observer.easting;
  const double y = lower_left_y - config.observer.northing;
  if (x < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      x > static_cast<double>(std::numeric_limits<float>::max()) ||
      y < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      y > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::overflow_error("Resident tile origin does not fit float32");
  }
  return {static_cast<float>(x), static_cast<float>(y), key.row, key.column};
}

/// One prepared source paired with its selected destination atlas slot.
struct AtlasInstallation {
  uint32_t slot;
  PreparedTile prepared;
  double lower_left_x;
  double lower_left_y;
};

/// Resources retained while one synchronous mipmap command is running.
struct MipmapSubmission {
  id<MTLBuffer> slot_buffer;
  id<MTLCommandBuffer> command;
};

} // namespace

/// Mutable atlas state hidden behind the cache's ownership-oriented interface.
struct ResidentTileCache::State {
  std::span<const TerrainSource> sources;
  const RaytraceConfig &config;
  id<MTLDevice> device;
  id<MTLIOCommandQueue> io_queue;
  id<MTLCommandQueue> mipmap_queue;
  id<MTLComputePipelineState> conversion_pipeline;
  id<MTLComputePipelineState> mipmap_pipeline;
  MetalTileHeader header_template;
  bool retain_quantized;
  double grid_origin_x;
  double grid_origin_y;
  double tile_width;
  uint32_t mip_count;
  uint32_t vertex_count;
  uint32_t slot_capacity;
  uint32_t hash_slot_count;
  uint32_t resident_count = 1U;
  uint64_t next_use_stamp = 2U;
  uint64_t installations = 1U;
  uint64_t bytes_copied;
  uint64_t bytes_loaded_with_metal_io = 0U;
  uint64_t evictions = 0U;
  id<MTLBuffer> mipmap_atlas;
  id<MTLBuffer> vertex_atlas;
  id<MTLBuffer> quantized_staging;
  id<MTLBuffer> metadata_buffer;
  id<MTLBuffer> hash_buffer;
  float *mipmaps;
  float *vertices;
  ResidentTile *metadata;
  ResidentTileHashEntry *hash;
  std::vector<uint32_t> slot_by_source;
  std::vector<uint32_t> source_by_slot;
  std::vector<uint64_t> last_used;

  /// Submit every maximum-mipmap level for the supplied resident slots.
  [[nodiscard]] MipmapSubmission submit_mipmaps(std::span<const uint32_t> slots);

  /// Load custom vertices directly, retaining or expanding fixed-point records.
  void load_custom_vertices(
      std::span<const MetalTileBufferLoad> loads,
      std::span<const uint32_t> slots,
      Timer &timer
  );

  /// Build mipmaps synchronously where the observer tile requires them now.
  void generate_mipmaps(std::span<const uint32_t> slots, Timer &timer);

  /// Rebuild the GPU key lookup from slots whose payloads are fully resident.
  void rebuild_hash(Timer *timer);

  /// Initialise fixed atlas dimensions before allocating Metal resources.
  State(
      std::span<const TerrainSource> source_values,
      const RaytraceConfig &config_value,
      id<MTLDevice> device_value,
      const MetalTileHeader &header_value,
      bool retain_quantized_value,
      double origin_x,
      double origin_y,
      double width,
      uint32_t mip_values,
      uint32_t vertex_values,
      uint32_t slot_values,
      uint32_t hash_values,
      uint64_t initial_bytes
  )
      : sources(source_values), config(config_value), device(device_value),
        header_template(header_value), retain_quantized(retain_quantized_value),
        grid_origin_x(origin_x), grid_origin_y(origin_y),
        tile_width(width), mip_count(mip_values), vertex_count(vertex_values),
        slot_capacity(slot_values), hash_slot_count(hash_values), bytes_copied(initial_bytes) {}
};

void ResidentTileCache::State::load_custom_vertices(
    std::span<const MetalTileBufferLoad> loads,
    std::span<const uint32_t> slots,
    Timer &timer
) {
  if (loads.empty() || loads.size() != slots.size()) {
    throw std::invalid_argument("Metal tile loads require one destination slot per file");
  }
  if (io_queue == nil) {
    throw std::logic_error("Metal tile loading resources are unavailable");
  }

  if (header_template.sample_type == MetalTileSampleType::Float32) {
    std::vector<MetalTileBufferLoad> direct_loads;
    direct_loads.reserve(loads.size());
    for (size_t index = 0U; index < loads.size(); index++) {
      direct_loads.push_back(
          {
              loads[index].path,
              static_cast<NSUInteger>(slots[index]) * header_template.vertex_byte_count,
              loads[index].file,
          }
      );
    }
    timer.start_wall("Metal tile I/O");
    load_metal_tiles_into_buffer(
        device,
        io_queue,
        direct_loads,
        header_template.vertex_offset,
        header_template.vertex_byte_count,
        vertex_atlas,
        vertex_atlas.length
    );
    timer.stop("Metal tile I/O");
    return;
  }
  if (retain_quantized) {
    const QuantizedMetalTileRecordLayout record =
        quantized_metal_tile_record_layout(header_template);
    std::vector<MetalTileBufferLoad> direct_loads;
    direct_loads.reserve(loads.size());
    for (size_t index = 0U; index < loads.size(); index++) {
      direct_loads.push_back(
          {
              loads[index].path,
              static_cast<NSUInteger>(slots[index]) * record.stride,
              loads[index].file,
          }
      );
    }
    timer.start_wall("Metal tile I/O");
    load_metal_tiles_into_buffer(
        device,
        io_queue,
        direct_loads,
        0U,
        record.logical_size,
        vertex_atlas,
        vertex_atlas.length
    );
    timer.stop("Metal tile I/O");
    return;
  }
  if (header_template.sample_type != MetalTileSampleType::Uint16Decimeters ||
      conversion_pipeline == nil || quantized_staging == nil) {
    throw std::logic_error("Fixed-point tile conversion resources are unavailable");
  }

  const QuantizedMetalTileRecordLayout record =
      quantized_metal_tile_record_layout(header_template);
  const uint32_t wave_capacity = static_cast<uint32_t>(metal_tile_io_concurrency());

  for (size_t wave_start = 0U; wave_start < loads.size(); wave_start += wave_capacity) {
    const size_t wave_size =
        std::min(static_cast<size_t>(wave_capacity), loads.size() - wave_start);
    std::vector<MetalTileBufferLoad> staged_loads;
    staged_loads.reserve(wave_size);
    for (size_t index = 0U; index < wave_size; index++) {
      const MetalTileBufferLoad &load = loads[wave_start + index];
      staged_loads.push_back(
          {
              load.path,
              static_cast<NSUInteger>(index * static_cast<size_t>(record.stride)),
              load.file,
          }
      );
    }

    timer.start_wall("Metal tile I/O");
    load_metal_tiles_into_buffer(
        device,
        io_queue,
        staged_loads,
        0U,
        record.logical_size,
        quantized_staging,
        quantized_staging.length
    );
    timer.stop("Metal tile I/O");

    const std::span<const uint32_t> wave_slots = slots.subspan(wave_start, wave_size);
    id<MTLBuffer> slot_buffer = [device newBufferWithBytes:wave_slots.data()
                                                    length:wave_slots.size_bytes()
                                                   options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [mipmap_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (slot_buffer == nil || command == nil || encoder == nil) {
      throw std::runtime_error("Could not create a fixed-point vertex-conversion command");
    }
    command.label = @"Convert fixed-point terrain vertices";
    encoder.label = @"Convert fixed-point terrain vertices";
    [encoder setComputePipelineState:conversion_pipeline];
    [encoder setBuffer:quantized_staging offset:0U atIndex:0];
    [encoder setBuffer:vertex_atlas offset:0U atIndex:1];
    [encoder setBuffer:slot_buffer offset:0U atIndex:2];
    [encoder setBytes:&record.stride length:sizeof(record.stride) atIndex:3];
    [encoder setBytes:&record.vertex_offset length:sizeof(record.vertex_offset) atIndex:4];
    [encoder setBytes:&record.elevation_base_offset
                length:sizeof(record.elevation_base_offset)
               atIndex:5];
    [encoder setBytes:&vertex_count length:sizeof(vertex_count) atIndex:6];
    const uint32_t tile_count = static_cast<uint32_t>(wave_size);
    [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(vertex_count, tile_count, 1U)
        threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
    [encoder endEncoding];

    timer.start_wall("GPU vertex conversion");
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      timer.stop("GPU vertex conversion");
      print_error(@"Metal vertex conversion failed", command.error);
      throw std::runtime_error("Metal fixed-point vertex conversion failed");
    }
    timer.add_work(
        "GPU vertex conversion",
        1'000.0 * (command.GPUEndTime - command.GPUStartTime)
    );
    timer.stop("GPU vertex conversion");
  }
}

MipmapSubmission
ResidentTileCache::State::submit_mipmaps(std::span<const uint32_t> slots) {
  if (slots.empty()) {
    throw std::invalid_argument("Cannot submit an empty mipmap-generation batch");
  }
  if (slots.size() > slot_capacity) {
    throw std::logic_error("Mipmap-generation batch exceeds its slot buffer");
  }

  // The slot list must remain available until this command completes.
  id<MTLBuffer> slot_buffer = [device newBufferWithBytes:slots.data()
                                                  length:slots.size_bytes()
                                                 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command = [mipmap_queue commandBuffer];
  if (slot_buffer == nil || command == nil) {
    throw std::runtime_error("Could not create a mipmap-generation command");
  }
  command.label = @"Generate resident tile mipmaps";

  // Generate level 1 for every tile from adjacent vertices, then reduce each
  // preceding level in turn. Ending each encoder supplies the global
  // dependency between dispatches; no threadgroup-local barrier would be
  // sufficient. The Z dimension batches slots without changing level order.
  uint32_t source_side = header_template.cell_count + 1U;
  uint32_t source_step = 1U;
  const QuantizedMetalTileRecordLayout quantized_record =
      retain_quantized ? quantized_metal_tile_record_layout(header_template)
                       : QuantizedMetalTileRecordLayout{};
  uint32_t source_tile_stride = retain_quantized ? quantized_record.stride / 2U : vertex_count;
  uint32_t destination_tile_stride = mip_count;
  const uint32_t tile_count = static_cast<uint32_t>(slots.size());
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Could not encode maximum mipmap level 1");
  }
  encoder.label = @"Build maximum mipmap level 1";
  [encoder setComputePipelineState:mipmap_pipeline];
  [encoder setBuffer:vertex_atlas
               offset:retain_quantized ? quantized_record.vertex_offset : 0U
              atIndex:0];
  [encoder setBuffer:mipmap_atlas offset:0U atIndex:1];
  [encoder setBytes:&source_side length:sizeof(source_side) atIndex:2];
  [encoder setBytes:&source_step length:sizeof(source_step) atIndex:3];
  [encoder setBuffer:slot_buffer offset:0U atIndex:4];
  [encoder setBytes:&source_tile_stride length:sizeof(source_tile_stride) atIndex:5];
  [encoder setBytes:&destination_tile_stride length:sizeof(destination_tile_stride) atIndex:6];
  [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:7];
  const uint32_t cell_count = header_template.cell_count;
  [encoder dispatchThreads:MTLSizeMake(cell_count, cell_count, tile_count)
      threadsPerThreadgroup:MTLSizeMake(32U, 8U, 1U)];
  [encoder endEncoding];

  size_t previous_offset = 0U;
  size_t output_offset = static_cast<size_t>(cell_count) * cell_count;
  source_side = cell_count;
  source_step = 2U;
  for (uint32_t level = 2U; level <= header_template.level_count; level++) {
    const uint32_t output_side = source_side / 2U;
    encoder = [command computeCommandEncoder];
    if (encoder == nil) {
      throw std::runtime_error("Could not encode a maximum mipmap reduction level");
    }
    encoder.label = @"Reduce maximum mipmap level";
    [encoder setComputePipelineState:mipmap_pipeline];
    const size_t sample_size = retain_quantized ? sizeof(uint16_t) : sizeof(float);
    [encoder setBuffer:mipmap_atlas offset:previous_offset * sample_size atIndex:0];
    [encoder setBuffer:mipmap_atlas offset:output_offset * sample_size atIndex:1];
    [encoder setBytes:&source_side length:sizeof(source_side) atIndex:2];
    [encoder setBytes:&source_step length:sizeof(source_step) atIndex:3];
    [encoder setBuffer:slot_buffer offset:0U atIndex:4];
    [encoder setBytes:&destination_tile_stride length:sizeof(destination_tile_stride) atIndex:5];
    [encoder setBytes:&destination_tile_stride length:sizeof(destination_tile_stride) atIndex:6];
    [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(output_side, output_side, tile_count)
        threadsPerThreadgroup:MTLSizeMake(32U, 8U, 1U)];
    [encoder endEncoding];

    previous_offset = output_offset;
    output_offset += static_cast<size_t>(output_side) * output_side;
    source_side = output_side;
  }
  if (output_offset != mip_count || source_side != 1U) {
    throw std::logic_error("GPU maximum mipmap layout calculation failed");
  }

  [command commit];
  return {slot_buffer, command};
}

void ResidentTileCache::State::generate_mipmaps(std::span<const uint32_t> slots, Timer &timer) {
  timer.start_wall("GPU mipmap generation");
  const MipmapSubmission submission = submit_mipmaps(slots);
  [submission.command waitUntilCompleted];
  timer.stop("GPU mipmap generation");
  if (submission.command.status == MTLCommandBufferStatusError) {
    print_error(@"Metal mipmap generation failed", submission.command.error);
    throw std::runtime_error("Metal mipmap generation failed");
  }
  timer.add_work(
      "GPU mipmap generation",
      1'000.0 * (submission.command.GPUEndTime - submission.command.GPUStartTime)
  );
}

void ResidentTileCache::State::rebuild_hash(Timer *timer) {
  if (timer != nullptr) {
    timer->start_wall("Resident hash rebuild");
  }
  std::fill_n(hash, hash_slot_count, ResidentTileHashEntry{});
  const uint32_t mask = hash_slot_count - 1U;
  for (uint32_t slot = 0U; slot < slot_capacity; slot++) {
    const uint32_t source = source_by_slot[slot];
    if (source == sources.size()) {
      continue;
    }
    const TileKey key = sources[source].key;
    uint32_t index = static_cast<uint32_t>(tile_key_hash(key)) & mask;
    while (hash[index].occupied != 0U) {
      // The table is at most half full, keeping linear probes short.
      index = (index + 1U) & mask;
    }
    hash[index] = {key.row, key.column, slot, 1U};
  }
  if (timer != nullptr) {
    timer->stop("Resident hash rebuild");
  }
}

ResidentTileCache::ResidentTileCache(
    id<MTLDevice> device,
    std::span<const TerrainSource> sources,
    const LoadedTile &origin,
    TileKey origin_key,
    const RaytraceConfig &config,
    uint32_t slot_capacity,
    Timer &timer
) {
  const bool custom_origin = !sources.empty() && is_metal_tile_path(sources.front().path);
  if (sources.empty() || slot_capacity == 0U || (!custom_origin && origin.vertices == nullptr)) {
    throw std::invalid_argument("Resident tile cache requires an origin tile and slots");
  }
  const uint32_t mip_count = static_cast<uint32_t>(metal_tile_mipmap_value_count(origin.size));
  const uint64_t vertex_side = static_cast<uint64_t>(origin.size) + 1U;
  const uint32_t vertex_count = static_cast<uint32_t>(vertex_side * vertex_side);
  const uint32_t hash_slot_count = resident_hash_capacity(slot_capacity);
  const uint64_t tile_bytes = static_cast<uint64_t>(mip_count + vertex_count) * sizeof(float);
  const double tile_width = static_cast<double>(origin.size) * origin.delta;
  const double grid_origin_x =
      origin.lower_left_x - static_cast<double>(origin_key.column) * tile_width;
  const double grid_origin_y =
      origin.lower_left_y + static_cast<double>(origin_key.row + 1) * tile_width;
  const MetalTileHeader header_template = custom_origin
                                               ? read_metal_tile_header(sources.front().path)
                                               : MetalTileHeader{
                                                     kMetalTileFloat32Magic,
                                                     kMetalTileFloat32Version,
                                                     kMetalTileFloat32HeaderSize,
                                                     MetalTileCompression::None,
                                                     origin.crs.epsg_code(),
                                                     origin.size,
                                                     origin.num_levels,
                                                     origin.maximum_elevation,
                                                     MetalTileSampleType::Float32,
                                                     0,
                                                     0U,
                                                     origin_key.row,
                                                     origin_key.column,
                                                     origin.lower_left_x,
                                                     origin.lower_left_y,
                                                     origin.delta,
                                                     kMetalTileFloat32HeaderSize,
                                                     static_cast<uint64_t>(vertex_count) *
                                                         sizeof(float),
                                                 };
  const bool retain_quantized = config.retain_quantized;
  if (retain_quantized &&
      (!custom_origin || header_template.sample_type != MetalTileSampleType::Uint16Decimeters)) {
    throw std::invalid_argument("Quantized atlas retention requires uint16 custom terrain");
  }
  const QuantizedMetalTileRecordLayout quantized_record =
      retain_quantized ? quantized_metal_tile_record_layout(header_template)
                       : QuantizedMetalTileRecordLayout{};

  // Each slot has the same payload length. Metal derives an address from a
  // slot index and this stride, so a work item never needs per-tile offsets.
  auto state = std::make_unique<State>(
      sources,
      config,
      device,
      header_template,
      retain_quantized,
      grid_origin_x,
      grid_origin_y,
      tile_width,
      mip_count,
      vertex_count,
      slot_capacity,
      hash_slot_count,
      custom_origin ? 0U : tile_bytes
  );
  // Shared storage supports both CPU GeoTIFF copies and direct Metal I/O.
  // Installation occurs only between completed frontier commands, and the
  // host waits for I/O before mipmap generation and for mipmaps before tracing.
  // Those explicit ordering points make whole-resource hazard tracking
  // unnecessary for these fixed, non-overlapping slot ranges.
  constexpr MTLResourceOptions kAtlasOptions =
      MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
  state->mipmap_atlas = make_buffer(
      device,
      checked_buffer_length(
          static_cast<size_t>(slot_capacity) * mip_count,
          retain_quantized ? sizeof(uint16_t) : sizeof(float),
          "mipmap atlas"
      ),
      "mipmap atlas",
      kAtlasOptions
  );
  state->vertex_atlas = make_buffer(
      device,
      retain_quantized
          ? checked_buffer_length(slot_capacity, quantized_record.stride, "vertex atlas")
          : checked_buffer_length(
                static_cast<size_t>(slot_capacity) * vertex_count,
                sizeof(float),
                "vertex atlas"
            ),
      "vertex atlas",
      kAtlasOptions
  );
  state->metadata_buffer = make_buffer(
      device,
      checked_buffer_length(slot_capacity, sizeof(ResidentTile), "tile metadata"),
      "tile metadata"
  );
  state->hash_buffer = make_buffer(
      device,
      checked_buffer_length(hash_slot_count, sizeof(ResidentTileHashEntry), "resident tile hash"),
      "resident tile hash"
  );
  const bool needs_metal_io =
      std::any_of(sources.begin(), sources.end(), [](const TerrainSource &source) {
        return is_metal_tile_path(source.path);
      });
  if (needs_metal_io) {
    state->io_queue = make_metal_io_queue(device);

    // Custom files contain atlas-ordered vertices. Both representations need
    // GPU mipmap reduction; the default fixed-point path additionally converts
    // vertices, while retained fixed-point records stay uint16 throughout.
    NSError *error = nil;
    NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      throw std::runtime_error("Could not load Metal library for terrain preparation");
    }
    NSString *mipmap_name = retain_quantized ? @"build_quantized_maximum_mipmap_level"
                                             : @"build_maximum_mipmap_level";
    id<MTLFunction> mipmap_function = [library newFunctionWithName:mipmap_name];
    if (mipmap_function == nil) {
      throw std::runtime_error("Metal terrain-preparation kernel is missing");
    }
    state->mipmap_pipeline =
        [device newComputePipelineStateWithFunction:mipmap_function error:&error];
    if (state->mipmap_pipeline == nil) {
      print_error(@"Could not create the mipmap-generation pipeline", error);
      throw std::runtime_error("Could not create the mipmap-generation pipeline");
    }
    state->mipmap_queue = [device newCommandQueue];
    if (state->mipmap_queue == nil) {
      throw std::runtime_error("Could not create the mipmap-generation command queue");
    }

    if (state->header_template.sample_type == MetalTileSampleType::Uint16Decimeters &&
        !retain_quantized) {
      id<MTLFunction> conversion_function =
          [library newFunctionWithName:@"convert_quantized_vertices"];
      if (conversion_function == nil) {
        throw std::runtime_error("Metal vertex-conversion kernel is missing");
      }
      state->conversion_pipeline =
          [device newComputePipelineStateWithFunction:conversion_function error:&error];
      if (state->conversion_pipeline == nil) {
        print_error(@"Could not create the vertex-conversion pipeline", error);
        throw std::runtime_error("Could not create the vertex-conversion pipeline");
      }

      const QuantizedMetalTileRecordLayout record =
          quantized_metal_tile_record_layout(state->header_template);
      const size_t staging_tiles = std::min(
          static_cast<size_t>(slot_capacity),
          static_cast<size_t>(metal_tile_io_concurrency())
      );
      state->quantized_staging = make_buffer(
          device,
          checked_buffer_length(
              staging_tiles,
              static_cast<size_t>(record.stride),
              "tile staging"
          ),
          "quantized tile staging"
      );
    }
  }
  state->mipmaps = static_cast<float *>(state->mipmap_atlas.contents);
  state->vertices = static_cast<float *>(state->vertex_atlas.contents);
  state->metadata = static_cast<ResidentTile *>(state->metadata_buffer.contents);
  state->hash = static_cast<ResidentTileHashEntry *>(state->hash_buffer.contents);
  if (state->mipmaps == nullptr || state->vertices == nullptr || state->metadata == nullptr ||
      state->hash == nullptr) {
    throw std::runtime_error("Could not map resident terrain atlas");
  }

  // Slot zero permanently starts as the observer tile, so the first frontier
  // can run before any asynchronous source has finished preparation. Custom
  // vertices take the same Metal I/O and GPU-reduction path as later tiles.
  if (custom_origin) {
    const MetalTileBufferLoad load = {sources.front().path, 0U, nil};
    const uint32_t slot = 0U;
    state->load_custom_vertices(
        std::span<const MetalTileBufferLoad>(&load, 1U),
        std::span<const uint32_t>(&slot, 1U),
        timer
    );
    state->generate_mipmaps(std::span<const uint32_t>(&slot, 1U), timer);
    state->bytes_loaded_with_metal_io =
        retain_quantized ? quantized_record.logical_size : state->header_template.vertex_byte_count;
  } else {
    if (origin.mipmap.size() != mip_count || origin.vertices->size() != vertex_count) {
      throw std::logic_error("GeoTIFF origin tile does not match the atlas dimensions");
    }
    std::memcpy(
        state->mipmaps,
        origin.mipmap.data(),
        static_cast<size_t>(mip_count) * sizeof(float)
    );
    std::memcpy(
        state->vertices,
        origin.vertices->data(),
        static_cast<size_t>(vertex_count) * sizeof(float)
    );
  }
  state->metadata[0] = make_resident_tile(origin, origin_key, config);
  state->slot_by_source.assign(sources.size(), slot_capacity);
  state->slot_by_source[0] = 0U;
  state->source_by_slot.assign(slot_capacity, static_cast<uint32_t>(sources.size()));
  state->source_by_slot[0] = 0U;
  state->last_used.assign(slot_capacity, 0U);
  state->last_used[0] = 1U;
  state_ = std::move(state);

  // Slot zero must be visible to the first frontier pass before any workers run.
  state_->rebuild_hash(nullptr);
}

ResidentTileCache::~ResidentTileCache() = default;

uint32_t ResidentTileCache::slot_for_source(uint32_t source_index) const {
  return state_->slot_by_source.at(source_index);
}

void ResidentTileCache::install_prepared(
    AsyncTilePreparer &preparer,
    std::span<const uint8_t> pinned_slots,
    Timer &timer
) {
  State &state = *state_;
  if (pinned_slots.size() != state.slot_capacity) {
    throw std::invalid_argument("Resident pin mask has the wrong size");
  }
  preparer.rethrow_if_failed();
  timer.start_wall("Atlas installation");

  // Protect both imminent frontier slots and every destination selected in
  // this call. A prepared queue can contain more tiles than free atlas slots;
  // without this mask the LRU search could select one destination repeatedly.
  std::vector<uint8_t> unavailable_slots(pinned_slots.begin(), pinned_slots.end());
  std::vector<AtlasInstallation> installations;
  installations.reserve(state.slot_capacity);

  while (true) {
    // Prefer an unused slot. Once full, choose the least-recently-used slot
    // that is neither pinned nor already selected by this installation batch.
    uint32_t slot = state.slot_capacity;
    for (uint32_t candidate = 0U; candidate < state.slot_capacity; candidate++) {
      if (state.source_by_slot[candidate] == state.sources.size() &&
          unavailable_slots[candidate] == 0U) {
        slot = candidate;
        break;
      }
    }
    if (slot == state.slot_capacity) {
      uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
      for (uint32_t candidate = 0U; candidate < state.slot_capacity; candidate++) {
        if (unavailable_slots[candidate] == 0U && state.last_used[candidate] < oldest_use) {
          slot = candidate;
          oldest_use = state.last_used[candidate];
        }
      }
    }
    if (slot == state.slot_capacity) {
      // Every resident tile is needed immediately. Leave completed CPU tiles
      // in the preparer's bounded hand-off queue until eviction becomes safe.
      break;
    }
    std::optional<PreparedTile> prepared = preparer.try_take_prepared();
    if (!prepared.has_value()) {
      break;
    }

    // Do not allow a later prepared tile to replace this one before the host
    // activates its deferred work and submits the next frontier pass.
    unavailable_slots[slot] = 1U;

    const TerrainSource &source = state.sources[prepared->source_index];
    const double lower_left_x =
        state.grid_origin_x + static_cast<double>(source.key.column) * state.tile_width;
    const double lower_left_y =
        state.grid_origin_y - static_cast<double>(source.key.row + 1) * state.tile_width;
    installations.push_back({slot, std::move(*prepared), lower_left_x, lower_left_y});
  }

  // GeoTIFFs already have CPU-resident payloads. Custom files instead provide
  // Metal I/O requests and slot IDs for synchronous GPU preparation.
  std::vector<MetalTileBufferLoad> custom_loads;
  std::vector<uint32_t> custom_slots;
  custom_loads.reserve(installations.size());
  custom_slots.reserve(installations.size());

  for (AtlasInstallation &installation : installations) {
    const TerrainSource &source = state.sources[installation.prepared.source_index];
    if (installation.prepared.tile != nullptr) {
      // GeoTIFF preparation produces host vectors, so copy both immutable
      // payloads into matching fixed-stride ranges in the shared atlas.
      timer.start_wall("Atlas copy");
      std::memcpy(
          state.mipmaps + static_cast<size_t>(installation.slot) * state.mip_count,
          installation.prepared.tile->mipmap.data(),
          static_cast<size_t>(state.mip_count) * sizeof(float)
      );
      std::memcpy(
          state.vertices + static_cast<size_t>(installation.slot) * state.vertex_count,
          installation.prepared.tile->vertices->data(),
          static_cast<size_t>(state.vertex_count) * sizeof(float)
      );
      timer.stop("Atlas copy");
      state.metadata[installation.slot] =
          make_resident_tile(*installation.prepared.tile, source.key, state.config);
      state.bytes_copied +=
          static_cast<uint64_t>(state.mip_count + state.vertex_count) * sizeof(float);
    } else {
      // Float32 payloads load directly into their final atlas slots. Fixed-point
      // records are either retained intact or staged and converted to Float32.
      custom_loads.push_back({source.path, 0U, installation.prepared.metal_file});
      custom_slots.push_back(installation.slot);
      state.metadata[installation.slot] = make_resident_tile(
          installation.lower_left_x,
          installation.lower_left_y,
          source.key,
          state.config
      );
    }
  }

  if (!custom_loads.empty()) {
    // Metal I/O loads the selected representation before mipmap generation
    // reads the new vertex slots.
    state.load_custom_vertices(custom_loads, custom_slots, timer);

    state.generate_mipmaps(custom_slots, timer);
    const uint64_t bytes_per_tile = state.retain_quantized
                                        ? quantized_metal_tile_record_layout(state.header_template)
                                              .logical_size
                                        : state.header_template.vertex_byte_count;
    state.bytes_loaded_with_metal_io +=
        static_cast<uint64_t>(custom_slots.size()) * bytes_per_tile;
  }

  // All payload writes are now complete. Publish slot mappings together so
  // the rebuilt hash cannot expose a partially installed tile.
  for (const AtlasInstallation &installation : installations) {
    const uint32_t slot = installation.slot;
    const uint32_t source = installation.prepared.source_index;
    const uint32_t evicted = state.source_by_slot[slot];
    if (evicted != state.sources.size()) {
      state.slot_by_source[evicted] = state.slot_capacity;
      preparer.mark_evicted(evicted);
      state.evictions++;
    } else {
      state.resident_count++;
    }

    state.slot_by_source[source] = slot;
    state.source_by_slot[slot] = source;
    state.last_used[slot] = state.next_use_stamp++;
    state.installations++;
    preparer.mark_resident(source);
  }

  if (!installations.empty()) {
    state.rebuild_hash(&timer);
  }
  timer.stop("Atlas installation");
}

std::vector<uint8_t>
ResidentTileCache::pin_slots(std::span<const uint32_t> slots, bool record_use) {
  State &state = *state_;
  std::vector<uint8_t> pinned(state.slot_capacity, 0U);
  for (uint32_t slot : slots) {
    if (slot >= state.slot_capacity || state.source_by_slot[slot] == state.sources.size()) {
      throw std::logic_error("GPU frontier refers to a nonresident tile slot");
    }
    pinned[slot] = 1U;
    if (record_use) {
      state.last_used[slot] = state.next_use_stamp++;
    }
  }
  return pinned;
}

ResidentTileCacheBindings ResidentTileCache::bindings() const {
  const State &state = *state_;
  const QuantizedMetalTileRecordLayout record =
      state.retain_quantized ? quantized_metal_tile_record_layout(state.header_template)
                             : QuantizedMetalTileRecordLayout{};
  return {
      state.mipmap_atlas,
      state.vertex_atlas,
      state.metadata_buffer,
      state.hash_buffer,
      state.hash_slot_count,
      {record.stride, record.vertex_offset, record.elevation_base_offset},
  };
}

uint32_t ResidentTileCache::slot_capacity() const { return state_->slot_capacity; }
uint32_t ResidentTileCache::resident_tile_count() const { return state_->resident_count; }

ResidentTileCacheStatistics ResidentTileCache::statistics() const {
  const State &state = *state_;
  return {state.installations,
          state.bytes_copied,
          state.bytes_loaded_with_metal_io,
          state.evictions,
          state.resident_count,
          state.slot_capacity};
}

} // namespace panorama
