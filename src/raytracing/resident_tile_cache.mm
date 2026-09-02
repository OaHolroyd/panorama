#include "resident_tile_cache.h"

#import <Foundation/Foundation.h>

#include "metal_tile.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

#ifndef PANORAMA_METALLIB_PATH
#define PANORAMA_METALLIB_PATH "obj/release/raytracing/panorama.metallib"
#endif

constexpr const char *kMetallibPath = PANORAMA_METALLIB_PATH;

static_assert(sizeof(ResidentTile) == 4U * sizeof(uint64_t));
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

/// Build resident metadata directly from projected tile coordinates.
[[nodiscard]] ResidentTile make_resident_tile(
    double lower_left_x,
    double lower_left_y,
    float maximum_elevation,
    TileKey key,
    uint32_t lod,
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
  return {static_cast<float>(x),
          static_cast<float>(y),
          maximum_elevation,
          lod,
          key.row,
          key.column};
}

/// One prepared source paired with its selected destination atlas slot.
struct AtlasInstallation {
  uint32_t slot;
  PreparedTile prepared;
  double lower_left_x;
  double lower_left_y;
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
  id<MTLComputePipelineState> initial_mipmap_pipeline;
  id<MTLComputePipelineState> mipmap_pipeline;
  MetalTileHeader header_template;
  QuantizedMetalTileRecordLayout quantized_record;
  bool retain_quantized;
  double grid_origin_x;
  double grid_origin_y;
  double tile_width;
  uint32_t mip_count;
  uint32_t vertex_count;
  uint32_t slot_capacity;
  uint32_t resident_count = 1U;
  uint64_t next_use_stamp = 2U;
  uint64_t installations = 1U;
  uint64_t bytes_loaded_with_metal_io = 0U;
  uint64_t evictions = 0U;
  id<MTLBuffer> mipmap_atlas;
  id<MTLBuffer> vertex_atlas;
  id<MTLBuffer> quantized_staging;
  id<MTLBuffer> preparation_slots;
  id<MTLBuffer> metadata_buffer;
  ResidentTile *metadata;
  std::map<TileVariant, uint32_t> slot_by_variant;
  std::vector<std::optional<TileVariant>> variant_by_slot;
  std::vector<uint64_t> last_used;

  /// Submit every maximum-mipmap level for the supplied resident slots.
  [[nodiscard]] id<MTLCommandBuffer>
  submit_mipmaps(std::span<const uint32_t> slots, uint32_t cell_count, uint32_t level_count);

  /// Copy one synchronous preparation batch into the reusable GPU slot list.
  void write_preparation_slots(std::span<const uint32_t> slots);

  /// Load custom payloads directly, retaining or expanding fixed-point records.
  void load_custom_vertices(
      std::span<const MetalTileBufferLoad> loads,
      std::span<const uint32_t> slots,
      std::span<const int32_t> elevation_bases,
      uint32_t vertex_value_count,
      Timer &timer
  );

  /// Work around compressed Metal I/O implementations which cannot begin a
  /// load in the small header/table prefix of a compressed LOD stream.
  void load_compressed_lod_ranges(
      std::span<const MetalTileBufferLoad> loads,
      std::span<const NSUInteger> destination_offsets,
      id<MTLBuffer> destination,
      Timer &timer
  );

  /// Build mipmaps synchronously where the observer tile requires them now.
  void generate_mipmaps(
      std::span<const uint32_t> slots,
      uint32_t cell_count,
      uint32_t level_count,
      Timer &timer
  );

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
      uint32_t slot_values
  )
      : sources(source_values), config(config_value), device(device_value),
        header_template(header_value),
        quantized_record(
            header_value.sample_type == MetalTileSampleType::Uint16Decimeters
                ? quantized_metal_tile_record_layout(header_value)
                : QuantizedMetalTileRecordLayout{}
        ),
        retain_quantized(retain_quantized_value), grid_origin_x(origin_x), grid_origin_y(origin_y),
        tile_width(width), mip_count(mip_values), vertex_count(vertex_values),
        slot_capacity(slot_values) {}
};

void ResidentTileCache::State::write_preparation_slots(std::span<const uint32_t> slots) {
  if (slots.empty() || slots.size() > slot_capacity) {
    throw std::invalid_argument("Terrain-preparation batch exceeds its slot buffer");
  }
  auto *destination = static_cast<uint32_t *>(preparation_slots.contents);
  if (destination == nullptr) {
    throw std::runtime_error("Could not map terrain-preparation slot buffer");
  }
  std::copy(slots.begin(), slots.end(), destination);
}

void ResidentTileCache::State::load_custom_vertices(
    std::span<const MetalTileBufferLoad> loads,
    std::span<const uint32_t> slots,
    std::span<const int32_t> elevation_bases,
    uint32_t vertex_value_count,
    Timer &timer
) {
  if (loads.empty() || loads.size() != slots.size() || loads.size() != elevation_bases.size()) {
    throw std::invalid_argument("Metal tile loads require one destination slot per file");
  }
  if (io_queue == nil) {
    throw std::logic_error("Metal tile loading resources are unavailable");
  }

  if (header_template.sample_type == MetalTileSampleType::Float32 || retain_quantized) {
    std::vector<MetalTileBufferLoad> direct_loads;
    std::vector<MetalTileBufferLoad> prefixed_loads;
    std::vector<NSUInteger> prefixed_destinations;
    direct_loads.reserve(loads.size());
    prefixed_loads.reserve(loads.size());
    prefixed_destinations.reserve(loads.size());
    for (size_t index = 0U; index < loads.size(); index++) {
      MetalTileBufferLoad load = loads[index];
      load.destination_offset +=
          retain_quantized
              ? static_cast<NSUInteger>(slots[index]) * quantized_record.stride +
                    quantized_record.vertex_offset
              : static_cast<NSUInteger>(slots[index]) * header_template.vertex_byte_count;
      const size_t compression_chunk = metal_tile_compression_chunk_size();
      if (metal_tile_compression(load.path) != MetalTileCompression::None &&
          load.source_offset % compression_chunk != 0U) {
        prefixed_destinations.push_back(load.destination_offset);
        prefixed_loads.push_back(load);
      } else {
        direct_loads.push_back(load);
      }
      if (retain_quantized) {
        auto *record = static_cast<std::byte *>(vertex_atlas.contents) +
                       static_cast<size_t>(slots[index]) * quantized_record.stride;
        std::memcpy(
            record + quantized_record.elevation_base_offset,
            &elevation_bases[index],
            sizeof(int32_t)
        );
      }
    }
    if (!direct_loads.empty()) {
      timer.start_wall("Metal tile I/O");
      load_metal_tiles_into_buffer(
          device,
          io_queue,
          direct_loads,
          vertex_atlas,
          vertex_atlas.length
      );
      timer.stop("Metal tile I/O");
    }
    if (!prefixed_loads.empty()) {
      load_compressed_lod_ranges(prefixed_loads, prefixed_destinations, vertex_atlas, timer);
    }
    return;
  }
  if (header_template.sample_type != MetalTileSampleType::Uint16Decimeters ||
      conversion_pipeline == nil || quantized_staging == nil) {
    throw std::logic_error("Fixed-point tile conversion resources are unavailable");
  }

  const uint32_t wave_capacity = static_cast<uint32_t>(metal_tile_io_concurrency());

  for (size_t wave_start = 0U; wave_start < loads.size(); wave_start += wave_capacity) {
    const size_t wave_size =
        std::min(static_cast<size_t>(wave_capacity), loads.size() - wave_start);
    std::vector<MetalTileBufferLoad> staged_loads;
    std::vector<MetalTileBufferLoad> prefixed_loads;
    std::vector<NSUInteger> prefixed_destinations;
    staged_loads.reserve(wave_size);
    for (size_t index = 0U; index < wave_size; index++) {
      const MetalTileBufferLoad &load = loads[wave_start + index];
      MetalTileBufferLoad staged = load;
      staged.destination_offset =
          static_cast<NSUInteger>(index) * quantized_record.stride + quantized_record.vertex_offset;
      const size_t compression_chunk = metal_tile_compression_chunk_size();
      if (metal_tile_compression(staged.path) != MetalTileCompression::None &&
          staged.source_offset % compression_chunk != 0U) {
        prefixed_destinations.push_back(staged.destination_offset);
        prefixed_loads.push_back(staged);
      } else {
        staged_loads.push_back(staged);
      }
      auto *record = static_cast<std::byte *>(quantized_staging.contents) +
                     static_cast<size_t>(index) * quantized_record.stride;
      std::memcpy(
          record + quantized_record.elevation_base_offset,
          &elevation_bases[wave_start + index],
          sizeof(int32_t)
      );
    }

    if (!staged_loads.empty()) {
      timer.start_wall("Metal tile I/O");
      load_metal_tiles_into_buffer(
          device,
          io_queue,
          staged_loads,
          quantized_staging,
          quantized_staging.length
      );
      timer.stop("Metal tile I/O");
    }
    if (!prefixed_loads.empty()) {
      load_compressed_lod_ranges(prefixed_loads, prefixed_destinations, quantized_staging, timer);
    }

    const std::span<const uint32_t> wave_slots = slots.subspan(wave_start, wave_size);
    write_preparation_slots(wave_slots);
    id<MTLCommandBuffer> command = [mipmap_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("Could not create a fixed-point vertex-conversion command");
    }
    command.label = @"Convert fixed-point terrain vertices";
    encoder.label = @"Convert fixed-point terrain vertices";
    [encoder setComputePipelineState:conversion_pipeline];
    [encoder setBuffer:quantized_staging offset:0U atIndex:0];
    [encoder setBuffer:vertex_atlas offset:0U atIndex:1];
    [encoder setBuffer:preparation_slots offset:0U atIndex:2];
    [encoder setBytes:&quantized_record.stride length:sizeof(quantized_record.stride) atIndex:3];
    [encoder setBytes:&quantized_record.vertex_offset
               length:sizeof(quantized_record.vertex_offset)
              atIndex:4];
    [encoder setBytes:&quantized_record.elevation_base_offset
               length:sizeof(quantized_record.elevation_base_offset)
              atIndex:5];
    [encoder setBytes:&vertex_value_count length:sizeof(vertex_value_count) atIndex:6];
    const uint32_t tile_count = static_cast<uint32_t>(wave_size);
    [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(vertex_value_count, tile_count, 1U)
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
    timer.add_work("GPU vertex conversion", 1'000.0 * (command.GPUEndTime - command.GPUStartTime));
    timer.stop("GPU vertex conversion");
  }
}

void ResidentTileCache::State::load_compressed_lod_ranges(
    std::span<const MetalTileBufferLoad> loads,
    std::span<const NSUInteger> destination_offsets,
    id<MTLBuffer> destination,
    Timer &timer
) {
  if (loads.empty() || loads.size() != destination_offsets.size() || destination == nil ||
      mipmap_queue == nil) {
    throw std::invalid_argument("Compressed LOD loading requires matching valid buffers");
  }

  // Metal I/O compressed handles normally translate a logical byte offset to
  // their compressed chunks. Some driver versions stall for the small offset
  // introduced by the LOD table, though. Read the required prefix from zero,
  // which is universally supported, then trim it with a GPU-local copy. This
  // preserves compatibility with already-generated unaligned LOD files.
  std::vector<id<MTLBuffer>> staging;
  staging.reserve(loads.size());
  for (const MetalTileBufferLoad &load : loads) {
    if (load.source_offset > std::numeric_limits<uint64_t>::max() - load.byte_count ||
        load.source_offset + load.byte_count >
            static_cast<uint64_t>(std::numeric_limits<NSUInteger>::max())) {
      throw std::overflow_error("Compressed Metal tile range exceeds NSUInteger");
    }
    const NSUInteger prefix_size = static_cast<NSUInteger>(load.source_offset + load.byte_count);
    id<MTLBuffer> buffer = [device newBufferWithLength:prefix_size
                                               options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      throw std::runtime_error("Could not allocate compressed LOD staging buffer");
    }
    const MetalTileBufferLoad prefixed = {
        load.path,
        0U,
        load.file,
        0U,
        static_cast<uint64_t>(prefix_size),
    };
    timer.start_wall("Metal tile I/O");
    load_metal_tiles_into_buffer(
        device,
        io_queue,
        std::span<const MetalTileBufferLoad>(&prefixed, 1U),
        buffer,
        buffer.length
    );
    timer.stop("Metal tile I/O");
    staging.push_back(buffer);
  }

  id<MTLCommandBuffer> command = [mipmap_queue commandBuffer];
  id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create compressed LOD copy command");
  }
  for (size_t index = 0U; index < loads.size(); index++) {
    const MetalTileBufferLoad &load = loads[index];
    [encoder copyFromBuffer:staging[index]
               sourceOffset:static_cast<NSUInteger>(load.source_offset)
                   toBuffer:destination
          destinationOffset:destination_offsets[index]
                       size:static_cast<NSUInteger>(load.byte_count)];
  }
  [encoder endEncoding];
  timer.start_wall("GPU compressed LOD copy");
  [command commit];
  [command waitUntilCompleted];
  timer.stop("GPU compressed LOD copy");
  if (command.status == MTLCommandBufferStatusError) {
    print_error(@"Compressed LOD copy failed", command.error);
    throw std::runtime_error("Could not trim compressed Metal tile LOD payload");
  }
}

id<MTLCommandBuffer> ResidentTileCache::State::submit_mipmaps(
    std::span<const uint32_t> slots,
    uint32_t cell_count,
    uint32_t level_count
) {
  // Mipmap submissions complete synchronously, so the cache-owned slot list
  // is never rewritten while a preparation command is using it.
  write_preparation_slots(slots);
  id<MTLCommandBuffer> command = [mipmap_queue commandBuffer];
  if (command == nil) {
    throw std::runtime_error("Could not create a mipmap-generation command");
  }
  command.label = @"Generate resident tile mipmaps";

  // Each initial thread builds a 2×2 group of level-1 cells and their level-2
  // parent from one shared 3×3 vertex patch. A one-cell format cannot use that
  // grouping and falls back to the generic level-1 dispatch.
  constexpr uint32_t fused_level_count = 2U;
  uint32_t source_tile_stride = retain_quantized ? quantized_record.stride / 2U : vertex_count;
  const uint32_t destination_tile_stride = mip_count;
  const uint32_t tile_count = static_cast<uint32_t>(slots.size());
  const bool fuse_initial_levels = cell_count >= 2U && level_count >= fused_level_count;
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Could not encode initial maximum mipmap levels");
  }
  encoder.label = fuse_initial_levels ? @"Build initial maximum mipmap levels"
                                      : @"Build maximum mipmap level 1";
  [encoder setComputePipelineState:fuse_initial_levels ? initial_mipmap_pipeline : mipmap_pipeline];
  [encoder setBuffer:vertex_atlas
              offset:retain_quantized ? quantized_record.vertex_offset : 0U
             atIndex:0];
  [encoder setBuffer:mipmap_atlas offset:0U atIndex:1];
  uint32_t source_side = fuse_initial_levels ? cell_count : cell_count + 1U;
  uint32_t source_step = 1U;
  [encoder setBytes:&source_side length:sizeof(source_side) atIndex:2];
  [encoder setBytes:&source_step length:sizeof(source_step) atIndex:3];
  [encoder setBuffer:preparation_slots offset:0U atIndex:4];
  [encoder setBytes:&source_tile_stride length:sizeof(source_tile_stride) atIndex:5];
  [encoder setBytes:&destination_tile_stride length:sizeof(destination_tile_stride) atIndex:6];
  [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:7];
  const uint32_t initial_output_side = fuse_initial_levels ? cell_count / 2U : cell_count;
  [encoder dispatchThreads:MTLSizeMake(initial_output_side, initial_output_side, tile_count)
      threadsPerThreadgroup:MTLSizeMake(32U, 8U, 1U)];
  [encoder endEncoding];

  size_t previous_offset = 0U;
  size_t output_offset = static_cast<size_t>(cell_count) * cell_count;
  uint32_t first_reduction_level = 2U;
  source_side = cell_count;
  if (fuse_initial_levels) {
    for (uint32_t level = 2U; level <= fused_level_count; level++) {
      previous_offset = output_offset;
      source_side /= 2U;
      output_offset += static_cast<size_t>(source_side) * source_side;
    }
    first_reduction_level = fused_level_count + 1U;
  }
  source_step = 2U;
  for (uint32_t level = first_reduction_level; level <= level_count; level++) {
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
    [encoder setBuffer:preparation_slots offset:0U atIndex:4];
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
  if (output_offset != metal_tile_mipmap_value_count(cell_count) || source_side != 1U) {
    throw std::logic_error("GPU maximum mipmap layout calculation failed");
  }

  [command commit];
  return command;
}

void ResidentTileCache::State::generate_mipmaps(
    std::span<const uint32_t> slots,
    uint32_t cell_count,
    uint32_t level_count,
    Timer &timer
) {
  timer.start_wall("GPU mipmap generation");
  id<MTLCommandBuffer> command = submit_mipmaps(slots, cell_count, level_count);
  [command waitUntilCompleted];
  timer.stop("GPU mipmap generation");
  if (command.status == MTLCommandBufferStatusError) {
    print_error(@"Metal mipmap generation failed", command.error);
    throw std::runtime_error("Metal mipmap generation failed");
  }
  timer.add_work("GPU mipmap generation", 1'000.0 * (command.GPUEndTime - command.GPUStartTime));
}

ResidentTileCache::ResidentTileCache(
    id<MTLDevice> device,
    std::span<const TerrainSource> sources,
    const TileGeometry &origin,
    TileKey origin_key,
    const RaytraceConfig &config,
    bool retain_quantized,
    uint32_t slot_capacity,
    Timer &timer
) {
  if (sources.empty() || slot_capacity == 0U) {
    throw std::invalid_argument("Resident tile cache requires an origin tile and slots");
  }
  const uint32_t mip_count =
      static_cast<uint32_t>(metal_tile_mipmap_value_count(origin.cell_count));
  const uint64_t vertex_side = static_cast<uint64_t>(origin.cell_count) + 1U;
  const uint32_t vertex_count = static_cast<uint32_t>(vertex_side * vertex_side);
  const double tile_width = static_cast<double>(origin.cell_count) * origin.cell_size;
  const double grid_origin_x =
      origin.lower_left_x - static_cast<double>(origin_key.column) * tile_width;
  const double grid_origin_y =
      origin.lower_left_y + static_cast<double>(origin_key.row + 1) * tile_width;
  const MetalTileHeader header_template = read_metal_tile_header(sources.front().path);
  if (retain_quantized && header_template.sample_type != MetalTileSampleType::Uint16Decimeters) {
    throw std::invalid_argument("Quantized atlas retention requires uint16 prepared terrain");
  }
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
      slot_capacity
  );
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
          ? checked_buffer_length(slot_capacity, state->quantized_record.stride, "vertex atlas")
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
  state->preparation_slots = make_buffer(
      device,
      checked_buffer_length(slot_capacity, sizeof(uint32_t), "terrain preparation slots"),
      "terrain preparation slots"
  );
  state->preparation_slots.label = @"Terrain preparation slots";
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
  NSString *initial_mipmap_name = retain_quantized
                                      ? @"build_quantized_initial_maximum_mipmap_levels"
                                      : @"build_initial_maximum_mipmap_levels";
  id<MTLFunction> initial_mipmap_function = [library newFunctionWithName:initial_mipmap_name];
  if (initial_mipmap_function == nil) {
    throw std::runtime_error("Metal terrain-preparation kernel is missing");
  }
  state->initial_mipmap_pipeline =
      [device newComputePipelineStateWithFunction:initial_mipmap_function error:&error];
  if (state->initial_mipmap_pipeline == nil) {
    print_error(@"Could not create the initial mipmap-generation pipeline", error);
    throw std::runtime_error("Could not create the initial mipmap-generation pipeline");
  }

  NSString *mipmap_name =
      retain_quantized ? @"build_quantized_maximum_mipmap_level" : @"build_maximum_mipmap_level";
  id<MTLFunction> mipmap_function = [library newFunctionWithName:mipmap_name];
  if (mipmap_function == nil) {
    throw std::runtime_error("Metal terrain-preparation kernel is missing");
  }
  state->mipmap_pipeline = [device newComputePipelineStateWithFunction:mipmap_function
                                                                 error:&error];
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
    state->conversion_pipeline = [device newComputePipelineStateWithFunction:conversion_function
                                                                       error:&error];
    if (state->conversion_pipeline == nil) {
      print_error(@"Could not create the vertex-conversion pipeline", error);
      throw std::runtime_error("Could not create the vertex-conversion pipeline");
    }

    const size_t staging_tiles = std::min(
        static_cast<size_t>(slot_capacity),
        static_cast<size_t>(metal_tile_io_concurrency())
    );
    state->quantized_staging = make_buffer(
        device,
        checked_buffer_length(
            staging_tiles,
            static_cast<size_t>(state->quantized_record.stride),
            "tile staging"
        ),
        "quantized tile staging"
    );
  }
  state->metadata = static_cast<ResidentTile *>(state->metadata_buffer.contents);
  if (state->metadata == nullptr) {
    throw std::runtime_error("Could not map resident terrain atlas");
  }

  // Slot zero permanently starts as the observer tile, so the first frontier
  // can run before any asynchronous source has finished preparation.
  const MetalTileBufferLoad load = {
      sources.front().path,
      0U,
      nil,
      state->header_template.vertex_offset,
      state->header_template.vertex_byte_count,
  };
  const uint32_t slot = 0U;
  const int32_t elevation_base = state->header_template.elevation_base_decimeters;
  state->load_custom_vertices(
      std::span<const MetalTileBufferLoad>(&load, 1U),
      std::span<const uint32_t>(&slot, 1U),
      std::span<const int32_t>(&elevation_base, 1U),
      vertex_count,
      timer
  );
  state->generate_mipmaps(
      std::span<const uint32_t>(&slot, 1U),
      state->header_template.cell_count,
      state->header_template.level_count,
      timer
  );
  state->bytes_loaded_with_metal_io = retain_quantized ? state->quantized_record.logical_size
                                                       : state->header_template.vertex_byte_count;
  state->metadata[0] = make_resident_tile(
      origin.lower_left_x,
      origin.lower_left_y,
      origin.maximum_elevation,
      origin_key,
      1U,
      config
  );
  state->slot_by_variant[{0U, 1U}] = 0U;
  state->variant_by_slot.assign(slot_capacity, std::nullopt);
  state->variant_by_slot[0] = TileVariant{0U, 1U};
  state->last_used.assign(slot_capacity, 0U);
  state->last_used[0] = 1U;
  state_ = std::move(state);
}

ResidentTileCache::~ResidentTileCache() = default;

uint32_t ResidentTileCache::slot_for_variant(TileVariant variant) const {
  if (variant.source_index >= state_->sources.size() || variant.lod == 0U) {
    throw std::invalid_argument("Resident terrain variant is invalid");
  }
  const auto found = state_->slot_by_variant.find(variant);
  return found == state_->slot_by_variant.end() ? state_->slot_capacity : found->second;
}

std::vector<TileVariant> ResidentTileCache::install_prepared(
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
      if (!state.variant_by_slot[candidate].has_value() && unavailable_slots[candidate] == 0U) {
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
      // Every resident tile is needed immediately. Leave completed sources in
      // the preparer's bounded hand-off queue until eviction becomes safe.
      break;
    }
    std::optional<PreparedTile> prepared = preparer.try_take_prepared();
    if (!prepared.has_value()) {
      break;
    }

    // Do not allow a later prepared tile to replace this one before the host
    // activates its deferred work and submits the next frontier pass.
    unavailable_slots[slot] = 1U;

    const TerrainSource &source = state.sources[prepared->variant.source_index];
    const double lower_left_x =
        state.grid_origin_x + static_cast<double>(source.key.column) * state.tile_width;
    const double lower_left_y =
        state.grid_origin_y - static_cast<double>(source.key.row + 1) * state.tile_width;
    installations.push_back({slot, std::move(*prepared), lower_left_x, lower_left_y});
  }

  std::vector<MetalTileBufferLoad> loads_by_variant;
  std::vector<uint32_t> slots_by_variant;
  std::vector<int32_t> elevation_bases_by_variant;
  std::map<uint32_t, std::vector<size_t>> load_indices_by_lod;
  loads_by_variant.reserve(installations.size());
  slots_by_variant.reserve(installations.size());
  elevation_bases_by_variant.reserve(installations.size());

  for (AtlasInstallation &installation : installations) {
    const TerrainSource &source = state.sources[installation.prepared.variant.source_index];
    const MetalTileLod &lod = installation.prepared.metal_lod;
    if (lod.lod != installation.prepared.variant.lod ||
        lod.cell_count != state.header_template.cell_count >> (lod.lod - 1U) ||
        lod.level_count != state.header_template.level_count - (lod.lod - 1U)) {
      throw std::runtime_error("Metal tile LOD does not match the resident atlas layout");
    }
    loads_by_variant.push_back(
        {source.path,
         0U,
         installation.prepared.metal_file,
         lod.vertex_offset,
         lod.vertex_byte_count}
    );
    slots_by_variant.push_back(installation.slot);
    elevation_bases_by_variant.push_back(lod.elevation_base_decimeters);
    load_indices_by_lod[installation.prepared.variant.lod].push_back(loads_by_variant.size() - 1U);
    state.metadata[installation.slot] = make_resident_tile(
        installation.lower_left_x,
        installation.lower_left_y,
        lod.maximum_elevation,
        source.key,
        installation.prepared.variant.lod,
        state.config
    );
  }

  if (!loads_by_variant.empty()) {
    // Metal I/O loads the selected representation before mipmap generation
    // reads the new vertex slots.
    for (const auto &[lod, indices] : load_indices_by_lod) {
      std::vector<MetalTileBufferLoad> loads;
      std::vector<uint32_t> slots;
      std::vector<int32_t> elevation_bases;
      loads.reserve(indices.size());
      slots.reserve(indices.size());
      elevation_bases.reserve(indices.size());
      for (size_t index : indices) {
        loads.push_back(loads_by_variant[index]);
        slots.push_back(slots_by_variant[index]);
        elevation_bases.push_back(elevation_bases_by_variant[index]);
      }
      const uint32_t cell_count = state.header_template.cell_count >> (lod - 1U);
      const uint32_t level_count = state.header_template.level_count - (lod - 1U);
      const uint32_t vertex_value_count = (cell_count + 1U) * (cell_count + 1U);
      state.load_custom_vertices(loads, slots, elevation_bases, vertex_value_count, timer);
      state.generate_mipmaps(slots, cell_count, level_count, timer);
    }
    for (const MetalTileBufferLoad &load : loads_by_variant) {
      state.bytes_loaded_with_metal_io += load.byte_count;
    }
  }

  // All payload writes are now complete. Publish the slot mappings together.
  for (const AtlasInstallation &installation : installations) {
    const uint32_t slot = installation.slot;
    const TileVariant variant = installation.prepared.variant;
    const std::optional<TileVariant> evicted = state.variant_by_slot[slot];
    if (evicted.has_value()) {
      state.slot_by_variant.erase(*evicted);
      preparer.mark_evicted(*evicted);
      state.evictions++;
    } else {
      state.resident_count++;
    }

    state.slot_by_variant[variant] = slot;
    state.variant_by_slot[slot] = variant;
    state.last_used[slot] = state.next_use_stamp++;
    state.installations++;
    preparer.mark_resident(variant);
  }

  timer.stop("Atlas installation");

  std::vector<TileVariant> installed_variants;
  installed_variants.reserve(installations.size());
  for (const AtlasInstallation &installation : installations) {
    installed_variants.push_back(installation.prepared.variant);
  }
  return installed_variants;
}

void ResidentTileCache::record_slot_use(std::span<const uint32_t> slots) {
  State &state = *state_;
  for (uint32_t slot : slots) {
    if (slot >= state.slot_capacity || !state.variant_by_slot[slot].has_value()) {
      throw std::logic_error("GPU frontier refers to a nonresident tile slot");
    }
    state.last_used[slot] = state.next_use_stamp++;
  }
}

void ResidentTileCache::rebase_observer(ObserverLocation observer) {
  State &state = *state_;
  if (!std::isfinite(observer.easting) || !std::isfinite(observer.northing)) {
    throw std::invalid_argument("Resident terrain rebase requires a finite observer");
  }
  for (uint32_t slot = 0U; slot < state.slot_capacity; slot++) {
    const std::optional<TileVariant> variant = state.variant_by_slot[slot];
    if (!variant.has_value()) {
      continue;
    }
    const TerrainSource &source = state.sources[variant->source_index];
    const double lower_left_x =
        state.grid_origin_x + static_cast<double>(source.key.column) * state.tile_width;
    const double lower_left_y =
        state.grid_origin_y - static_cast<double>(source.key.row + 1) * state.tile_width;
    const double relative_x = lower_left_x - observer.easting;
    const double relative_y = lower_left_y - observer.northing;
    if (relative_x < static_cast<double>(std::numeric_limits<float>::lowest()) ||
        relative_x > static_cast<double>(std::numeric_limits<float>::max()) ||
        relative_y < static_cast<double>(std::numeric_limits<float>::lowest()) ||
        relative_y > static_cast<double>(std::numeric_limits<float>::max())) {
      throw std::overflow_error("Rebased resident tile origin does not fit float32");
    }
    state.metadata[slot].tile_x_min = static_cast<float>(relative_x);
    state.metadata[slot].tile_y_min = static_cast<float>(relative_y);
  }
}

TileManagerBindings ResidentTileCache::bindings() const {
  const State &state = *state_;
  return {
      state.mipmap_atlas,
      state.vertex_atlas,
      state.metadata_buffer,
      state.retain_quantized
          ? QuantizedTerrainLayout{
                state.quantized_record.stride,
                state.quantized_record.vertex_offset,
                state.quantized_record.elevation_base_offset,
            }
          : QuantizedTerrainLayout{},
  };
}

uint32_t ResidentTileCache::slot_capacity() const { return state_->slot_capacity; }

TileManagerStatistics ResidentTileCache::statistics() const {
  const State &state = *state_;
  return {0U,
          0U,
          0U,
          0U,
          state.installations,
          state.bytes_loaded_with_metal_io,
          state.evictions,
          state.resident_count,
          state.slot_capacity};
}

} // namespace panorama
