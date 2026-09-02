#include "tile_manager_state.h"

#import <Foundation/Foundation.h>

#include "metal_tile.h"
#include "timer.h"

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
  /// Reserved destination which remains pinned for this installation batch.
  uint32_t slot;
  /// Worker-prepared file handle and selected LOD range.
  PreparedTile prepared;
  /// Absolute projected easting used to publish observer-relative metadata.
  double lower_left_x;
  /// Absolute projected northing used to publish observer-relative metadata.
  double lower_left_y;
};

} // namespace

void TileManager::State::write_preparation_slots(std::span<const uint32_t> slots) {
  if (slots.empty() || slots.size() > slot_capacity) {
    throw std::invalid_argument("Terrain-preparation batch exceeds its slot buffer");
  }
  auto *destination = static_cast<uint32_t *>(preparation_slots.contents);
  if (destination == nullptr) {
    throw std::runtime_error("Could not map terrain-preparation slot buffer");
  }
  std::copy(slots.begin(), slots.end(), destination);
}

void TileManager::State::load_custom_vertices(
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

  // Float32 sources and retained uint16 records already match their final
  // atlas representation. Split out only compressed, unaligned ranges which
  // require the compatibility staging path below.
  if (header_template.sample_type == MetalTileSampleType::Float32 || trace_quantized) {
    std::vector<MetalTileBufferLoad> direct_loads;
    std::vector<MetalTileBufferLoad> prefixed_loads;
    std::vector<NSUInteger> prefixed_destinations;
    direct_loads.reserve(loads.size());
    prefixed_loads.reserve(loads.size());
    prefixed_destinations.reserve(loads.size());
    for (size_t index = 0U; index < loads.size(); index++) {
      MetalTileBufferLoad load = loads[index];
      load.destination_offset +=
          trace_quantized
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
      if (trace_quantized) {
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

  // Expanded uint16 data first enters a small packed staging area. Process no
  // more records per wave than Metal I/O can keep in flight, then convert each
  // vertex directly into its final Float32 slot on the GPU.
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

void TileManager::State::load_compressed_lod_ranges(
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

id<MTLCommandBuffer> TileManager::State::submit_mipmaps(
    std::span<const uint32_t> slots,
    uint32_t cell_count,
    uint32_t level_count
) {
  // Mipmap submissions complete synchronously, so the manager-owned slot list
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
  uint32_t source_tile_stride = trace_quantized ? quantized_record.stride / 2U : vertex_count;
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
              offset:trace_quantized ? quantized_record.vertex_offset : 0U
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
    const size_t sample_size = trace_quantized ? sizeof(uint16_t) : sizeof(float);
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

void TileManager::State::generate_mipmaps(
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

void TileManager::State::attach_atlas(
    id<MTLDevice> device_value,
    uint32_t slot_capacity_value,
    Timer &timer_value
) {
  if (atlas_attached || device_value == nil) {
    throw std::logic_error("TileManager atlas is already attached or has no Metal device");
  }
  const std::span<const TerrainSource> sources = catalogue->sources();
  const TileGeometry &origin_tile = *origin;
  const TileKey origin_key = catalogue->origin().key;
  const bool retain = trace_quantized;
  id<MTLDevice> metal_device = device_value;
  const uint32_t capacity = slot_capacity_value;
  Timer &timer = timer_value;
  if (sources.empty() || capacity == 0U) {
    throw std::invalid_argument("TileManager requires an origin tile and atlas slots");
  }
  const uint32_t mip_values =
      static_cast<uint32_t>(metal_tile_mipmap_value_count(origin_tile.cell_count));
  const uint64_t vertex_side = static_cast<uint64_t>(origin_tile.cell_count) + 1U;
  const uint32_t vertex_values = static_cast<uint32_t>(vertex_side * vertex_side);
  const double width = static_cast<double>(origin_tile.cell_count) * origin_tile.cell_size;
  const double origin_x = origin_tile.lower_left_x - static_cast<double>(origin_key.column) * width;
  const double origin_y =
      origin_tile.lower_left_y + static_cast<double>(origin_key.row + 1) * width;
  const MetalTileHeader header = read_metal_tile_header(sources.front().path);
  if (retain && header.sample_type != MetalTileSampleType::Uint16Decimeters) {
    throw std::invalid_argument("Quantized atlas retention requires uint16 prepared terrain");
  }
  device = metal_device;
  // All prepared tiles in a catalogue share the reference grid and encoding.
  // Keep one template so later installations need only their compact LOD row.
  header_template = header;
  quantized_record = header.sample_type == MetalTileSampleType::Uint16Decimeters
                         ? quantized_metal_tile_record_layout(header)
                         : QuantizedMetalTileRecordLayout{};
  grid_origin_x = origin_x;
  grid_origin_y = origin_y;
  tile_width = width;
  mip_count = mip_values;
  vertex_count = vertex_values;
  slot_capacity = capacity;
  resident_count = 1U;
  next_use_stamp = 2U;
  installation_count = 1U;
  State *state = this;
  // Installation occurs only between completed frontier commands, and the
  // host waits for I/O before mipmap generation and for mipmaps before tracing.
  // Those explicit ordering points make whole-resource hazard tracking
  // unnecessary for these fixed, non-overlapping slot ranges.
  constexpr MTLResourceOptions kAtlasOptions =
      MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
  state->mipmap_atlas = make_buffer(
      device,
      checked_buffer_length(
          static_cast<size_t>(capacity) * mip_values,
          retain ? sizeof(uint16_t) : sizeof(float),
          "mipmap atlas"
      ),
      "mipmap atlas",
      kAtlasOptions
  );
  state->vertex_atlas = make_buffer(
      device,
      retain ? checked_buffer_length(capacity, state->quantized_record.stride, "vertex atlas")
             : checked_buffer_length(
                   static_cast<size_t>(capacity) * vertex_values,
                   sizeof(float),
                   "vertex atlas"
               ),
      "vertex atlas",
      kAtlasOptions
  );
  state->metadata_buffer = make_buffer(
      device,
      checked_buffer_length(capacity, sizeof(ResidentTile), "tile metadata"),
      "tile metadata"
  );
  state->preparation_slots = make_buffer(
      device,
      checked_buffer_length(capacity, sizeof(uint32_t), "terrain preparation slots"),
      "terrain preparation slots"
  );
  state->preparation_slots.label = @"Terrain preparation slots";
  state->io_queue = make_metal_io_queue(metal_device);

  // Custom files contain atlas-ordered vertices. Both representations need
  // GPU mipmap reduction; the default fixed-point path additionally converts
  // vertices, while retained fixed-point records stay uint16 throughout.
  NSError *error = nil;
  NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
  id<MTLLibrary> library = [metal_device newLibraryWithURL:library_url error:&error];
  if (library == nil) {
    print_error(@"Could not load the Metal library", error);
    throw std::runtime_error("Could not load Metal library for terrain preparation");
  }
  NSString *initial_mipmap_name = retain ? @"build_quantized_initial_maximum_mipmap_levels"
                                         : @"build_initial_maximum_mipmap_levels";
  id<MTLFunction> initial_mipmap_function = [library newFunctionWithName:initial_mipmap_name];
  if (initial_mipmap_function == nil) {
    throw std::runtime_error("Metal terrain-preparation kernel is missing");
  }
  state->initial_mipmap_pipeline =
      [metal_device newComputePipelineStateWithFunction:initial_mipmap_function error:&error];
  if (state->initial_mipmap_pipeline == nil) {
    print_error(@"Could not create the initial mipmap-generation pipeline", error);
    throw std::runtime_error("Could not create the initial mipmap-generation pipeline");
  }

  NSString *mipmap_name =
      retain ? @"build_quantized_maximum_mipmap_level" : @"build_maximum_mipmap_level";
  id<MTLFunction> mipmap_function = [library newFunctionWithName:mipmap_name];
  if (mipmap_function == nil) {
    throw std::runtime_error("Metal terrain-preparation kernel is missing");
  }
  state->mipmap_pipeline = [metal_device newComputePipelineStateWithFunction:mipmap_function
                                                                       error:&error];
  if (state->mipmap_pipeline == nil) {
    print_error(@"Could not create the mipmap-generation pipeline", error);
    throw std::runtime_error("Could not create the mipmap-generation pipeline");
  }
  state->mipmap_queue = [metal_device newCommandQueue];
  if (state->mipmap_queue == nil) {
    throw std::runtime_error("Could not create the mipmap-generation command queue");
  }

  if (state->header_template.sample_type == MetalTileSampleType::Uint16Decimeters && !retain) {
    id<MTLFunction> conversion_function =
        [library newFunctionWithName:@"convert_quantized_vertices"];
    if (conversion_function == nil) {
      throw std::runtime_error("Metal vertex-conversion kernel is missing");
    }
    state->conversion_pipeline =
        [metal_device newComputePipelineStateWithFunction:conversion_function error:&error];
    if (state->conversion_pipeline == nil) {
      print_error(@"Could not create the vertex-conversion pipeline", error);
      throw std::runtime_error("Could not create the vertex-conversion pipeline");
    }

    const size_t staging_tiles =
        std::min(static_cast<size_t>(capacity), static_cast<size_t>(metal_tile_io_concurrency()));
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
      vertex_values,
      timer
  );
  state->generate_mipmaps(
      std::span<const uint32_t>(&slot, 1U),
      state->header_template.cell_count,
      state->header_template.level_count,
      timer
  );
  state->bytes_loaded_with_metal_io =
      retain ? state->quantized_record.logical_size : state->header_template.vertex_byte_count;
  state->metadata[0] = make_resident_tile(
      origin_tile.lower_left_x,
      origin_tile.lower_left_y,
      origin_tile.maximum_elevation,
      origin_key,
      1U,
      config
  );
  state->slot_by_variant[{0U, 1U}] = 0U;
  state->variant_by_slot.assign(capacity, std::nullopt);
  state->variant_by_slot[0] = TileVariant{0U, 1U};
  state->last_used.assign(capacity, 0U);
  state->last_used[0] = 1U;
  atlas_attached = true;
}

uint32_t TileManager::State::slot_for_variant(TileVariant variant) const {
  if (variant.source_index >= catalogue->sources().size() || variant.lod == 0U) {
    throw std::invalid_argument("Resident terrain variant is invalid");
  }
  const auto found = slot_by_variant.find(variant);
  return found == slot_by_variant.end() ? slot_capacity : found->second;
}

std::vector<TileVariant>
TileManager::State::install_prepared(std::span<const uint8_t> pinned_slots, Timer &timer) {
  State &state = *this;
  rethrow_if_failed();
  if (pinned_slots.size() != state.slot_capacity) {
    throw std::invalid_argument("Resident pin mask has the wrong size");
  }
  timer.start_wall("Atlas installation");

  // Protect both imminent frontier slots and every destination selected in
  // this call. A prepared queue can contain more tiles than free atlas slots;
  // without this mask the LRU search could select one destination repeatedly.
  std::vector<uint8_t> unavailable_slots(pinned_slots.begin(), pinned_slots.end());
  std::vector<AtlasInstallation> pending_installations;
  pending_installations.reserve(state.slot_capacity);

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
      // the manager's bounded prepared queue until eviction becomes safe.
      break;
    }
    std::optional<PreparedTile> prepared_tile = try_take_prepared();
    if (!prepared_tile.has_value()) {
      break;
    }

    // Do not allow a later prepared tile to replace this one before the host
    // activates its deferred work and submits the next frontier pass.
    unavailable_slots[slot] = 1U;

    const TerrainSource &source = state.catalogue->sources()[prepared_tile->variant.source_index];
    const double lower_left_x =
        state.grid_origin_x + static_cast<double>(source.key.column) * state.tile_width;
    const double lower_left_y =
        state.grid_origin_y - static_cast<double>(source.key.row + 1) * state.tile_width;
    pending_installations.push_back({slot, std::move(*prepared_tile), lower_left_x, lower_left_y});
  }

  std::vector<MetalTileBufferLoad> loads_by_variant;
  std::vector<uint32_t> slots_by_variant;
  std::vector<int32_t> elevation_bases_by_variant;
  std::map<uint32_t, std::vector<size_t>> load_indices_by_lod;
  loads_by_variant.reserve(pending_installations.size());
  slots_by_variant.reserve(pending_installations.size());
  elevation_bases_by_variant.reserve(pending_installations.size());

  for (AtlasInstallation &installation : pending_installations) {
    const TerrainSource &source =
        state.catalogue->sources()[installation.prepared.variant.source_index];
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

  std::vector<TileVariant> installed;
  installed.reserve(pending_installations.size());

  // All payload writes are now complete. Publish the slot mappings together.
  // Loader workers also inspect load_states, so publish every lifecycle
  // transition directly while holding their shared manager lock.
  std::lock_guard<std::mutex> lifecycle_lock(state.loader_mutex);
  for (const AtlasInstallation &installation : pending_installations) {
    const uint32_t slot = installation.slot;
    const TileVariant variant = installation.prepared.variant;
    const std::optional<TileVariant> evicted = state.variant_by_slot[slot];
    if (evicted.has_value()) {
      state.slot_by_variant.erase(*evicted);
      state.load_states.at(*evicted) = TileLoadState::Unrequested;
      state.evictions++;
    } else {
      state.resident_count++;
    }

    state.slot_by_variant[variant] = slot;
    state.variant_by_slot[slot] = variant;
    state.last_used[slot] = state.next_use_stamp++;
    state.installation_count++;
    state.load_states.at(variant) = TileLoadState::Resident;
    installed.push_back(variant);
  }

  timer.stop("Atlas installation");

  return installed;
}

void TileManager::State::record_slot_use(std::span<const uint32_t> slots) {
  State &state = *this;
  for (uint32_t slot : slots) {
    if (slot >= state.slot_capacity || !state.variant_by_slot[slot].has_value()) {
      throw std::logic_error("GPU frontier refers to a nonresident tile slot");
    }
    // HostFrontier deduplicates this span, so one pass advances each active
    // slot's stamp exactly once regardless of how many rays reference it.
    state.last_used[slot] = state.next_use_stamp++;
  }
}

void TileManager::State::rebase_observer(ObserverLocation observer) {
  State &state = *this;
  if (!std::isfinite(observer.easting) || !std::isfinite(observer.northing)) {
    throw std::invalid_argument("Resident terrain rebase requires a finite observer");
  }
  for (uint32_t slot = 0U; slot < state.slot_capacity; slot++) {
    const std::optional<TileVariant> variant = state.variant_by_slot[slot];
    if (!variant.has_value()) {
      continue;
    }
    const TerrainSource &source = state.catalogue->sources()[variant->source_index];
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

std::optional<float> TileManager::State::sample_terrain(double easting, double northing) {
  if (!std::isfinite(easting) || !std::isfinite(northing)) {
    throw std::invalid_argument("Terrain sampling requires a finite projected coordinate");
  }
  if (!atlas_attached || device == nil || io_queue == nil) {
    throw std::logic_error("Terrain sampling requires an attached TileManager atlas");
  }

  const TileKey key = tile_key_at(catalogue->grid(), easting, northing);
  const std::optional<uint32_t> source_index = catalogue->find_source(key);
  if (!source_index.has_value()) {
    return std::nullopt;
  }

  const TerrainSource &source = catalogue->sources()[*source_index];
  const uint32_t cell_count = header_template.cell_count;
  const size_t side = static_cast<size_t>(cell_count) + 1U;
  const double lower_left_x =
      catalogue->grid().origin_x + static_cast<double>(key.column) * catalogue->grid().width;
  const double lower_left_y =
      catalogue->grid().origin_y - static_cast<double>(key.row + 1) * catalogue->grid().width;
  const double x = (easting - lower_left_x) / header_template.cell_size;
  const double y = (northing - lower_left_y) / header_template.cell_size;
  if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 || y < 0.0 || x > cell_count ||
      y > cell_count) {
    return std::nullopt;
  }

  const uint32_t x0 = std::min(cell_count - 1U, static_cast<uint32_t>(std::floor(x)));
  const uint32_t y0 = std::min(cell_count - 1U, static_cast<uint32_t>(std::floor(y)));
  const double tx = std::clamp(x - x0, 0.0, 1.0);
  const double ty = std::clamp(y - y0, 0.0, 1.0);

  const void *values = nullptr;
  MetalTileSampleType sample_type = header_template.sample_type;
  int32_t elevation_base = 0;
  const auto resident = slot_by_variant.find({*source_index, 1U});
  if (resident != slot_by_variant.end()) {
    // Inspection requires exact LOD-1 terrain. Reuse it in place when the
    // render atlas already contains that variant, decoding packed records
    // with the per-slot base written during installation.
    const uint32_t slot = resident->second;
    if (trace_quantized) {
      const auto *record = static_cast<const std::byte *>(vertex_atlas.contents) +
                           static_cast<size_t>(slot) * quantized_record.stride;
      std::memcpy(
          &elevation_base,
          record + quantized_record.elevation_base_offset,
          sizeof(elevation_base)
      );
      values = record + quantized_record.vertex_offset;
    } else {
      values = static_cast<const float *>(vertex_atlas.contents) +
               static_cast<size_t>(slot) * vertex_count;
      sample_type = MetalTileSampleType::Float32;
    }
  } else {
    // A render may legitimately retain only a coarse variant. Keep a separate
    // one-tile LOD-1 payload so cursor queries do not perturb frontier
    // residency or force the selected rendering LOD to change.
    if (!sampled_source_index.has_value() || *sampled_source_index != *source_index) {
      const MetalTileHeader header = read_metal_tile_header(source.path);
      if (header.cell_count != cell_count || header.sample_type != header_template.sample_type ||
          header.vertex_byte_count > std::numeric_limits<NSUInteger>::max()) {
        throw std::runtime_error("Terrain sampling tile disagrees with the resident atlas layout");
      }
      const NSUInteger byte_count = static_cast<NSUInteger>(header.vertex_byte_count);
      if (sampled_vertices == nil || sampled_vertices.length < byte_count) {
        sampled_vertices = [device newBufferWithLength:byte_count
                                               options:MTLResourceStorageModeShared];
      }
      if (sampled_vertices == nil || sampled_vertices.contents == nullptr) {
        throw std::runtime_error("Could not allocate terrain sampling buffer");
      }
      const MetalTileBufferLoad load = {
          source.path,
          0U,
          nil,
          header.vertex_offset,
          header.vertex_byte_count,
      };
      load_metal_tiles_into_buffer(
          device,
          io_queue,
          std::span<const MetalTileBufferLoad>(&load, 1U),
          sampled_vertices,
          sampled_vertices.length
      );
      bytes_loaded_with_metal_io += header.vertex_byte_count;
      sampled_header = header;
      sampled_source_index = *source_index;
    }
    values = sampled_vertices.contents;
    sample_type = sampled_header.sample_type;
    elevation_base = sampled_header.elevation_base_decimeters;
  }

  if (values == nullptr) {
    throw std::runtime_error("Could not access terrain sampling vertices");
  }
  // Use the same bilinear surface definition as the finest collision test.
  // Boundary samples select the final cell with interpolation weight one.
  const auto vertex = [&](uint32_t column, uint32_t row) {
    const size_t index = static_cast<size_t>(row) * side + column;
    if (sample_type == MetalTileSampleType::Float32) {
      return static_cast<double>(static_cast<const float *>(values)[index]);
    }
    return static_cast<double>(elevation_base) / 10.0 +
           static_cast<double>(static_cast<const uint16_t *>(values)[index]) / 10.0;
  };
  const double south = std::lerp(vertex(x0, y0), vertex(x0 + 1U, y0), tx);
  const double north = std::lerp(vertex(x0, y0 + 1U), vertex(x0 + 1U, y0 + 1U), tx);
  return static_cast<float>(std::lerp(south, north, ty));
}

TileManagerBindings TileManager::State::bindings() const {
  const State &state = *this;
  return {
      state.mipmap_atlas,
      state.vertex_atlas,
      state.metadata_buffer,
      state.trace_quantized
          ? QuantizedTerrainLayout{
                state.quantized_record.stride,
                state.quantized_record.vertex_offset,
                state.quantized_record.elevation_base_offset,
            }
          : QuantizedTerrainLayout{},
  };
}

TileManagerStatistics TileManager::State::statistics() const {
  const State &state = *this;
  std::lock_guard<std::mutex> lock(loader_mutex);
  return {request_count,
          unique_request_count,
          duplicate_request_count,
          worker_count,
          state.installation_count,
          state.bytes_loaded_with_metal_io,
          state.evictions,
          state.resident_count,
          state.slot_capacity};
}

} // namespace panorama
