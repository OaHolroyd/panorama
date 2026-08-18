#include "resident_tile_cache.h"

#import <Foundation/Foundation.h>

#include "metal_tile.h"

#include <algorithm>
#include <cstring>
#include <deque>
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

/// One prepared source paired with its reserved destination atlas slot.
///
/// Residency is not published until every payload in the installation batch
/// has finished loading and every custom tile has a complete GPU mipmap.
struct AtlasInstallation {
  uint32_t slot;
  PreparedTile prepared;
  double lower_left_x;
  double lower_left_y;
};

/// Host-visible lifecycle of one fixed atlas slot.
enum class AtlasSlotState : uint8_t {
  Empty,
  Loading,
  Resident,
};

/// One committed mipmap command and the resources it reads while in flight.
struct PendingAtlasInstallation {
  std::vector<AtlasInstallation> installations;
  std::vector<MetalTileIoSubmission> io_submissions;
  id<MTLBuffer> slot_buffer;
  id<MTLCommandBuffer> mipmap_command;
  id<MTLSharedEvent> completion_event;
};

/// Completed payload commands retained until Metal releases their queue slots.
///
/// A shared event can become visible just before the command-buffer status
/// changes to Complete. Retaining and counting these commands prevents a call
/// to `commandBuffer` from blocking on the I/O queue's bounded object pool.
struct RetiredAtlasCommands {
  std::vector<MetalTileIoSubmission> io_submissions;
  id<MTLCommandBuffer> mipmap_command;
};

/// Resources returned when a mipmap batch has been submitted asynchronously.
struct MipmapSubmission {
  id<MTLBuffer> slot_buffer;
  id<MTLCommandBuffer> command;
  id<MTLSharedEvent> completion_event;
};

} // namespace

/// Mutable atlas state hidden behind the cache's ownership-oriented interface.
struct ResidentTileCache::State {
  std::span<const TerrainSource> sources;
  const RaytraceConfig &config;
  id<MTLDevice> device;
  id<MTLIOCommandQueue> io_queue;
  id<MTLCommandQueue> mipmap_queue;
  id<MTLComputePipelineState> mipmap_pipeline;
  MetalTileHeader header_template;
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
  uint64_t bytes_loaded_directly = 0U;
  uint64_t evictions = 0U;
  id<MTLBuffer> mipmap_atlas;
  id<MTLBuffer> vertex_atlas;
  id<MTLBuffer> metadata_buffer;
  id<MTLBuffer> hash_buffer;
  float *mipmaps;
  float *vertices;
  ResidentTile *metadata;
  ResidentTileHashEntry *hash;
  std::vector<uint32_t> slot_by_source;
  std::vector<uint32_t> source_by_slot;
  std::vector<AtlasSlotState> slot_states;
  std::vector<uint64_t> last_used;
  std::deque<PendingAtlasInstallation> pending_installations;
  std::deque<RetiredAtlasCommands> retired_commands;

  /// Submit every maximum-mipmap level after the corresponding I/O events.
  [[nodiscard]] MipmapSubmission submit_mipmaps(
      std::span<const uint32_t> slots,
      std::span<const MetalTileIoSubmission> io_submissions
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
        header_template(header_value), grid_origin_x(origin_x), grid_origin_y(origin_y),
        tile_width(width), mip_count(mip_values), vertex_count(vertex_values),
        slot_capacity(slot_values), hash_slot_count(hash_values), bytes_copied(initial_bytes) {}
};

MipmapSubmission ResidentTileCache::State::submit_mipmaps(
    std::span<const uint32_t> slots,
    std::span<const MetalTileIoSubmission> io_submissions
) {
  if (slots.empty()) {
    throw std::invalid_argument("Cannot submit an empty mipmap-generation batch");
  }
  if (slots.size() > slot_capacity) {
    throw std::logic_error("Mipmap-generation batch exceeds its slot buffer");
  }

  // The slot list is immutable until this command completes. A dedicated
  // shared buffer avoids overwriting indices still consumed by an earlier
  // asynchronous batch.
  id<MTLBuffer> slot_buffer = [device newBufferWithBytes:slots.data()
                                                  length:slots.size_bytes()
                                                 options:MTLResourceStorageModeShared];
  id<MTLSharedEvent> completion_event = [device newSharedEvent];
  id<MTLCommandBuffer> command = [mipmap_queue commandBuffer];
  if (slot_buffer == nil || completion_event == nil || command == nil) {
    throw std::runtime_error("Could not create an asynchronous mipmap-generation command");
  }
  command.label = @"Generate resident tile mipmaps";

  // Metal I/O and compute use different queues. Waiting in the GPU command
  // buffer lets the CPU return to frontier scheduling immediately while
  // preserving the vertex-write-to-mipmap-read dependency for every slot.
  for (const MetalTileIoSubmission &io : io_submissions) {
    [command encodeWaitForEvent:io.event value:1U];
  }

  // Generate level 1 for every tile from adjacent vertices, then reduce each
  // preceding level in turn. Ending each encoder supplies the global
  // dependency between dispatches; no threadgroup-local barrier would be
  // sufficient. The Z dimension batches slots without changing level order.
  uint32_t source_side = header_template.cell_count + 1U;
  uint32_t source_step = 1U;
  uint32_t source_tile_stride = vertex_count;
  uint32_t destination_tile_stride = mip_count;
  const uint32_t tile_count = static_cast<uint32_t>(slots.size());
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Could not encode maximum mipmap level 1");
  }
  encoder.label = @"Build maximum mipmap level 1";
  [encoder setComputePipelineState:mipmap_pipeline];
  [encoder setBuffer:vertex_atlas offset:0U atIndex:0];
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
    [encoder setBuffer:mipmap_atlas offset:previous_offset * sizeof(float) atIndex:0];
    [encoder setBuffer:mipmap_atlas offset:output_offset * sizeof(float) atIndex:1];
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

  // Signal only after the final mip level is complete. The host polls this
  // event at command-buffer boundaries before publishing the new hash entries.
  [command encodeSignalEvent:completion_event value:1U];
  [command commit];
  return {slot_buffer, command, completion_event};
}

void ResidentTileCache::State::generate_mipmaps(std::span<const uint32_t> slots, Timer &timer) {
  timer.start_wall("GPU mipmap generation");
  const MipmapSubmission submission = submit_mipmaps(slots, {});
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
    if (slot_states[slot] != AtlasSlotState::Resident) {
      continue;
    }
    const uint32_t source = source_by_slot[slot];
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
  const MetalTileHeader header_template = {
      kMetalTileMagic,
      kMetalTileVersion,
      static_cast<uint32_t>(sizeof(MetalTileHeader)),
      MetalTileCompression::None,
      origin.crs.epsg_code(),
      origin.size,
      origin.num_levels,
      origin.maximum_elevation,
      MetalTileSampleType::Float32,
      origin_key.row,
      origin_key.column,
      origin.lower_left_x,
      origin.lower_left_y,
      origin.delta,
      sizeof(MetalTileHeader),
      static_cast<uint64_t>(vertex_count) * sizeof(float),
  };

  // Each slot has the same payload length. Metal derives an address from a
  // slot index and this stride, so a work item never needs per-tile offsets.
  auto state = std::make_unique<State>(
      sources,
      config,
      device,
      header_template,
      grid_origin_x,
      grid_origin_y,
      tile_width,
      mip_count,
      vertex_count,
      slot_capacity,
      hash_slot_count,
      custom_origin ? 0U : tile_bytes
  );
  // Shared buffers let the main thread install a prepared tile between two
  // completed command buffers. No worker thread has access to these buffers.
  // I/O/mipmap commands write only Loading slots while ray commands read only
  // Resident slots. Disable whole-resource hazard tracking on these atlases so
  // Metal does not serialize logically disjoint slot ranges across queues; the
  // cache state machine and I/O-to-mipmap shared events provide the required
  // explicit synchronization.
  constexpr MTLResourceOptions kAtlasOptions =
      MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
  state->mipmap_atlas = make_buffer(
      device,
      checked_buffer_length(
          static_cast<size_t>(slot_capacity) * mip_count,
          sizeof(float),
          "mipmap atlas"
      ),
      "mipmap atlas",
      kAtlasOptions
  );
  state->vertex_atlas = make_buffer(
      device,
      checked_buffer_length(
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

    // Custom files contain vertices only. Compile the small reduction kernel
    // once; every direct installation reuses it to populate a slot batch.
    NSError *error = nil;
    NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      throw std::runtime_error("Could not load Metal library for mipmap generation");
    }
    id<MTLFunction> function = [library newFunctionWithName:@"build_maximum_mipmap_level"];
    if (function == nil) {
      throw std::runtime_error("Metal mipmap-generation kernel is missing");
    }
    state->mipmap_pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (state->mipmap_pipeline == nil) {
      print_error(@"Could not create the mipmap-generation pipeline", error);
      throw std::runtime_error("Could not create the mipmap-generation pipeline");
    }
    state->mipmap_queue = [device newCommandQueue];
    if (state->mipmap_queue == nil) {
      throw std::runtime_error("Could not create the mipmap-generation command queue");
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
  // vertices take the same direct I/O and GPU-reduction path as later tiles.
  if (custom_origin) {
    const MetalTileBufferLoad load = {sources.front().path, 0U, nil};
    timer.start_wall("Metal tile I/O");
    load_metal_tiles_into_buffer(
        device,
        state->io_queue,
        std::span<const MetalTileBufferLoad>(&load, 1U),
        state->header_template.vertex_offset,
        state->header_template.vertex_byte_count,
        state->vertex_atlas,
        state->vertex_atlas.length
    );
    timer.stop("Metal tile I/O");
    const uint32_t slot = 0U;
    state->generate_mipmaps(std::span<const uint32_t>(&slot, 1U), timer);
    state->bytes_loaded_directly = static_cast<uint64_t>(vertex_count) * sizeof(float);
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
  state->slot_states.assign(slot_capacity, AtlasSlotState::Empty);
  state->slot_states[0] = AtlasSlotState::Resident;
  state->last_used.assign(slot_capacity, 0U);
  state->last_used[0] = 1U;
  state_ = std::move(state);

  // Slot zero must be visible to the first frontier pass before any workers run.
  state_->rebuild_hash(nullptr);
}

ResidentTileCache::~ResidentTileCache() {
  if (state_ == nullptr) {
    return;
  }

  // A final frontier can finish while speculative neighbour loads are still
  // in flight. Their commands retain the atlas resources, but draining here
  // also keeps every Objective-C object in the pending records alive until
  // its command has finished using it.
  for (PendingAtlasInstallation &pending : state_->pending_installations) {
    for (const MetalTileIoSubmission &io : pending.io_submissions) {
      [io.command waitUntilCompleted];
    }
    [pending.mipmap_command waitUntilCompleted];
  }
  for (RetiredAtlasCommands &retired : state_->retired_commands) {
    for (const MetalTileIoSubmission &io : retired.io_submissions) {
      [io.command waitUntilCompleted];
    }
    [retired.mipmap_command waitUntilCompleted];
  }
}

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

  bool hash_changed = false;

  // In addition to imminent-frontier pins, protect every slot published or
  // reserved during this call. Deferred work is activated only after this
  // method returns, so immediately evicting a newly completed tile would
  // otherwise cause an endless load/evict cycle without tracing that tile.
  std::vector<uint8_t> unavailable_slots(pinned_slots.begin(), pinned_slots.end());

  // Reap commands only after Metal reports their command buffers complete.
  // The batch-completion event is sufficient for publishing data, but it can
  // precede command-buffer retirement by a short interval. Counting that
  // interval avoids blocking when requesting another bounded I/O command.
  for (auto iterator = state.retired_commands.begin(); iterator != state.retired_commands.end();) {
    bool io_complete = true;
    for (const MetalTileIoSubmission &io : iterator->io_submissions) {
      if (io.command.status == MTLIOStatusError || io.command.status == MTLIOStatusCancelled) {
        const char *detail = io.command.error == nil
                                 ? "unknown error"
                                 : io.command.error.localizedDescription.UTF8String;
        throw std::runtime_error(
            "Could not load Metal tile batch beginning with " + io.paths.front().string() + ": " +
            detail
        );
      }
      io_complete = io_complete && io.command.status == MTLIOStatusComplete;
    }
    if (iterator->mipmap_command.status == MTLCommandBufferStatusError) {
      print_error(@"Asynchronous Metal mipmap generation failed", iterator->mipmap_command.error);
      throw std::runtime_error("Asynchronous Metal mipmap generation failed");
    }
    if (!io_complete || iterator->mipmap_command.status != MTLCommandBufferStatusCompleted) {
      iterator++;
      continue;
    }
    timer.add_work(
        "GPU mipmap generation",
        1'000.0 * (iterator->mipmap_command.GPUEndTime - iterator->mipmap_command.GPUStartTime)
    );
    iterator = state.retired_commands.erase(iterator);
  }

  // Shared-event completion is the publication boundary. Until this point a
  // loading slot has no source-to-slot mapping and no resident-hash entry, so
  // frontier commands can safely keep using every other resident slot while
  // Metal I/O and mipmap generation operate in parallel.
  while (!state.pending_installations.empty()) {
    PendingAtlasInstallation &pending = state.pending_installations.front();
    for (const MetalTileIoSubmission &io : pending.io_submissions) {
      if (io.command.status == MTLIOStatusError || io.command.status == MTLIOStatusCancelled) {
        const char *detail = io.command.error == nil
                                 ? "unknown error"
                                 : io.command.error.localizedDescription.UTF8String;
        throw std::runtime_error(
            "Could not load Metal tile batch beginning with " + io.paths.front().string() + ": " +
            detail
        );
      }
    }
    if (pending.mipmap_command.status == MTLCommandBufferStatusError) {
      print_error(@"Asynchronous Metal mipmap generation failed", pending.mipmap_command.error);
      throw std::runtime_error("Asynchronous Metal mipmap generation failed");
    }
    if (pending.completion_event.signaledValue < 1U) {
      break;
    }

    // All payload writes preceded the completion signal. Publish the entire
    // batch atomically from the scheduler's point of view, then rebuild the
    // hash once after every completed batch has been consumed.
    for (const AtlasInstallation &installation : pending.installations) {
      const uint32_t slot = installation.slot;
      const uint32_t source = installation.prepared.source_index;
      if (state.slot_states[slot] != AtlasSlotState::Loading) {
        throw std::logic_error("Completed atlas installation does not own a loading slot");
      }
      state.slot_states[slot] = AtlasSlotState::Resident;
      unavailable_slots[slot] = 1U;
      state.slot_by_source[source] = slot;
      state.last_used[slot] = state.next_use_stamp++;
      state.resident_count++;
      state.installations++;
      preparer.mark_resident(source);
    }
    state.retired_commands.push_back(
        {
            std::move(pending.io_submissions),
            pending.mipmap_command,
        }
    );
    state.pending_installations.pop_front();
    hash_changed = true;
  }

  // A tile installed during this call cannot be used until control returns to
  // the frontier scheduler. Treat its slot like an imminent-frontier pin for
  // the rest of this batch. Without this additional mask, a full prepared
  // queue could repeatedly overwrite the same small set of unpinned slots,
  // evicting tiles before their deferred rays had any opportunity to run.
  // Reserve destinations before performing any I/O. This separates cache
  // admission from payload transfer and gives the custom path a complete set
  // of independent files which it can submit to Metal I/O concurrently.
  std::vector<AtlasInstallation> installations;
  installations.reserve(state.slot_capacity);
  size_t io_commands_in_flight = 0U;
  for (const PendingAtlasInstallation &pending : state.pending_installations) {
    for (const MetalTileIoSubmission &io : pending.io_submissions) {
      io_commands_in_flight += io.command.status == MTLIOStatusComplete ? 0U : io.files.size();
    }
  }
  for (const RetiredAtlasCommands &retired : state.retired_commands) {
    for (const MetalTileIoSubmission &io : retired.io_submissions) {
      io_commands_in_flight += io.command.status == MTLIOStatusComplete ? 0U : io.files.size();
    }
  }
  const size_t io_capacity = static_cast<size_t>(metal_tile_io_concurrency());
  size_t available_io_commands = io_capacity - io_commands_in_flight;

  while (true) {
    // Custom tiles require one I/O command each. Do not remove another item
    // from the preparer's queue until the bounded Metal I/O queue can accept it.
    if (state.io_queue != nil && available_io_commands == 0U) {
      break;
    }

    // Prefer an unused slot. Once full, choose the least-recently-used slot
    // that is neither pinned by the next GPU command nor already selected by
    // this installation batch.
    uint32_t slot = state.slot_capacity;
    for (uint32_t candidate = 0U; candidate < state.slot_capacity; candidate++) {
      if (state.slot_states[candidate] == AtlasSlotState::Empty) {
        slot = candidate;
        break;
      }
    }
    if (slot == state.slot_capacity) {
      uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
      for (uint32_t candidate = 0U; candidate < state.slot_capacity; candidate++) {
        if (state.slot_states[candidate] == AtlasSlotState::Resident &&
            unavailable_slots[candidate] == 0U && state.last_used[candidate] < oldest_use) {
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

    // Remove the old source before direct I/O can overwrite its payload. The
    // hash is rebuilt below while the replacement remains explicitly Loading.
    const uint32_t evicted = state.source_by_slot[slot];
    if (state.slot_states[slot] == AtlasSlotState::Resident) {
      state.slot_by_source[evicted] = state.slot_capacity;
      preparer.mark_evicted(evicted);
      state.resident_count--;
      state.evictions++;
      hash_changed = true;
    }
    state.slot_states[slot] = AtlasSlotState::Loading;
    state.source_by_slot[slot] = prepared->source_index;
    installations.push_back({slot, std::move(*prepared), lower_left_x, lower_left_y});
    if (is_metal_tile_path(source.path)) {
      available_io_commands--;
    }
  }

  // GeoTIFFs already have CPU-resident payloads, while custom files contribute
  // direct-I/O requests and slot IDs for the following two batched operations.
  std::vector<MetalTileBufferLoad> custom_loads;
  std::vector<uint32_t> custom_slots;
  custom_loads.reserve(installations.size());
  custom_slots.reserve(installations.size());
  std::vector<AtlasInstallation> asynchronous_installations;
  asynchronous_installations.reserve(installations.size());
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

      // A CPU copy is complete before this method returns, so GeoTIFF slots
      // can be published immediately without entering the asynchronous queue.
      state.slot_states[installation.slot] = AtlasSlotState::Resident;
      state.slot_by_source[installation.prepared.source_index] = installation.slot;
      state.last_used[installation.slot] = state.next_use_stamp++;
      state.resident_count++;
      state.installations++;
      preparer.mark_resident(installation.prepared.source_index);
      hash_changed = true;
    } else {
      // The catalogue and origin header establish the common payload layout.
      // Installation therefore needs only a file and final buffer offset.
      custom_loads.push_back(
          {
              source.path,
              static_cast<NSUInteger>(installation.slot) * state.header_template.vertex_byte_count,
              installation.prepared.metal_file,
          }
      );
      custom_slots.push_back(installation.slot);
      state.metadata[installation.slot] = make_resident_tile(
          installation.lower_left_x,
          installation.lower_left_y,
          source.key,
          state.config
      );
      asynchronous_installations.push_back(std::move(installation));
    }
  }

  if (!custom_loads.empty()) {
    // Commit direct loads and immediately chain one mipmap command behind all
    // of their individual shared events. Neither submission waits on the CPU.
    timer.start_wall("Metal I/O submission");
    const std::vector<MetalTileIoSubmission> io_submissions = submit_metal_tiles_into_buffer(
        state.device,
        state.io_queue,
        custom_loads,
        state.header_template.vertex_offset,
        state.header_template.vertex_byte_count,
        state.vertex_atlas,
        state.vertex_atlas.length
    );
    timer.stop("Metal I/O submission");

    timer.start_wall("Mipmap submission");
    const MipmapSubmission mipmaps = state.submit_mipmaps(custom_slots, io_submissions);
    timer.stop("Mipmap submission");
    state.pending_installations.push_back(
        {
            std::move(asynchronous_installations),
            io_submissions,
            mipmaps.slot_buffer,
            mipmaps.command,
            mipmaps.completion_event,
        }
    );
    state.bytes_loaded_directly +=
        static_cast<uint64_t>(custom_slots.size()) * state.vertex_count * sizeof(float);
  }

  if (hash_changed) {
    state.rebuild_hash(&timer);
  }
  timer.stop("Atlas installation");
}

bool ResidentTileCache::has_pending_installations() const {
  return !state_->pending_installations.empty();
}

void ResidentTileCache::wait_for_pending_installation(Timer &timer) {
  State &state = *state_;
  if (state.pending_installations.empty()) {
    return;
  }

  // The mipmap queue is ordered, so the oldest command is the first batch
  // which can become publishable. This is the only host wait in the loading
  // pipeline and the scheduler invokes it only with an empty resident frontier.
  timer.start_wall("Pending atlas wait");
  PendingAtlasInstallation &pending = state.pending_installations.front();
  for (const MetalTileIoSubmission &io : pending.io_submissions) {
    [io.command waitUntilCompleted];
    if (io.command.status != MTLIOStatusComplete) {
      const char *detail = io.command.error == nil
                               ? "unknown error"
                               : io.command.error.localizedDescription.UTF8String;
      throw std::runtime_error(
          "Could not load Metal tile batch beginning with " + io.paths.front().string() + ": " +
          detail
      );
    }
  }
  [pending.mipmap_command waitUntilCompleted];
  timer.stop("Pending atlas wait");
  if (pending.mipmap_command.status == MTLCommandBufferStatusError) {
    print_error(@"Asynchronous Metal mipmap generation failed", pending.mipmap_command.error);
    throw std::runtime_error("Asynchronous Metal mipmap generation failed");
  }
}

std::vector<uint8_t>
ResidentTileCache::pin_slots(std::span<const uint32_t> slots, bool record_use) {
  State &state = *state_;
  std::vector<uint8_t> pinned(state.slot_capacity, 0U);
  for (uint32_t slot : slots) {
    if (slot >= state.slot_capacity || state.slot_states[slot] != AtlasSlotState::Resident) {
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
  return {state.mipmap_atlas,
          state.vertex_atlas,
          state.metadata_buffer,
          state.hash_buffer,
          state.hash_slot_count};
}

uint32_t ResidentTileCache::slot_capacity() const { return state_->slot_capacity; }
uint32_t ResidentTileCache::resident_tile_count() const { return state_->resident_count; }

ResidentTileCacheStatistics ResidentTileCache::statistics() const {
  const State &state = *state_;
  return {state.installations,
          state.bytes_copied,
          state.bytes_loaded_directly,
          state.evictions,
          state.resident_count,
          state.slot_capacity};
}

} // namespace panorama
