#include "resident_tile_cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

static_assert(sizeof(ResidentTile) == 3U * sizeof(uint64_t));
static_assert(sizeof(ResidentTileHashEntry) == 3U * sizeof(uint64_t));

/// Check that a byte count fits Metal's NSUInteger buffer-length argument.
[[nodiscard]] NSUInteger checked_buffer_length(size_t count, size_t size, const char *name) {
  if (count > std::numeric_limits<size_t>::max() / size) {
    throw std::overflow_error(std::string(name) + " buffer is too large");
  }
  return static_cast<NSUInteger>(count * size);
}

/// Allocate shared storage which the host can update between command buffers.
[[nodiscard]] id<MTLBuffer> make_buffer(id<MTLDevice> device, NSUInteger length, const char *name) {
  id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(std::string("Could not allocate ") + name + " Metal buffer");
  }
  return buffer;
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

} // namespace

/// Mutable atlas state hidden behind the cache's ownership-oriented interface.
struct ResidentTileCache::State {
  std::span<const TerrainSource> sources;
  const RaytraceConfig &config;
  uint32_t mip_count;
  uint32_t vertex_count;
  uint32_t slot_capacity;
  uint32_t hash_slot_count;
  uint32_t resident_count = 1U;
  uint64_t next_use_stamp = 2U;
  uint64_t installations = 1U;
  uint64_t bytes_copied;
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
  std::vector<uint64_t> last_used;

  /// Initialise fixed atlas dimensions before allocating Metal resources.
  State(
      std::span<const TerrainSource> source_values,
      const RaytraceConfig &config_value,
      uint32_t mip_values,
      uint32_t vertex_values,
      uint32_t slot_values,
      uint32_t hash_values,
      uint64_t initial_bytes
  )
      : sources(source_values), config(config_value), mip_count(mip_values),
        vertex_count(vertex_values), slot_capacity(slot_values), hash_slot_count(hash_values),
        bytes_copied(initial_bytes) {}
};

ResidentTileCache::ResidentTileCache(
    id<MTLDevice> device,
    std::span<const TerrainSource> sources,
    const LoadedTile &origin,
    TileKey origin_key,
    const RaytraceConfig &config,
    uint32_t slot_capacity
) {
  if (sources.empty() || slot_capacity == 0U || origin.vertices == nullptr) {
    throw std::invalid_argument("Resident tile cache requires an origin tile and slots");
  }
  const uint32_t mip_count = static_cast<uint32_t>(origin.mipmap.size());
  const uint32_t vertex_count = static_cast<uint32_t>(origin.vertices->size());
  const uint32_t hash_slot_count = resident_hash_capacity(slot_capacity);
  const uint64_t tile_bytes = static_cast<uint64_t>(mip_count + vertex_count) * sizeof(float);

  // Each slot has the same payload length. Metal derives an address from a
  // slot index and this stride, so a work item never needs per-tile offsets.
  auto state = std::make_unique<
      State>(sources, config, mip_count, vertex_count, slot_capacity, hash_slot_count, tile_bytes);
  // Shared buffers let the main thread install a prepared tile between two
  // completed command buffers. No worker thread has access to these buffers.
  state->mipmap_atlas = make_buffer(
      device,
      checked_buffer_length(
          static_cast<size_t>(slot_capacity) * mip_count,
          sizeof(float),
          "mipmap atlas"
      ),
      "mipmap atlas"
  );
  state->vertex_atlas = make_buffer(
      device,
      checked_buffer_length(
          static_cast<size_t>(slot_capacity) * vertex_count,
          sizeof(float),
          "vertex atlas"
      ),
      "vertex atlas"
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
  state->mipmaps = static_cast<float *>(state->mipmap_atlas.contents);
  state->vertices = static_cast<float *>(state->vertex_atlas.contents);
  state->metadata = static_cast<ResidentTile *>(state->metadata_buffer.contents);
  state->hash = static_cast<ResidentTileHashEntry *>(state->hash_buffer.contents);
  if (state->mipmaps == nullptr || state->vertices == nullptr || state->metadata == nullptr ||
      state->hash == nullptr) {
    throw std::runtime_error("Could not map resident terrain atlas");
  }

  // Slot zero permanently starts as the observer tile, so the first frontier
  // can run before any asynchronous source has finished preparation.
  std::memcpy(state->mipmaps, origin.mipmap.data(), static_cast<size_t>(mip_count) * sizeof(float));
  std::memcpy(
      state->vertices,
      origin.vertices->data(),
      static_cast<size_t>(vertex_count) * sizeof(float)
  );
  state->metadata[0] = make_resident_tile(origin, origin_key, config);
  state->slot_by_source.assign(sources.size(), slot_capacity);
  state->slot_by_source[0] = 0U;
  state->source_by_slot.assign(slot_capacity, static_cast<uint32_t>(sources.size()));
  state->source_by_slot[0] = 0U;
  state->last_used.assign(slot_capacity, 0U);
  state->last_used[0] = 1U;
  state_ = std::move(state);

  // Slot zero must be visible to the first frontier pass before any workers run.
  std::fill_n(state_->hash, state_->hash_slot_count, ResidentTileHashEntry{});
  const uint32_t mask = state_->hash_slot_count - 1U;
  const uint32_t index = static_cast<uint32_t>(tile_key_hash(origin_key)) & mask;
  state_->hash[index] = {origin_key.row, origin_key.column, 0U, 1U};
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
  bool hash_changed = false;
  while (true) {
    // Prefer an unused slot. Once full, choose the least-recently-used slot
    // that no imminent frontier work item pins for the next GPU command.
    uint32_t slot = state.slot_capacity;
    if (state.resident_count < state.slot_capacity) {
      slot = state.resident_count;
    } else {
      uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
      for (uint32_t candidate = 0U; candidate < state.slot_capacity; candidate++) {
        if (pinned_slots[candidate] == 0U && state.last_used[candidate] < oldest_use) {
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

    // The prepared vectors remain CPU-owned until both payloads have been
    // copied to matching fixed-stride ranges in the shared atlas.
    timer.start_wall("Atlas copy");
    std::memcpy(
        state.mipmaps + static_cast<size_t>(slot) * state.mip_count,
        prepared->tile->mipmap.data(),
        static_cast<size_t>(state.mip_count) * sizeof(float)
    );
    std::memcpy(
        state.vertices + static_cast<size_t>(slot) * state.vertex_count,
        prepared->tile->vertices->data(),
        static_cast<size_t>(state.vertex_count) * sizeof(float)
    );
    timer.stop("Atlas copy");
    state.metadata[slot] = make_resident_tile(
        *prepared->tile,
        state.sources[prepared->source_index].key,
        state.config
    );

    const uint32_t evicted = state.source_by_slot[slot];
    if (evicted != state.sources.size()) {
      // The old source no longer has GPU residency. It may later be requested
      // again; the preparer will then count and perform a reload.
      state.slot_by_source[evicted] = state.slot_capacity;
      preparer.mark_evicted(evicted);
      state.evictions++;
    }
    // Publish the host-side bidirectional mapping before rebuilding the GPU
    // key hash after this installation batch.
    state.slot_by_source[prepared->source_index] = slot;
    state.source_by_slot[slot] = prepared->source_index;
    state.last_used[slot] = state.next_use_stamp++;
    if (state.resident_count < state.slot_capacity) {
      state.resident_count++;
    }
    state.installations++;
    state.bytes_copied +=
        static_cast<uint64_t>(state.mip_count + state.vertex_count) * sizeof(float);
    preparer.mark_resident(prepared->source_index);
    hash_changed = true;
  }

  if (hash_changed) {
    // The GPU performs open-addressed lookup with the same hash/probe scheme.
    // Rebuilding once here avoids publishing a partially changed table.
    timer.start_wall("Resident hash rebuild");
    std::fill_n(state.hash, state.hash_slot_count, ResidentTileHashEntry{});
    const uint32_t mask = state.hash_slot_count - 1U;
    for (uint32_t slot = 0U; slot < state.resident_count; slot++) {
      const uint32_t source = state.source_by_slot[slot];
      if (source == state.sources.size()) {
        continue;
      }
      const TileKey key = state.sources[source].key;
      uint32_t index = static_cast<uint32_t>(tile_key_hash(key)) & mask;
      while (state.hash[index].occupied != 0U) {
        // The table is at most half full, keeping linear probes short.
        index = (index + 1U) & mask;
      }
      state.hash[index] = {key.row, key.column, slot, 1U};
    }
    timer.stop("Resident hash rebuild");
  }
  timer.stop("Atlas installation");
}

std::vector<uint8_t>
ResidentTileCache::pin_slots(std::span<const uint32_t> slots, bool record_use) {
  State &state = *state_;
  std::vector<uint8_t> pinned(state.slot_capacity, 0U);
  for (uint32_t slot : slots) {
    if (slot >= state.resident_count || state.source_by_slot[slot] == state.sources.size()) {
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
          state.evictions,
          state.resident_count,
          state.slot_capacity};
}

} // namespace panorama
