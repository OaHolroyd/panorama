#pragma once

#include "metal_tile.h"
#include "tile_manager.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace panorama {

enum class TileLoadState : uint8_t {
  Unrequested,
  Queued,
  Loading,
  Prepared,
  Resident,
};

struct TileLoadRequest {
  float priority;
  TileVariant variant;
};

struct TileLoadRequestGreater {
  [[nodiscard]] bool operator()(const TileLoadRequest &left, const TileLoadRequest &right) const {
    return left.priority > right.priority;
  }
};

struct PreparedTile {
  TileVariant variant;
  id<MTLIOFileHandle> metal_file;
  MetalTileLod metal_lod;
};

struct TileManager::State {
  RaytraceConfig config;
  std::unique_ptr<TerrainCatalogue> catalogue;
  std::unique_ptr<TileGeometry> origin;
  std::vector<uint32_t> lod_by_source;
  float pixel_angle = 0.0F;
  uint32_t observer_source_index = 0U;
  uint32_t mipmap_values = 0U;
  bool trace_quantized = false;

  id<MTLDevice> device = nil;
  Timer *loader_timer = nullptr;
  uint32_t prepared_capacity = 0U;
  std::atomic<bool> stop_requested{false};
  mutable std::mutex loader_mutex;
  std::condition_variable request_available;
  std::condition_variable prepared_available;
  std::condition_variable prepared_space_available;
  std::priority_queue<TileLoadRequest, std::vector<TileLoadRequest>, TileLoadRequestGreater>
      requests;
  std::map<TileVariant, TileLoadState> load_states;
  std::map<TileVariant, float> queued_priorities;
  std::map<TileVariant, uint8_t> requested_before;
  std::deque<PreparedTile> prepared_tiles;
  std::vector<std::thread> workers;
  std::exception_ptr loader_error;
  uint64_t request_count = 0U;
  uint64_t unique_request_count = 0U;
  uint64_t duplicate_request_count = 0U;
  uint32_t worker_count = 0U;
  bool workers_started = false;

  id<MTLIOCommandQueue> io_queue = nil;
  id<MTLCommandQueue> mipmap_queue = nil;
  id<MTLComputePipelineState> conversion_pipeline = nil;
  id<MTLComputePipelineState> initial_mipmap_pipeline = nil;
  id<MTLComputePipelineState> mipmap_pipeline = nil;
  MetalTileHeader header_template = {};
  QuantizedMetalTileRecordLayout quantized_record = {};
  bool atlas_attached = false;
  double grid_origin_x = 0.0;
  double grid_origin_y = 0.0;
  double tile_width = 0.0;
  uint32_t mip_count = 0U;
  uint32_t vertex_count = 0U;
  uint32_t slot_capacity = 0U;
  uint32_t resident_count = 0U;
  uint64_t next_use_stamp = 1U;
  uint64_t installation_count = 0U;
  uint64_t bytes_loaded_with_metal_io = 0U;
  uint64_t evictions = 0U;
  id<MTLBuffer> mipmap_atlas = nil;
  id<MTLBuffer> vertex_atlas = nil;
  id<MTLBuffer> quantized_staging = nil;
  id<MTLBuffer> preparation_slots = nil;
  id<MTLBuffer> metadata_buffer = nil;
  ResidentTile *metadata = nullptr;
  std::map<TileVariant, uint32_t> slot_by_variant;
  std::vector<std::optional<TileVariant>> variant_by_slot;
  std::vector<uint64_t> last_used;
  id<MTLBuffer> sampled_vertices = nil;
  std::optional<uint32_t> sampled_source_index;
  MetalTileHeader sampled_header = {};

  void rebuild_lod_plan(float angle);
  void start_workers(uint32_t configured_workers);
  void request_tile(uint32_t source_index, uint32_t lod, float priority);
  [[nodiscard]] std::optional<PreparedTile> try_take_prepared();
  void wait_for_prepared();
  void rethrow_if_failed() const;
  void stop_workers();

  void attach_atlas(id<MTLDevice> value, uint32_t slot_capacity, Timer &timer);
  [[nodiscard]] uint32_t slot_for_variant(TileVariant variant) const;
  [[nodiscard]] std::vector<TileVariant>
  install_prepared(std::span<const uint8_t> pinned_slots, Timer &timer);
  void record_slot_use(std::span<const uint32_t> slots);
  void rebase_observer(ObserverLocation observer);
  [[nodiscard]] std::optional<float> sample_terrain(double easting, double northing);
  [[nodiscard]] TileManagerBindings bindings() const;
  [[nodiscard]] TileManagerStatistics statistics() const;

  void write_preparation_slots(std::span<const uint32_t> slots);
  void load_custom_vertices(
      std::span<const MetalTileBufferLoad> loads,
      std::span<const uint32_t> slots,
      std::span<const int32_t> elevation_bases,
      uint32_t vertex_value_count,
      Timer &timer
  );
  void load_compressed_lod_ranges(
      std::span<const MetalTileBufferLoad> loads,
      std::span<const NSUInteger> destination_offsets,
      id<MTLBuffer> destination,
      Timer &timer
  );
  [[nodiscard]] id<MTLCommandBuffer>
  submit_mipmaps(std::span<const uint32_t> slots, uint32_t cell_count, uint32_t level_count);
  void generate_mipmaps(
      std::span<const uint32_t> slots,
      uint32_t cell_count,
      uint32_t level_count,
      Timer &timer
  );
};

} // namespace panorama
