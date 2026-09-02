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

/// Complete lifecycle of one source/LOD variant inside TileManager.
enum class TileLoadState : uint8_t {
  /// No request or resident payload currently exists.
  Unrequested,
  /// A priority-queue entry is waiting for a worker.
  Queued,
  /// A worker is reading metadata and opening the source file.
  Loading,
  /// The file and LOD record are waiting for an atlas slot.
  Prepared,
  /// The selected payload is published in the atlas.
  Resident,
};

/// Priority-queue item produced by HostFrontier through `TileManager::request`.
struct TileLoadRequest {
  /// Smaller distances sort ahead of farther terrain.
  float priority;
  /// Exact source and one-based LOD to prepare.
  TileVariant variant;
};

/// Turn `std::priority_queue` into a minimum-priority queue.
struct TileLoadRequestGreater {
  [[nodiscard]] bool operator()(const TileLoadRequest &left, const TileLoadRequest &right) const {
    return left.priority > right.priority;
  }
};

/// Worker output that deliberately contains no decoded terrain payload.
/// Metal I/O keeps the selected bytes on disk until an atlas slot is available.
struct PreparedTile {
  /// Source and LOD whose lifecycle is currently `Prepared`.
  TileVariant variant;
  /// Open handle retained across the worker-to-render-thread handoff.
  id<MTLIOFileHandle> metal_file;
  /// Validated byte range and geometry for the selected representation.
  MetalTileLod metal_lod;
};

/// Private implementation shared by tile-manager lifecycle and atlas code.
///
/// Worker threads access only the loader section under `loader_mutex`. Atlas
/// state is owned by the render thread and changes only between completed GPU
/// frontier passes.
struct TileManager::State {
  // Immutable discovery inputs and the observer-dependent LOD plan.
  /// Active configuration, updated when observer position or LOD scale changes.
  RaytraceConfig config;
  /// Finite, immutable source list and source-index lookup for this session.
  std::unique_ptr<TerrainCatalogue> catalogue;
  /// LOD-1 geometry which defines common tile width and atlas strides.
  std::unique_ptr<TileGeometry> origin;
  /// Currently selected one-based LOD for every catalogue source.
  std::vector<uint32_t> lod_by_source;
  /// Conservative angular span of a current output pixel, in radians.
  float pixel_angle = 0.0F;
  /// Source containing `config.observer` after the latest relocation.
  uint32_t observer_source_index = 0U;
  /// Number of maximum-hierarchy samples reserved per atlas slot.
  uint32_t mipmap_values = 0U;
  /// True when vertices and maxima remain uint16 throughout traversal.
  bool trace_quantized = false;

  // Asynchronous control-plane loading state. Terrain bytes are not read here.
  /// Metal device used by workers to open Metal-I/O file handles.
  id<MTLDevice> device = nil;
  /// Session timer receiving worker-side metadata/open measurements.
  Timer *loader_timer = nullptr;
  /// Bound on files waiting in `prepared_tiles`; normally the slot count.
  uint32_t prepared_capacity = 0U;
  /// Cooperative shutdown flag checked at every worker wait point.
  std::atomic<bool> stop_requested{false};
  /// Protects queues, lifecycle maps, errors, and loader counters below.
  mutable std::mutex loader_mutex;
  /// Wakes workers after a new or reprioritized request.
  std::condition_variable request_available;
  /// Wakes the render thread after preparation or failure.
  std::condition_variable prepared_available;
  /// Wakes workers after installation frees prepared-queue capacity.
  std::condition_variable prepared_space_available;
  /// Lazy-deletion priority queue; stale entries are rejected by the maps.
  std::priority_queue<TileLoadRequest, std::vector<TileLoadRequest>, TileLoadRequestGreater>
      requests;
  /// Authoritative lifecycle for every variant encountered by this manager.
  std::map<TileVariant, TileLoadState> load_states;
  /// Best queued priority, used to identify obsolete queue entries.
  std::map<TileVariant, float> queued_priorities;
  /// Lifetime deduplication marks used only to report request statistics.
  std::map<TileVariant, uint8_t> requested_before;
  /// Bounded worker-to-render-thread handoff queue.
  std::deque<PreparedTile> prepared_tiles;
  /// Workers which read headers/LOD tables and open selected files.
  std::vector<std::thread> workers;
  /// First worker exception, rethrown on the coordinating render thread.
  std::exception_ptr loader_error;
  /// Total calls reaching the internal request service.
  uint64_t request_count = 0U;
  /// Number of distinct variants ever requested.
  uint64_t unique_request_count = 0U;
  /// Requests for a variant already seen at least once.
  uint64_t duplicate_request_count = 0U;
  /// Actual worker count after hardware/source-count clamping.
  uint32_t worker_count = 0U;
  /// Prevents accidental construction of a second worker pool.
  bool workers_started = false;

  // GPU atlas, preparation pipelines, and render-thread residency policy.
  /// Concurrent queue used for selected terrain payload byte ranges.
  id<MTLIOCommandQueue> io_queue = nil;
  /// Ordered compute queue for conversion and mipmap generation.
  id<MTLCommandQueue> mipmap_queue = nil;
  /// Optional uint16-to-Float32 vertex conversion pipeline.
  id<MTLComputePipelineState> conversion_pipeline = nil;
  /// Fused kernels which build the first maximum-hierarchy levels.
  id<MTLComputePipelineState> initial_mipmap_pipeline = nil;
  /// Generic kernel used for later hierarchy reductions.
  id<MTLComputePipelineState> mipmap_pipeline = nil;
  /// Reference format/layout shared by compatible catalogue tiles.
  MetalTileHeader header_template = {};
  /// Fixed packed-record layout when source samples are uint16.
  QuantizedMetalTileRecordLayout quantized_record = {};
  /// True after atlas allocation and synchronous origin installation.
  bool atlas_attached = false;
  /// Projected western edge of catalogue column zero.
  double grid_origin_x = 0.0;
  /// Projected northern edge of catalogue row zero.
  double grid_origin_y = 0.0;
  /// Common physical width of every prepared tile.
  double tile_width = 0.0;
  /// Maximum-hierarchy element stride between atlas slots.
  uint32_t mip_count = 0U;
  /// LOD-1 vertex element stride between Float32 atlas slots.
  uint32_t vertex_count = 0U;
  /// Number of fixed slots permitted by cache budget and source count.
  uint32_t slot_capacity = 0U;
  /// Number of slots currently containing published variants.
  uint32_t resident_count = 0U;
  /// Monotonic stamp assigned whenever a slot participates in a frontier.
  uint64_t next_use_stamp = 1U;
  /// Cumulative successful publications, including the origin tile.
  uint64_t installation_count = 0U;
  /// Cumulative terrain payload bytes transferred through Metal I/O.
  uint64_t bytes_loaded_with_metal_io = 0U;
  /// Cumulative published variants displaced from atlas slots.
  uint64_t evictions = 0U;
  /// Conservative maximum hierarchies, one fixed LOD-1 stride per slot.
  id<MTLBuffer> mipmap_atlas = nil;
  /// Float32 vertices or packed uint16 records, depending on specialization.
  id<MTLBuffer> vertex_atlas = nil;
  /// Bounded conversion input used only by the expanded Float32 path.
  id<MTLBuffer> quantized_staging = nil;
  /// Slot indices consumed by batched conversion/mipmap compute kernels.
  id<MTLBuffer> preparation_slots = nil;
  /// GPU-visible `ResidentTile` array indexed by atlas slot.
  id<MTLBuffer> metadata_buffer = nil;
  /// Mapped host pointer into `metadata_buffer` for publication and rebasing.
  ResidentTile *metadata = nullptr;
  /// Forward mapping used by HostFrontier source residency queries.
  std::map<TileVariant, uint32_t> slot_by_variant;
  /// Reverse mapping used to identify the variant displaced from an LRU slot.
  std::vector<std::optional<TileVariant>> variant_by_slot;
  /// Per-slot LRU stamps; the smallest unpinned stamp is evicted first.
  std::vector<uint64_t> last_used;

  // Exact LOD-1 point inspection reuses resident data or one retained payload.
  /// Shared buffer for the most recently sampled nonresident source.
  id<MTLBuffer> sampled_vertices = nil;
  /// Catalogue source currently stored in `sampled_vertices`.
  std::optional<uint32_t> sampled_source_index;
  /// Per-source encoding and quantization base for the retained payload.
  MetalTileHeader sampled_header = {};

  // Loader lifecycle operations implemented in tile_manager.mm.
  /// Select exactly one LOD for every source using the current view policy.
  void rebuild_lod_plan(float angle);
  /// Create the bounded worker pool after GPU/file resources are available.
  void start_workers(uint32_t configured_workers);
  /// Deduplicate or improve the priority of one exact variant request.
  void request_tile(uint32_t source_index, uint32_t lod, float priority);
  /// Remove one prepared result without waiting for a worker.
  [[nodiscard]] std::optional<PreparedTile> try_take_prepared();
  /// Wait for at least one prepared result or rethrow a worker failure.
  void wait_for_prepared();
  /// Propagate a previously recorded worker failure, if any.
  void rethrow_if_failed() const;
  /// Signal and join the worker pool; repeated calls are harmless.
  void stop_workers();

  // Atlas and point-sampling operations implemented in tile_manager_atlas.mm.
  /// Allocate fixed-stride buffers and synchronously install the origin tile.
  void attach_atlas(id<MTLDevice> value, uint32_t slot_capacity, Timer &timer);
  /// Find an exact source/LOD variant or return the slot-capacity sentinel.
  [[nodiscard]] uint32_t slot_for_variant(TileVariant variant) const;
  /// Reserve LRU destinations, upload prepared payloads, and publish mappings.
  [[nodiscard]] std::vector<TileVariant>
  install_prepared(std::span<const uint8_t> pinned_slots, Timer &timer);
  /// Advance the LRU stamp for every slot used by the current frontier.
  void record_slot_use(std::span<const uint32_t> slots);
  /// Rewrite only observer-relative coordinates after an in-catalogue move.
  void rebase_observer(ObserverLocation observer);
  /// Bilinearly sample exact LOD-1 terrain without changing render LOD policy.
  [[nodiscard]] std::optional<float> sample_terrain(double easting, double northing);
  /// Return the complete Metal buffer ABI for a frontier dispatch.
  [[nodiscard]] TileManagerBindings bindings() const;
  /// Snapshot loader and atlas counters.
  [[nodiscard]] TileManagerStatistics statistics() const;

  // Low-level upload and maximum-hierarchy helpers.
  /// Upload a batch's destination slot numbers for preparation kernels.
  void write_preparation_slots(std::span<const uint32_t> slots);
  /// Load final-format vertices or stage and convert quantized vertices.
  void load_custom_vertices(
      std::span<const MetalTileBufferLoad> loads,
      std::span<const uint32_t> slots,
      std::span<const int32_t> elevation_bases,
      uint32_t vertex_value_count,
      Timer &timer
  );
  /// Work around unaligned compressed ranges by loading and trimming a prefix.
  void load_compressed_lod_ranges(
      std::span<const MetalTileBufferLoad> loads,
      std::span<const NSUInteger> destination_offsets,
      id<MTLBuffer> destination,
      Timer &timer
  );
  /// Encode all maximum-hierarchy levels for a batch without waiting.
  [[nodiscard]] id<MTLCommandBuffer>
  submit_mipmaps(std::span<const uint32_t> slots, uint32_t cell_count, uint32_t level_count);
  /// Submit, wait for, validate, and time one hierarchy-generation batch.
  void generate_mipmaps(
      std::span<const uint32_t> slots,
      uint32_t cell_count,
      uint32_t level_count,
      Timer &timer
  );
};

} // namespace panorama
