#pragma once

#include "terrain_catalogue.h"
#include "tile_geometry.h"
#include "tile_manager_gpu.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace panorama {

class Timer;

/// Identity of one independently loadable representation of a terrain source.
struct TileVariant {
  /// Stable index into `TerrainCatalogue::sources()`.
  uint32_t source_index;
  /// One-based decimation level; LOD 1 is the source-resolution surface.
  uint32_t lod;

  /// Provide deterministic ordering for maps keyed by source and LOD.
  [[nodiscard]] bool operator<(const TileVariant &other) const {
    return source_index != other.source_index ? source_index < other.source_index : lod < other.lod;
  }
};

/// GPU buffers and layout selected by TileManager for a frontier dispatch.
struct TileManagerBindings {
  /// Per-slot conservative maximum hierarchy consumed by traversal.
  id<MTLBuffer> mipmap_atlas;
  /// Per-slot collision vertices, as Float32 values or packed uint16 records.
  id<MTLBuffer> vertex_atlas;
  /// Observer-relative origins, LODs, maxima, and grid keys for resident slots.
  id<MTLBuffer> metadata;
  /// Packed-record offsets; zeroed when the Float32 specialization is active.
  QuantizedTerrainLayout quantized_layout;
};

/// Cumulative loading and residency counters owned by TileManager.
struct TileManagerStatistics {
  /// All host requests, including repeats for already known variants.
  uint64_t requests;
  /// Variants requested at least once during this manager's lifetime.
  uint64_t unique_requests;
  /// Requests after the first request for the same variant.
  uint64_t duplicate_requests;
  /// Background workers created for metadata reads and file opening.
  uint32_t worker_count;
  /// Atlas publications, including the synchronously installed origin tile.
  uint64_t installations;
  /// Terrain payload bytes transferred through Metal I/O.
  uint64_t bytes_loaded_with_metal_io;
  /// Resident variants displaced by the LRU policy.
  uint64_t evictions;
  /// Slots currently containing a published terrain variant.
  uint32_t resident_tiles;
  /// Total number of fixed-stride slots allocated in the atlas.
  uint32_t slot_capacity;
};

/// Own prepared-terrain discovery, LOD selection, residency, and loading.
///
/// Discovery is deliberately separate from `attach_gpu`: ray resources need
/// the immutable source catalogue, while atlas allocation needs their device.
/// This keeps all tile lifetime and I/O policy behind one interface without a
/// construction-order dependency on the ray frontier.
class TileManager {
public:
  /// Discover the finite source catalogue and calculate the initial LOD plan.
  /// GPU resources are deliberately deferred until `attach_gpu`.
  TileManager(const RaytraceConfig &config, float initial_pixel_angle);

  /// Stop loading workers before releasing Metal and catalogue resources.
  ~TileManager();

  TileManager(const TileManager &) = delete;
  TileManager &operator=(const TileManager &) = delete;

  /// Allocate the atlas, install the origin tile, and start loading workers.
  void attach_gpu(id<MTLDevice> device, Timer &timer);

  /// Recompute the per-source LOD plan for this pixel angle.
  void set_pixel_angle(float pixel_angle);
  /// Change the footprint multiplier and rebuild the plan for the current view.
  void set_lod_scale(float lod_scale);

  /// Rebase retained tile metadata and select the observer source.
  /// Return false when the point lies outside this manager's finite catalogue.
  [[nodiscard]] bool relocate_observer(ObserverLocation observer);

  /// Immutable source catalogue shared with GPU resource construction.
  [[nodiscard]] const TerrainCatalogue &catalogue() const;
  /// Stable source array; indices in frontier work refer to this array.
  [[nodiscard]] const std::vector<TerrainSource> &sources() const;
  /// Reference geometry that defines LOD-1 dimensions and atlas strides.
  [[nodiscard]] const TileGeometry &origin_geometry() const;
  /// Return the selected representation for one catalogue source.
  [[nodiscard]] uint32_t lod_for_source(uint32_t source_index) const;
  /// Width of one catalogue tile in the shared projected coordinate system.
  [[nodiscard]] float tile_width() const;
  /// Catalogue source containing the current observer.
  [[nodiscard]] uint32_t observer_source_index() const;
  /// Conservative angular size currently used by the LOD policy.
  [[nodiscard]] float pixel_angle() const;
  /// Fixed per-slot element stride of the LOD-1 maximum hierarchy.
  [[nodiscard]] uint32_t mipmap_value_count() const;
  /// Whether traversal reads retained uint16 records rather than Float32.
  [[nodiscard]] bool traces_quantized() const;
  /// Atlas slot count; this value is also the nonresident slot sentinel.
  [[nodiscard]] uint32_t slot_capacity() const;
  /// Buffers and packed layout required by a terrain frontier dispatch.
  [[nodiscard]] TileManagerBindings bindings() const;

  /// Return the selected variant's slot, or `slot_capacity()` if nonresident.
  [[nodiscard]] uint32_t slot_for_source(uint32_t source_index) const;
  /// Queue the source's currently selected variant, improving priority as needed.
  void request(uint32_t source_index, float priority);
  /// Install every prepared variant for which an unpinned slot is available.
  /// Returned identities tell HostFrontier which waiting buckets can reactivate.
  [[nodiscard]] std::vector<TileVariant>
  install_available(std::span<const uint8_t> pinned_slots, Timer &timer);
  /// Block until a prepared variant or worker error becomes available.
  void wait_for_available();
  /// Refresh LRU stamps for slots read by the active GPU frontier.
  void record_slot_use(std::span<const uint32_t> slots);
  /// Load and return the current observer tile before a frame begins.
  [[nodiscard]] uint32_t ensure_observer_resident(Timer &timer);

  /// Bilinearly sample full-resolution terrain at one projected coordinate.
  /// A resident LOD-1 atlas slot is used when possible; otherwise the manager
  /// loads and retains one LOD-1 payload through its existing Metal-I/O queue.
  [[nodiscard]] std::optional<float> sample_terrain(double easting, double northing);
  /// Signal every worker and join it; safe to call repeatedly.
  void stop();

  /// Snapshot cumulative loader, I/O, and residency counters.
  [[nodiscard]] TileManagerStatistics statistics() const;

private:
  /// Private loader and atlas implementation shared by the two `.mm` files.
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
