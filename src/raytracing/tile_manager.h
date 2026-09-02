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
  uint32_t source_index;
  uint32_t lod;

  [[nodiscard]] bool operator<(const TileVariant &other) const {
    return source_index != other.source_index ? source_index < other.source_index : lod < other.lod;
  }
};

/// GPU buffers and layout selected by TileManager for a frontier dispatch.
struct TileManagerBindings {
  id<MTLBuffer> mipmap_atlas;
  id<MTLBuffer> vertex_atlas;
  id<MTLBuffer> metadata;
  QuantizedTerrainLayout quantized_layout;
};

/// Cumulative loading and residency counters owned by TileManager.
struct TileManagerStatistics {
  uint64_t requests;
  uint64_t unique_requests;
  uint64_t duplicate_requests;
  uint32_t worker_count;
  uint64_t installations;
  uint64_t bytes_loaded_with_metal_io;
  uint64_t evictions;
  uint32_t resident_tiles;
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
  TileManager(const RaytraceConfig &config, float initial_pixel_angle);
  ~TileManager();

  TileManager(const TileManager &) = delete;
  TileManager &operator=(const TileManager &) = delete;

  /// Allocate the atlas, install the origin tile, and start loading workers.
  void attach_gpu(id<MTLDevice> device, Timer &timer);

  /// Recompute the per-source LOD plan for this pixel angle.
  void set_pixel_angle(float pixel_angle);
  void set_lod_scale(float lod_scale);

  /// Rebase retained tile metadata and select the observer source.
  [[nodiscard]] bool relocate_observer(ObserverLocation observer);

  [[nodiscard]] const TerrainCatalogue &catalogue() const;
  [[nodiscard]] const std::vector<TerrainSource> &sources() const;
  [[nodiscard]] const TileGeometry &origin_geometry() const;
  /// Return the selected representation for one catalogue source.
  [[nodiscard]] uint32_t lod_for_source(uint32_t source_index) const;
  /// Width of one catalogue tile in the shared projected coordinate system.
  [[nodiscard]] float tile_width() const;
  [[nodiscard]] uint32_t observer_source_index() const;
  [[nodiscard]] float pixel_angle() const;
  [[nodiscard]] uint32_t mipmap_value_count() const;
  [[nodiscard]] bool traces_quantized() const;
  [[nodiscard]] uint32_t slot_capacity() const;
  [[nodiscard]] TileManagerBindings bindings() const;

  [[nodiscard]] uint32_t slot_for_source(uint32_t source_index) const;
  void request(uint32_t source_index, float priority);
  [[nodiscard]] std::vector<TileVariant>
  install_available(std::span<const uint8_t> pinned_slots, Timer &timer);
  void wait_for_available();
  void record_slot_use(std::span<const uint32_t> slots);
  [[nodiscard]] uint32_t ensure_observer_resident(Timer &timer);

  /// Bilinearly sample full-resolution terrain at one projected coordinate.
  /// A resident LOD-1 atlas slot is used when possible; otherwise the manager
  /// loads and retains one LOD-1 payload through its existing Metal-I/O queue.
  [[nodiscard]] std::optional<float> sample_terrain(double easting, double northing);
  void stop();

  [[nodiscard]] TileManagerStatistics statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
