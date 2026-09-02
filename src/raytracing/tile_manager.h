#pragma once

#include "resident_tile_cache.h"
#include "terrain_catalogue.h"
#include "tile_geometry.h"

#include <memory>
#include <span>
#include <vector>

namespace panorama {

class AsyncTilePreparer;
class Timer;

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
  [[nodiscard]] const std::vector<uint32_t> &lod_by_source() const;
  [[nodiscard]] uint32_t observer_source_index() const;
  [[nodiscard]] float pixel_angle() const;
  [[nodiscard]] uint32_t mipmap_value_count() const;
  [[nodiscard]] bool traces_quantized() const;
  [[nodiscard]] uint32_t slot_capacity() const;
  [[nodiscard]] ResidentTileCacheBindings bindings() const;

  [[nodiscard]] uint32_t slot_for_source(uint32_t source_index) const;
  void request(uint32_t source_index, float priority);
  [[nodiscard]] std::vector<TerrainTileVariant>
  install_prepared(std::span<const uint8_t> pinned_slots, Timer &timer);
  void wait_for_prepared();
  void record_slot_use(std::span<const uint32_t> slots);
  [[nodiscard]] uint32_t ensure_observer_resident(Timer &timer);
  void stop();

  [[nodiscard]] TilePreparationStatistics preparation_statistics() const;
  [[nodiscard]] ResidentTileCacheStatistics residency_statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
