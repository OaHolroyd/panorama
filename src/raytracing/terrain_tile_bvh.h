#pragma once

#include "metal_bvh_resources.h"

namespace panorama {

/// Session-owned catalogue acceleration, independent of the within-tile tracer.
/// Bounds include all selected terrain LODs and conservative curvature guards.
class TerrainTileBvh {
public:
  explicit TerrainTileBvh(GpuRaytraceResources &gpu);
  ~TerrainTileBvh();
  bool
  prepare(TileManager &manager, ObserverLocation observer, const RaytraceParameters &parameters);
  id<MTLAccelerationStructure> acceleration() const;
  /// Height-bounded candidates used to find unloaded possible occluders.
  id<MTLAccelerationStructure> candidate_acceleration() const;
  id<MTLBuffer> tiles() const;
  /// Coarse transformed patches used by catalogue coverage traversal.
  id<MTLBuffer> patches() const;
  /// Fine transformed patches used by height-bounded candidate traversal.
  id<MTLBuffer> candidate_patches() const;
  std::span<const BvhTile> metadata() const;
  uint64_t bytes() const;
  uint64_t generation() const;
  double build_milliseconds() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
