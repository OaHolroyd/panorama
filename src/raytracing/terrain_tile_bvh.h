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
  id<MTLBuffer> tiles() const;
  std::span<const BvhTile> metadata() const;
  uint64_t bytes() const;
  uint64_t generation() const;
  double build_milliseconds() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
