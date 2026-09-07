#pragma once

#include "raytrace_gpu.h"

namespace panorama {

struct MetalBvhStatistics {
  uint64_t resident_bytes = 0U;
  uint64_t peak_bytes = 0U;
  uint64_t budget_bytes = 0U;
  uint64_t builds = 0U;
  uint64_t cache_hits = 0U;
  uint64_t evictions = 0U;
  uint64_t catalogue_bytes = 0U;
};

/// Procedural terrain acceleration structures with bounded immutable tile storage.
/// Calls are serialized by TerrainTraceSession; every submission finishes before
/// an atlas slot, cached tile, parameter buffer, or ray output may be reused.
class MetalBvhTrace {
public:
  MetalBvhTrace(
      GpuRaytraceResources &gpu,
      uint32_t block_cells,
      GpuTraceOutputRequirements outputs,
      uint64_t cache_size_bytes
  );
  ~MetalBvhTrace();
  MetalBvhTrace(const MetalBvhTrace &) = delete;
  MetalBvhTrace &operator=(const MetalBvhTrace &) = delete;

  /// Update the small inter-tile hierarchy. Detailed BVHs use fixed tile anchors.
  void prepare(
      TileManager &tiles,
      ObserverLocation observer,
      const RaytraceParameters &parameters,
      Timer &timer
  );
  /// Trace primary rays into the session's ordinary distance/gradient buffers.
  void trace(const RaytraceParameters &parameters, Timer &timer);
  [[nodiscard]] MetalBvhStatistics statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
