#pragma once

#include "raytrace_gpu.h"

namespace panorama {

struct MetalBvhStatistics {
  // Counts and durations accumulate over this backend's lifetime. GPU build
  // time includes bounds generation, construction and compaction; traversal
  // times exclude that work and the gaps between command buffers.
  uint64_t resident_bytes = 0U;
  uint64_t peak_bytes = 0U;
  uint64_t budget_bytes = 0U;
  uint64_t builds = 0U;
  uint64_t cache_hits = 0U;
  uint64_t evictions = 0U;
  uint64_t catalogue_bytes = 0U;
  uint64_t catalogue_builds = 0U;
  uint64_t instance_builds = 0U;
  uint64_t selection_passes = 0U;
  uint64_t trace_passes = 0U;
  uint64_t submissions = 0U;
  uint64_t scene_builds = 0U;
  uint64_t scene_passes = 0U;
  uint64_t scene_fallback_rays = 0U;
  uint64_t streaming_rounds = 0U;
  uint64_t streaming_groups = 0U;
  uint64_t streaming_rays = 0U;
  uint64_t lod_plan_changes = 0U;
  uint64_t lod_sources_changed = 0U;
  uint64_t scene_catalogue_invalidations = 0U;
  uint64_t scene_lod_invalidations = 0U;
  uint64_t scene_admission_invalidations = 0U;
  uint64_t scene_eviction_invalidations = 0U;
  // Small scene metadata/TLAS overhead, separate from detailed tile storage.
  uint64_t scene_bytes = 0U;
  uint64_t cached_tiles = 0U;
  uint64_t scene_tiles = 0U;
  // Synchronous shadow repair only; resident shadow GPU time belongs to the
  // combined producer. Caster builds are also included in the total builds.
  uint64_t shadow_passes = 0U;
  uint64_t shadow_tiles_built = 0U;
  uint64_t shadow_cache_fallbacks = 0U;
  double shadow_gpu_ms = 0.0;
  double build_gpu_ms = 0.0;
  double selection_gpu_ms = 0.0;
  double trace_gpu_ms = 0.0;
  double grouping_cpu_ms = 0.0;
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

  /// Encode a resident primary pass without committing or waiting. False
  /// means no scene is available and the caller must use synchronous tracing.
  bool encode_scene(id<MTLCommandBuffer> command, Timer &timer);
  /// Prepare a resident scene before encoding its GPU camera dependency.
  bool prepare_scene(Timer &timer);
  /// Inspect the primary missing-ray counter after the caller completes its command.
  bool scene_complete();
  /// Encode sun visibility using the same scene. Resources remain stable
  /// until completion; false from shadows_complete requires trace_shadows repair.
  bool encode_shadows(id<MTLCommandBuffer> command, double azimuth, double elevation);
  bool shadows_complete();
  /// Load GPU-requested shadow casters into the ordinary BVH cache and retry.
  /// False means the working set cannot fit; use exact software streaming.
  /// Primary outputs must be complete and no command may be in flight.
  bool trace_shadows(double azimuth, double elevation, Timer &timer);
  id<MTLBuffer> shadow_visibility() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
