#pragma once

#include "gpu_image_renderer.h"
#include "terrain_presentation_settings.h"
#include "terrain_trace_session.h"
#include <functional>

namespace panorama {
struct GpuTerrainFrameTiming {
  double wall_milliseconds = 0.0;
  // Sum of completed producer command durations. Synchronous BVH preparation
  // and separate terrain-loading/tracing submissions are not included here.
  double gpu_milliseconds = 0.0;
  uint32_t producer_submissions = 0U;
  bool streamed = false;
};

/// Render into an unpublished target. A resident frame encodes primary,
/// shadows, colouring and optional dependent work into one producer command.
/// On a cache miss, streaming repairs the trace and the image is regenerated
/// before returning. Pass nullptr for an appearance-only update.
GpuTerrainFrameTiming render_terrain_frame(
    TerrainTraceSession &trace,
    const RayField *field,
    GpuImageRenderer &image,
    const TerrainPresentationSettings &settings,
    const std::function<void(id<MTLCommandBuffer>)> &encode_dependent = {}
);
} // namespace panorama
