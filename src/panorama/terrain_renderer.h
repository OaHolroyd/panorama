#pragma once

#include "ray_projection.h"
#include "raytrace_config.h"
#include "synthetic_render_options.h"

namespace panorama {

/// Independently selectable image products produced after tracing completes.
struct TerrainRenderOutputs {
  /// Write distances.png plus enabled elevation and normal diagnostic fields.
  bool write_diagnostics;
  /// Write synthetic.png using white terrain and normal-based lighting.
  bool write_synthetic;
  /// Lighting parameters used only when `write_synthetic` is true.
  SyntheticRenderOptions synthetic_options;
};

/// Trace an explicitly supplied per-pixel ray field and render selected outputs.
///
/// This is the command-line application's orchestration boundary. The tracing
/// and rendering implementations remain independently reusable beneath it.
void render_terrain(
    const RaytraceConfig &config,
    const RayField &field,
    const TerrainRenderOutputs &outputs
);

} // namespace panorama
