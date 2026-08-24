#pragma once

#include "ray_projection.h"
#include "raytrace_config.h"
#include "synthetic_render_options.h"

namespace panorama {

/// Independently selectable products for one trace-and-render operation.
///
/// The requirement helpers are the single source of truth for optional trace
/// fields, ensuring presentation choices cannot allocate unused per-ray data.
struct TerrainRenderOutputs {
  /// Write distances.png plus enabled elevation and normal diagnostic fields.
  bool write_diagnostics;
  /// Include elevations.png when diagnostic outputs are enabled.
  bool write_elevation_diagnostic;
  /// Include normals.png when diagnostic outputs are enabled.
  bool write_normal_diagnostic;
  /// Write synthetic.png using white terrain and normal-based lighting.
  bool write_synthetic;
  /// Lighting parameters used only when `write_synthetic` is true.
  SyntheticRenderOptions synthetic_options;

  /// Return whether any selected output consumes per-collision elevations.
  [[nodiscard]] bool requires_elevations() const {
    return write_diagnostics && write_elevation_diagnostic;
  }

  /// Return whether any selected output consumes per-collision gradients.
  [[nodiscard]] bool requires_normals() const {
    return write_synthetic || (write_diagnostics && write_normal_diagnostic);
  }
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
