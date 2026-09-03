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
  /// Include num_steps.png and num_evaluations when diagnostic outputs are enabled.
  bool write_debugging_diagnostic;
  /// Write synthetic.png using configurable base colour and normal-based lighting.
  bool write_synthetic;
  /// Lighting parameters used only when `write_synthetic` is true.
  SyntheticRenderOptions synthetic_options;
  /// Fixed interval used by every scalar diagnostic or terrain colourmap.
  ScalarColourRange scalar_colour_range;

  /// Return whether any selected output consumes per-collision elevations.
  [[nodiscard]] bool requires_elevations() const {
    return (write_diagnostics && write_elevation_diagnostic) ||
           (write_synthetic && synthetic_options.raytraced_shadows) ||
           (write_synthetic && synthetic_options.colour_source == TerrainColourSource::Elevation);
  }

  /// Return whether any selected output consumes per-collision gradients.
  [[nodiscard]] bool requires_normals() const {
    return write_synthetic || (write_diagnostics && write_normal_diagnostic);
  }

  /// Return whether any selected output consumes per-collision debugging info.
  [[nodiscard]] bool requires_debugging_info() const {
    return (write_diagnostics && write_debugging_diagnostic);
  }
};

/// Trace one explicitly supplied per-pixel ray field and render selected outputs.
///
/// This convenience boundary constructs a short-lived `TerrainTraceSession`
/// for the command-line application. Interactive clients retain their session
/// directly so catalogue, atlas, and pipeline state survives between views.
void render_terrain(
    const RaytraceConfig &config,
    const RayField &field,
    const TerrainRenderOutputs &outputs
);

} // namespace panorama
