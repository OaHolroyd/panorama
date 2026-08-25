#pragma once

#include "synthetic_render_options.h"

namespace panorama {

/// Complete appearance configuration for one synthetic terrain image.
///
/// This value is deliberately independent of tracing and GPU resources. An
/// interactive caller can replace it at runtime and present the most recent
/// distance/elevation/gradient buffers again without retracing any rays.
struct TerrainPresentationSettings {
  /// Base colour, preset palette, sun direction, and ambient contribution.
  SyntheticRenderOptions appearance;
  /// Fixed interval mapped across the palette for distance/elevation colour.
  ScalarColourRange colour_range;
  /// Apply directional and ambient lighting from collision surface normals.
  /// When false, visible terrain is shown at its unshaded base colour.
  bool use_surface_normals = true;
};

} // namespace panorama
