#pragma once

#include <cstdint>

namespace panorama {

/// Scalar field used as the synthetic terrain's base colour.
enum class TerrainColourSource : uint32_t {
  White = 0U,
  Distance = 1U,
  Elevation = 2U,
};

/// Built-in perceptual colourmaps available to GPU presentation kernels.
enum class PresetColourmap : uint32_t {
  Viridis = 0U,
  Plasma = 1U,
  Inferno = 2U,
  Magma = 3U,
  Cividis = 4U,
  Turbo = 5U,
};

/// Base-colour and lighting controls for synthetic terrain presentation.
struct SyntheticRenderOptions {
  /// Sun bearing in radians, clockwise from projected grid north.
  double sun_azimuth;
  /// Sun angle in radians above the local horizontal plane.
  double sun_elevation;
  /// Direction-independent light fraction in the inclusive range [0, 1].
  float ambient_light;
  /// White terrain, or a hit-only auto-normalised collision field.
  TerrainColourSource colour_source = TerrainColourSource::White;
  /// Preset applied when `colour_source` selects a scalar field.
  PresetColourmap colourmap = PresetColourmap::Viridis;
};

} // namespace panorama
