#pragma once

#include <cstdint>

namespace panorama {

/// Scalar field used as the synthetic terrain's base colour.
enum class TerrainColourSource : uint32_t {
  White = 0U,
  Distance = 1U,
  Elevation = 2U,
};

/// Built-in colourmaps available to GPU presentation kernels.
enum class PresetColourmap : uint32_t {
  Viridis = 0U,
  Plasma = 1U,
  Inferno = 2U,
  Magma = 3U,
  Cividis = 4U,
  Turbo = 5U,
  Viewfinder = 6U,
};

/// Transform applied after range normalisation and before palette lookup.
enum class ScalarColourScale : uint32_t {
  Linear = 0U,
  Logarithmic = 1U,
  SquareRoot = 2U,
  Quadratic = 3U,
};

/// Fixed scalar interval mapped onto the full span of a preset colourmap.
struct ScalarColourRange {
  float minimum;
  float maximum;
};
static_assert(sizeof(ScalarColourRange) == 2U * sizeof(float));

/// Base-colour and lighting controls for synthetic terrain presentation.
struct SyntheticRenderOptions {
  /// Sun bearing in radians, clockwise from projected grid north.
  double sun_azimuth;
  /// Sun angle in radians above the local horizontal plane.
  double sun_elevation;
  /// Direction-independent light fraction in the inclusive range [0, 1].
  float ambient_light;
  /// Strength of the directional Lambertian term in the inclusive range [0, 1].
  float diffusivity = 1.0F;
  /// White terrain, or a collision field normalised over a fixed range.
  TerrainColourSource colour_source = TerrainColourSource::White;
  /// Preset applied when `colour_source` selects a scalar field.
  PresetColourmap colourmap = PresetColourmap::Viridis;
  /// Distribution of the selected scalar interval across the palette.
  ScalarColourScale colour_scale = ScalarColourScale::Linear;
  /// Draw one-pixel black outlines at multiscale geometric surface separations.
  bool feature_outlines = false;
  /// Outline sensitivity in [0, 1], from only major divisions to fine detail.
  float feature_outline_detail = 0.7F;
  /// Occlude the directional term using one terrain ray towards the sun.
  bool raytraced_shadows = false;
};

} // namespace panorama
