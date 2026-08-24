#pragma once

namespace panorama {

/// Lighting controls for the synthetic terrain presentation kernel.
struct SyntheticRenderOptions {
  /// Sun bearing in radians, clockwise from projected grid north.
  double sun_azimuth;
  /// Sun angle in radians above the local horizontal plane.
  double sun_elevation;
  /// Direction-independent light fraction in the inclusive range [0, 1].
  float ambient_light;
};

} // namespace panorama
