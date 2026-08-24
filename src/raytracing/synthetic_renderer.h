#pragma once

#include "ray_projection.h"

#include <cstdint>
#include <filesystem>
#include <span>

namespace panorama {

/// Lighting controls for the synthetic terrain image.
struct SyntheticRenderOptions {
  /// Sun bearing in radians, clockwise from projected grid north.
  double sun_azimuth;
  /// Sun angle in radians above the local horizontal plane.
  double sun_elevation;
  /// Direction-independent light fraction in the inclusive range [0, 1].
  float ambient_light;
};

/// Render white terrain with normal-based sunlight on a solid black background.
///
/// Packed gradients use the trace kernel's float16 east/north representation.
/// A positive finite distance marks a terrain collision; other rays render as
/// solid black. Every input span must contain exactly `image.width * image.height`
/// elements.
void write_synthetic_terrain_png(
    const std::filesystem::path &path,
    std::span<const uint32_t> packed_gradients,
    std::span<const float> distances,
    ImageSize image,
    const SyntheticRenderOptions &options
);

} // namespace panorama
