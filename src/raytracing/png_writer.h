#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace panorama {

/// An opaque sRGB colour returned by a colormap.
struct Rgb {
  uint8_t red;
  uint8_t green;
  uint8_t blue;
};

/// Map a normalised scalar in the inclusive range [0, 1] to an sRGB colour.
using Colormap = Rgb (*)(float normalised_value);

namespace colormaps {

/// Map black to white through neutral greys.
[[nodiscard]] Rgb grayscale(float normalised_value);

/// Map low-to-high values with the perceptually uniform viridis palette.
[[nodiscard]] Rgb viridis(float normalised_value);

/// Map low-to-high values with the conventional blue-cyan-yellow-red palette.
[[nodiscard]] Rgb jet(float normalised_value);

} // namespace colormaps

/// Write an already-coloured row-major RGB image as an opaque sRGB PNG.
void write_rgb_png(
    const std::filesystem::path &path,
    std::span<const Rgb> pixels,
    uint32_t width,
    uint32_t height
);

/// Write a row-major float32 array as an opaque sRGB PNG.
///
/// `values` must contain exactly `width * height` elements; row zero becomes
/// the top row of the PNG. Finite values are normalised over their own minimum
/// and maximum before `colormap` is called. NaN and infinite values are
/// written as black so sparse diagnostic fields remain viewable. Throws
/// std::invalid_argument for invalid dimensions or data and std::runtime_error
/// when ImageIO cannot create the PNG.
void write_colormapped_png(
    const std::filesystem::path &path,
    std::span<const float> values,
    uint32_t width,
    uint32_t height,
    Colormap colormap = colormaps::viridis
);

/// Write packed terrain-surface gradients as an RGB normal-map PNG.
///
/// Each uint32 contains east and north gradients as two IEEE float16 values,
/// matching the Metal trace output. Valid collisions reconstruct the upward
/// normal `normalize(-east, -north, 1)` and map its east/north/up components
/// from [-1, 1] to RGB. Pixels without a positive finite collision distance
/// are black. All spans must contain exactly `width * height` elements.
void write_surface_normals_png(
    const std::filesystem::path &path,
    std::span<const uint32_t> packed_gradients,
    std::span<const float> distances,
    uint32_t width,
    uint32_t height
);

} // namespace panorama
