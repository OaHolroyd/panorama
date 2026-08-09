#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

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
    const std::vector<float> &values,
    uint32_t width,
    uint32_t height,
    Colormap colormap = colormaps::viridis
);

} // namespace panorama
