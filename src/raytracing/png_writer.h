#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace panorama {

/// Encode a row-major GPU RGB8 readback as an opaque sRGB PNG.
///
/// Padding at the end of each row is accepted, although the current GPU
/// presentation readback is tightly packed.
void write_rgb_png(
    const std::filesystem::path &path,
    std::span<const uint8_t> bytes,
    uint32_t width,
    uint32_t height,
    size_t bytes_per_row
);

} // namespace panorama
