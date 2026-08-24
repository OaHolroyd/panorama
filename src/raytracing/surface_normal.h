#pragma once

#include <cstdint>

namespace panorama {

/// One upward-facing terrain normal in projected east/north/up coordinates.
struct SurfaceNormal {
  float east;
  float north;
  float up;
};

/// Reconstruct a unit normal from packed float16 dz/deast and dz/dnorth values.
///
/// The eastward gradient occupies the low 16 bits and the northward gradient
/// the high 16 bits, matching Metal's `as_type<uint>(half2(east, north))`.
[[nodiscard]] SurfaceNormal decode_surface_normal(uint32_t packed_gradients);

} // namespace panorama
