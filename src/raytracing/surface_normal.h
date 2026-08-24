#pragma once

#include <cstdint>

namespace panorama {

/// One upward-facing terrain normal in projected east/north/up coordinates.
struct SurfaceNormal {
  float east;
  float north;
  float up;
};

/// Reconstruct a unit normal from the trace kernel's two packed float16 slopes.
[[nodiscard]] SurfaceNormal decode_surface_normal(uint32_t packed_gradients);

} // namespace panorama
