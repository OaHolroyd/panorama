#include "surface_normal.h"

#include <cmath>
#include <limits>

namespace panorama {
namespace {

/// Expand one IEEE 754 binary16 bit pattern without relying on a host-specific
/// half type. Trace output is finite, but handling special values keeps this
/// shared decoder well-defined for malformed data.
[[nodiscard]] float decode_binary16(uint16_t bits) {
  const bool negative = (bits & 0x8000U) != 0U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t significand = bits & 0x03ffU;
  float value = 0.0F;
  if (exponent == 0U) {
    value = std::ldexp(static_cast<float>(significand), -24);
  } else if (exponent == 0x1fU) {
    value = significand == 0U ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
  } else {
    value =
        std::ldexp(static_cast<float>(0x0400U + significand), static_cast<int>(exponent) - 25);
  }
  return negative ? -value : value;
}

} // namespace

SurfaceNormal decode_surface_normal(uint32_t packed_gradients) {
  const float east_gradient = decode_binary16(static_cast<uint16_t>(packed_gradients));
  const float north_gradient = decode_binary16(static_cast<uint16_t>(packed_gradients >> 16U));
  const float inverse_length =
      1.0F / std::sqrt(east_gradient * east_gradient + north_gradient * north_gradient + 1.0F);
  return {-east_gradient * inverse_length, -north_gradient * inverse_length, inverse_length};
}

} // namespace panorama
