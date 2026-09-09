#pragma once

#include "ray_projection.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace panorama::app {

/// View-relative coordinates, with the origin at the top left. Keep requests
/// independent of drawable/ray resolution until the worker samples a frame.
struct InspectionLocation {
  double x;
  double y;
};

struct InspectionPixel {
  uint32_t x;
  uint32_t y;
};

[[nodiscard]] inline std::optional<InspectionPixel>
inspection_pixel(InspectionLocation location, ImageSize image) {
  if (image.width == 0 || image.height == 0 || !std::isfinite(location.x) ||
      !std::isfinite(location.y) || location.x < 0.0 || location.x > 1.0 || location.y < 0.0 ||
      location.y > 1.0)
    return std::nullopt;
  return InspectionPixel{
      static_cast<uint32_t>(std::min(location.x * image.width, double(image.width - 1))),
      static_cast<uint32_t>(std::min(location.y * image.height, double(image.height - 1))),
  };
}

} // namespace panorama::app
