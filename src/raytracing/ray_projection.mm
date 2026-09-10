#include "ray_projection.h"
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace panorama {
namespace {
[[nodiscard]] size_t checked_pixel_count(ImageSize image) {
  const uint64_t count = static_cast<uint64_t>(image.width) * image.height;
  if (image.width == 0U || image.height == 0U ||
      count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::invalid_argument("Output image dimensions exceed the ray-field range");
  }
  return static_cast<size_t>(count);
}

} // namespace
CameraIntrinsics
CameraIntrinsics::from_horizontal_field_of_view(ImageSize image, double horizontal_field_of_view) {
  (void)checked_pixel_count(image);
  if (!std::isfinite(horizontal_field_of_view) || horizontal_field_of_view <= 0.0 ||
      horizontal_field_of_view >= std::numbers::pi_v<double>) {
    throw std::invalid_argument("Horizontal field of view must lie between zero and pi");
  }
  const double focal =
      0.5 * static_cast<double>(image.width) / std::tan(0.5 * horizontal_field_of_view);
  return {
      focal,
      focal,
      0.5 * static_cast<double>(image.width),
      0.5 * static_cast<double>(image.height),
  };
}

CameraIntrinsics
CameraIntrinsics::from_vertical_field_of_view(ImageSize image, double vertical_field_of_view) {
  (void)checked_pixel_count(image);
  if (!std::isfinite(vertical_field_of_view) || vertical_field_of_view <= 0.0 ||
      vertical_field_of_view >= std::numbers::pi_v<double>) {
    throw std::invalid_argument("Vertical field of view must lie between zero and pi");
  }
  const double focal =
      0.5 * static_cast<double>(image.height) / std::tan(0.5 * vertical_field_of_view);
  return {
      focal,
      focal,
      0.5 * static_cast<double>(image.width),
      0.5 * static_cast<double>(image.height),
  };
}

} // namespace panorama
