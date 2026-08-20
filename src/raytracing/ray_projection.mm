#include "ray_projection.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace panorama {
namespace {

struct Direction3 {
  double east;
  double north;
  double up;
};

struct Coordinate2 {
  double x;
  double y;
};

[[nodiscard]] size_t checked_pixel_count(ImageSize image) {
  const uint64_t count = static_cast<uint64_t>(image.width) * image.height;
  if (image.width == 0U || image.height == 0U ||
      count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::invalid_argument("Output image dimensions exceed the ray-field range");
  }
  return static_cast<size_t>(count);
}

[[nodiscard]] RayDirection
make_horizontal_ray_direction(float x, float y, float slope) {
  const float horizontal_length = std::hypot(x, y);
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(slope) ||
      !std::isfinite(horizontal_length) || std::abs(horizontal_length - 1.0F) > 1e-4F) {
    throw std::invalid_argument("Projection produced an invalid terrain-ray direction");
  }
  return {
      x,
      y,
      x == 0.0F ? std::numeric_limits<float>::infinity() : 1.0F / x,
      y == 0.0F ? std::numeric_limits<float>::infinity() : 1.0F / y,
      slope,
  };
}

[[nodiscard]] RayDirection make_ray_direction(const Direction3 &direction) {
  const double horizontal_length = std::hypot(direction.east, direction.north);
  if (!std::isfinite(horizontal_length) || horizontal_length <= 1e-12 ||
      !std::isfinite(direction.up)) {
    throw std::invalid_argument(
        "Camera projection produced a vertical or non-finite terrain ray"
    );
  }
  const double horizontal_x = direction.east / horizontal_length;
  const double horizontal_y = direction.north / horizontal_length;
  const double slope = direction.up / horizontal_length;
  if (std::abs(slope) > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::invalid_argument("Camera projection produced an unrepresentable ray slope");
  }
  return make_horizontal_ray_direction(
      static_cast<float>(horizontal_x),
      static_cast<float>(horizontal_y),
      static_cast<float>(slope)
  );
}

[[nodiscard]] Coordinate2 distort(
    Coordinate2 undistorted,
    const BrownConradyDistortion &distortion
) {
  const double x2 = undistorted.x * undistorted.x;
  const double y2 = undistorted.y * undistorted.y;
  const double xy = undistorted.x * undistorted.y;
  const double radius2 = x2 + y2;
  const double radial =
      1.0 + radius2 *
                (distortion.radial_1 +
                 radius2 * (distortion.radial_2 + radius2 * distortion.radial_3));
  return {
      undistorted.x * radial + 2.0 * distortion.tangential_1 * xy +
          distortion.tangential_2 * (radius2 + 2.0 * x2),
      undistorted.y * radial + distortion.tangential_1 * (radius2 + 2.0 * y2) +
          2.0 * distortion.tangential_2 * xy,
  };
}

[[nodiscard]] Coordinate2 undistort(
    Coordinate2 distorted,
    const BrownConradyDistortion &distortion
) {
  Coordinate2 estimate = distorted;
  for (uint32_t iteration = 0U; iteration < 16U; iteration++) {
    const double x2 = estimate.x * estimate.x;
    const double y2 = estimate.y * estimate.y;
    const double xy = estimate.x * estimate.y;
    const double radius2 = x2 + y2;
    const double radial =
        1.0 + radius2 *
                  (distortion.radial_1 +
                   radius2 * (distortion.radial_2 + radius2 * distortion.radial_3));
    if (!std::isfinite(radial) || std::abs(radial) < 1e-12) {
      throw std::invalid_argument("Lens distortion cannot be inverted for an output pixel");
    }
    const double tangential_x = 2.0 * distortion.tangential_1 * xy +
                                distortion.tangential_2 * (radius2 + 2.0 * x2);
    const double tangential_y = distortion.tangential_1 * (radius2 + 2.0 * y2) +
                                2.0 * distortion.tangential_2 * xy;
    const Coordinate2 next = {
        (distorted.x - tangential_x) / radial,
        (distorted.y - tangential_y) / radial,
    };
    if (!std::isfinite(next.x) || !std::isfinite(next.y)) {
      throw std::invalid_argument("Lens distortion produced a non-finite camera ray");
    }
    const double change = std::hypot(next.x - estimate.x, next.y - estimate.y);
    estimate = next;
    if (change <= 1e-12 * (1.0 + std::hypot(estimate.x, estimate.y))) {
      break;
    }
  }
  const Coordinate2 check = distort(estimate, distortion);
  const double error = std::hypot(check.x - distorted.x, check.y - distorted.y);
  if (!std::isfinite(error) || error > 1e-8 * (1.0 + std::hypot(distorted.x, distorted.y))) {
    throw std::invalid_argument("Lens distortion failed to converge for an output pixel");
  }
  return estimate;
}

[[nodiscard]] Coordinate2
undistort(Coordinate2 distorted, const LensDistortion &distortion) {
  return std::visit(
      [distorted](const auto &model) -> Coordinate2 {
        using Model = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<Model, NoDistortion>) {
          return distorted;
        } else {
          return undistort(distorted, model);
        }
      },
      distortion
  );
}

void validate_camera_projection(const CameraProjection &projection) {
  const CameraOrientation &orientation = projection.orientation;
  const CameraIntrinsics &intrinsics = projection.intrinsics;
  if (!std::isfinite(orientation.heading) || !std::isfinite(orientation.pitch) ||
      !std::isfinite(orientation.roll) || !std::isfinite(intrinsics.focal_x) ||
      !std::isfinite(intrinsics.focal_y) || intrinsics.focal_x <= 0.0 ||
      intrinsics.focal_y <= 0.0 || !std::isfinite(intrinsics.principal_x) ||
      !std::isfinite(intrinsics.principal_y)) {
    throw std::invalid_argument("Camera projection parameters must be finite and valid");
  }
  std::visit(
      [](const auto &model) {
        using Model = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<Model, BrownConradyDistortion>) {
          if (!std::isfinite(model.radial_1) || !std::isfinite(model.radial_2) ||
              !std::isfinite(model.radial_3) || !std::isfinite(model.tangential_1) ||
              !std::isfinite(model.tangential_2)) {
            throw std::invalid_argument("Lens-distortion coefficients must be finite");
          }
        }
      },
      projection.distortion
  );
}

} // namespace

CameraIntrinsics CameraIntrinsics::from_horizontal_field_of_view(
    ImageSize image,
    double horizontal_field_of_view
) {
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

RayField make_angular_ray_field(ImageSize image, const AngularProjection &projection) {
  const size_t ray_count = checked_pixel_count(image);
  if (!std::isfinite(projection.azimuth_start) || !std::isfinite(projection.azimuth_end) ||
      !std::isfinite(projection.elevation_start) || !std::isfinite(projection.elevation_end)) {
    throw std::invalid_argument("Angular projection ranges must be finite");
  }
  RayField field = {image, std::vector<RayDirection>(ray_count)};
  const double azimuth_step =
      (projection.azimuth_end - projection.azimuth_start) / image.width;
  const double elevation_step =
      (projection.elevation_end - projection.elevation_start) / image.height;
  for (uint32_t row = 0U; row < image.height; row++) {
    const double elevation =
        projection.elevation_start + (static_cast<double>(row) + 0.5) * elevation_step;
    const double slope = std::tan(elevation);
    if (!std::isfinite(slope) || std::abs(slope) > std::numeric_limits<float>::max()) {
      throw std::invalid_argument("Elevation range produces an invalid float32 slope");
    }
    for (uint32_t column = 0U; column < image.width; column++) {
      const double azimuth =
          projection.azimuth_start + (static_cast<double>(column) + 0.5) * azimuth_step;
      field.rays[static_cast<size_t>(row) * image.width + column] =
          make_horizontal_ray_direction(
              static_cast<float>(std::sin(azimuth)),
              static_cast<float>(std::cos(azimuth)),
              static_cast<float>(slope)
          );
    }
  }
  return field;
}

RayField make_camera_ray_field(ImageSize image, const CameraProjection &projection) {
  const size_t ray_count = checked_pixel_count(image);
  validate_camera_projection(projection);
  RayField field = {image, std::vector<RayDirection>(ray_count)};

  const CameraOrientation &orientation = projection.orientation;
  const double sin_heading = std::sin(orientation.heading);
  const double cos_heading = std::cos(orientation.heading);
  const double sin_pitch = std::sin(orientation.pitch);
  const double cos_pitch = std::cos(orientation.pitch);
  const double sin_roll = std::sin(orientation.roll);
  const double cos_roll = std::cos(orientation.roll);
  const Direction3 heading_forward = {sin_heading, cos_heading, 0.0};
  const Direction3 heading_right = {cos_heading, -sin_heading, 0.0};
  const Direction3 pitched_forward = {
      cos_pitch * heading_forward.east,
      cos_pitch * heading_forward.north,
      sin_pitch,
  };
  const Direction3 pitched_up = {
      -sin_pitch * heading_forward.east,
      -sin_pitch * heading_forward.north,
      cos_pitch,
  };
  const Direction3 camera_right = {
      cos_roll * heading_right.east + sin_roll * pitched_up.east,
      cos_roll * heading_right.north + sin_roll * pitched_up.north,
      sin_roll * pitched_up.up,
  };
  const Direction3 camera_up = {
      -sin_roll * heading_right.east + cos_roll * pitched_up.east,
      -sin_roll * heading_right.north + cos_roll * pitched_up.north,
      cos_roll * pitched_up.up,
  };

  const CameraIntrinsics &intrinsics = projection.intrinsics;
  for (uint32_t row = 0U; row < image.height; row++) {
    for (uint32_t column = 0U; column < image.width; column++) {
      const Coordinate2 distorted = {
          (static_cast<double>(column) + 0.5 - intrinsics.principal_x) / intrinsics.focal_x,
          (static_cast<double>(row) + 0.5 - intrinsics.principal_y) / intrinsics.focal_y,
      };
      const Coordinate2 camera = undistort(distorted, projection.distortion);
      const Direction3 world = {
          pitched_forward.east + camera.x * camera_right.east - camera.y * camera_up.east,
          pitched_forward.north + camera.x * camera_right.north - camera.y * camera_up.north,
          pitched_forward.up + camera.x * camera_right.up - camera.y * camera_up.up,
      };
      field.rays[static_cast<size_t>(row) * image.width + column] =
          make_ray_direction(world);
    }
  }
  return field;
}

RayField make_ray_field(const RayFieldRequest &request) {
  return std::visit(
      [&request](const auto &projection) {
        using Model = std::decay_t<decltype(projection)>;
        if constexpr (std::is_same_v<Model, AngularProjection>) {
          return make_angular_ray_field(request.image, projection);
        } else {
          return make_camera_ray_field(request.image, projection);
        }
      },
      request.projection
  );
}

} // namespace panorama
