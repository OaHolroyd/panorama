#include "ray_projection_arguments.h"

#include "arguments.h"

#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

/// Convert user-facing degrees to the projection API's radians.
[[nodiscard]] double radians(double degrees) {
  return degrees * std::numbers::pi_v<double> / 180.0;
}

} // namespace

bool RayProjectionArguments::parse_option(std::string_view option, std::string_view value) {
  if (option == "--projection") {
    if (value == "angular") {
      mode_ = Mode::Angular;
    } else if (value == "camera") {
      mode_ = Mode::Camera;
    } else {
      throw std::invalid_argument(
          "Invalid value for --projection: " + std::string(value) + " (expected angular or camera)"
      );
    }
  } else if (option == "--azimuth-count") {
    azimuth_count_ = arguments::parse_uint32(value, option, false);
    angular_dimensions_specified_ = true;
  } else if (option == "--polar-count") {
    polar_count_ = arguments::parse_uint32(value, option, false);
    angular_dimensions_specified_ = true;
  } else if (option == "--image-width") {
    image_width_ = arguments::parse_uint32(value, option, false);
    camera_settings_specified_ = true;
  } else if (option == "--image-height") {
    image_height_ = arguments::parse_uint32(value, option, false);
    camera_settings_specified_ = true;
  } else if (option == "--horizontal-fov") {
    horizontal_field_of_view_ = arguments::parse_finite_double(value, option);
    camera_settings_specified_ = true;
  } else if (option == "--heading") {
    heading_ = arguments::parse_finite_double(value, option);
    camera_settings_specified_ = true;
  } else if (option == "--pitch") {
    pitch_ = arguments::parse_finite_double(value, option);
    camera_settings_specified_ = true;
  } else if (option == "--roll") {
    roll_ = arguments::parse_finite_double(value, option);
    camera_settings_specified_ = true;
  } else if (option == "--distortion-k1") {
    distortion_.radial_1 = arguments::parse_finite_double(value, option);
    distortion_specified_ = true;
    camera_settings_specified_ = true;
  } else if (option == "--distortion-k2") {
    distortion_.radial_2 = arguments::parse_finite_double(value, option);
    distortion_specified_ = true;
    camera_settings_specified_ = true;
  } else if (option == "--distortion-k3") {
    distortion_.radial_3 = arguments::parse_finite_double(value, option);
    distortion_specified_ = true;
    camera_settings_specified_ = true;
  } else if (option == "--distortion-p1") {
    distortion_.tangential_1 = arguments::parse_finite_double(value, option);
    distortion_specified_ = true;
    camera_settings_specified_ = true;
  } else if (option == "--distortion-p2") {
    distortion_.tangential_2 = arguments::parse_finite_double(value, option);
    distortion_specified_ = true;
    camera_settings_specified_ = true;
  } else {
    return false;
  }
  return true;
}

void RayProjectionArguments::validate() const {
  if (mode_ == Mode::Angular && camera_settings_specified_) {
    throw std::invalid_argument("Camera options require --projection camera");
  }
  if (mode_ == Mode::Camera && angular_dimensions_specified_) {
    throw std::invalid_argument("--azimuth-count and --polar-count require --projection angular");
  }
  if (mode_ == Mode::Camera &&
      (horizontal_field_of_view_ <= 0.0 || horizontal_field_of_view_ >= 180.0)) {
    throw std::out_of_range("--horizontal-fov must lie strictly between 0 and 180 degrees");
  }
}

RayField RayProjectionArguments::make_ray_field() const {
  if (mode_ == Mode::Angular) {
    return make_angular_ray_field(
        {azimuth_count_, polar_count_},
        {
            0.0,
            2.0 * std::numbers::pi_v<double>,
            -0.5 * std::numbers::pi_v<double>,
            0.5 * std::numbers::pi_v<double>,
        }
    );
  }

  const ImageSize image = {image_width_, image_height_};
  LensDistortion distortion = NoDistortion{};
  if (distortion_specified_) {
    distortion = distortion_;
  }
  return make_camera_ray_field(
      image,
      {
          {radians(heading_), radians(pitch_), radians(roll_)},
          CameraIntrinsics::from_horizontal_field_of_view(
              image,
              radians(horizontal_field_of_view_)
          ),
          distortion,
      }
  );
}

void RayProjectionArguments::print_settings() const {
  if (mode_ == Mode::Angular) {
    std::printf("%u azimuths x %u polars", azimuth_count_, polar_count_);
    return;
  }

  std::printf(
      "%u x %u camera, horizontal FOV %.6g deg, heading %.6g deg, "
      "pitch %.6g deg, roll %.6g deg",
      image_width_,
      image_height_,
      horizontal_field_of_view_,
      heading_,
      pitch_,
      roll_
  );
  if (!distortion_specified_) {
    std::printf(", lens distortion none");
    return;
  }
  std::printf(
      ", Brown-Conrady k1 %.6g, k2 %.6g, k3 %.6g, p1 %.6g, p2 %.6g",
      distortion_.radial_1,
      distortion_.radial_2,
      distortion_.radial_3,
      distortion_.tangential_1,
      distortion_.tangential_2
  );
}

void print_ray_projection_usage() {
  std::printf(
      "  --projection MODE     output projection: angular or camera (default: angular)\n"
      "\n"
      "Angular projection options:\n"
      "  --azimuth-count N     number of azimuth columns (default: 2048)\n"
      "  --polar-count N       number of polar rays per column (default: 1024)\n"
      "\n"
      "Camera projection options (angles are degrees):\n"
      "  --image-width N       output width in pixels (default: 1920)\n"
      "  --image-height N      output height in pixels (default: 1080)\n"
      "  --horizontal-fov D    horizontal field of view (default: 60)\n"
      "  --heading D           clockwise from grid north (default: 0)\n"
      "  --pitch D             optical-axis elevation (default: 0)\n"
      "  --roll D              rotation about the optical axis (default: 0)\n"
      "  --distortion-k1 V     Brown-Conrady radial coefficient (default: 0)\n"
      "  --distortion-k2 V     Brown-Conrady radial coefficient (default: 0)\n"
      "  --distortion-k3 V     Brown-Conrady radial coefficient (default: 0)\n"
      "  --distortion-p1 V     Brown-Conrady tangential coefficient (default: 0)\n"
      "  --distortion-p2 V     Brown-Conrady tangential coefficient (default: 0)\n"
      "\n"
  );
}

} // namespace panorama
