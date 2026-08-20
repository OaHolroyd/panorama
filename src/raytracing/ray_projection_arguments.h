#pragma once

#include "ray_projection.h"

#include <cstdint>
#include <string_view>

namespace panorama {

/// Command-line configuration for one supported output projection.
///
/// The default preserves the existing 2048-by-1024 angular panorama. Camera
/// angles are accepted in degrees for usability, while `make_ray_field`
/// converts them to the projection API's radians. Projection-specific options
/// are tracked so settings for one mode cannot be silently ignored by another.
class RayProjectionArguments {
public:
  /// Consume a recognized option and return true, or leave it for the caller.
  [[nodiscard]] bool parse_option(std::string_view option, std::string_view value);

  /// Reject incompatible modes/options and invalid camera field-of-view values.
  void validate() const;

  /// Generate the row-major per-pixel ray field described by these arguments.
  [[nodiscard]] RayField make_ray_field() const;

  /// Print a concise, reproducible description inside the main settings line.
  void print_settings() const;

private:
  enum class Mode {
    Angular,
    Camera,
  };

  Mode mode_ = Mode::Angular;

  uint32_t azimuth_count_ = 2048U;
  uint32_t polar_count_ = 1024U;
  bool angular_dimensions_specified_ = false;

  uint32_t image_width_ = 1920U;
  uint32_t image_height_ = 1080U;
  double horizontal_field_of_view_ = 60.0;
  double heading_ = 0.0;
  double pitch_ = 0.0;
  double roll_ = 0.0;
  BrownConradyDistortion distortion_ = {};
  bool distortion_specified_ = false;
  bool camera_settings_specified_ = false;
};

/// Print the projection-mode portion of the command-line help text.
void print_ray_projection_usage();

} // namespace panorama
