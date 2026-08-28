#pragma once

#include <cstdint>
#include <variant>
#include <vector>

namespace panorama {

/// Pixel dimensions shared by a projection request and its generated ray field.
///
/// Pixels are stored in row-major order. Pixel coordinates use the usual image
/// convention: the origin is the top-left corner, x increases to the right,
/// and y increases downwards. Width and height must both be positive and their
/// product must fit the Metal ray-index representation.
struct ImageSize {
  uint32_t width;
  uint32_t height;
};

/// Orientation of a camera optical axis in the projected terrain frame.
///
/// All angles are radians. Heading is clockwise from grid north, pitch is
/// positive above the horizontal plane, and roll is applied last about the
/// pitched optical axis. Positive roll rotates a right-of-centre camera ray
/// towards the camera's up direction.
struct CameraOrientation {
  double heading;
  double pitch;
  double roll;
};

/// Pinhole-camera calibration expressed directly in output-pixel units.
///
/// `focal_x` and `focal_y` allow non-square pixels or an anisotropically scaled
/// image. `principal_x` and `principal_y` locate the optical axis relative to
/// the image's top-left corner; the centre of the top-left pixel is (0.5, 0.5).
/// Calibrated values may place the principal point outside the cropped output.
struct CameraIntrinsics {
  double focal_x;
  double focal_y;
  double principal_x;
  double principal_y;

  /// Construct centred, square-pixel intrinsics from a horizontal field of view.
  /// The field of view must be strictly between zero and pi radians.
  [[nodiscard]] static CameraIntrinsics
  from_horizontal_field_of_view(ImageSize image, double horizontal_field_of_view);

  /// Construct centred, square-pixel intrinsics from a vertical field of view.
  /// The field of view must be strictly between zero and pi radians.
  [[nodiscard]] static CameraIntrinsics
  from_vertical_field_of_view(ImageSize image, double vertical_field_of_view);
};

/// Select an ideal pinhole lens with no optical distortion.
struct NoDistortion {};

/// Brown-Conrady radial and tangential lens-distortion coefficients.
///
/// Coefficients operate on normalized pinhole coordinates, with x rightwards
/// and y downwards as in common camera-calibration packages. The forward model
/// maps an undistorted camera ray to a distorted image coordinate. Ray-field
/// construction numerically inverts that mapping for every output pixel.
struct BrownConradyDistortion {
  double radial_1;
  double radial_2;
  double radial_3;
  double tangential_1;
  double tangential_2;
};

/// Supported mappings between ideal pinhole rays and observed image pixels.
using LensDistortion = std::variant<NoDistortion, BrownConradyDistortion>;

/// Existing equally spaced azimuth/elevation output projection.
///
/// Columns advance from `azimuth_start` to `azimuth_end`; azimuth is clockwise
/// from grid north. Rows advance from `elevation_start` to
/// `elevation_end`; elevation is positive upwards. Samples lie at pixel centres
/// and the end values are exclusive, preserving the existing panorama output.
struct AngularProjection {
  double azimuth_start;
  double azimuth_end;
  double elevation_start;
  double elevation_end;
};

/// Perspective camera projection used for photo-mimic output.
///
/// Each distorted output-pixel centre is unprojected through the calibrated
/// lens, then rotated from camera coordinates into projected east/north/up
/// coordinates using `orientation`.
struct CameraProjection {
  CameraOrientation orientation;
  CameraIntrinsics intrinsics;
  LensDistortion distortion;
};

/// Projection models which can populate the projection-independent ray tracer.
using Projection = std::variant<AngularProjection, CameraProjection>;

/// Complete description of the per-pixel rays required by one output image.
struct RayFieldRequest {
  ImageSize image;
  Projection projection;
};

/// One projected ray direction expressed per unit horizontal distance.
///
/// `x` and `y` form a normalized east/north direction. `slope` is vertical
/// change per unit horizontal distance, while the reciprocals avoid repeated
/// DDA divisions in Metal. This layout is mirrored by Metal's `RayDirection`.
struct RayDirection {
  float x;
  float y;
  float inverse_x;
  float inverse_y;
  float slope;
};

/// Arbitrary row-major per-pixel rays for a rectangular output image.
///
/// After construction the tracing and frontier code use only this type; they
/// do not depend on the projection model which generated it.
struct RayField {
  ImageSize image;
  std::vector<RayDirection> rays;
  /// Conservative angular span of one output pixel, in radians. This is kept
  /// beside the generated rays so terrain LOD planning does not need to know
  /// which projection produced them.
  float minimum_pixel_angle;
};

/// Generate the existing equally spaced angular ray field.
[[nodiscard]] RayField make_angular_ray_field(ImageSize image, const AngularProjection &projection);

/// Generate perspective camera rays, including inverse lens distortion.
[[nodiscard]] RayField make_camera_ray_field(ImageSize image, const CameraProjection &projection);

/// Dispatch a runtime-selected projection to its ray-field generator.
[[nodiscard]] RayField make_ray_field(const RayFieldRequest &request);

} // namespace panorama
