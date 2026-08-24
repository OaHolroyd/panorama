#include "arguments.h"
#include "gpu_image_renderer.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "synthetic_render_options.h"
#include "terrain_presentation_settings.h"
#include "terrain_trace_session.h"
#include "timer.h"

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace panorama::app {
namespace {

constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
constexpr double kDegreesToRadians = std::numbers::pi / 180.0;
constexpr double kRadiansToDegrees = 180.0 / std::numbers::pi;
constexpr double kDefaultVerticalFieldOfView = 70.0 * kDegreesToRadians;

struct ViewerSettings {
  std::filesystem::path tile_dir = "data/swissalti3d-10-level-0";
  uint64_t tile_cache_size_bytes = 128ULL * kBytesPerMiB;
  uint32_t workers = 8U;
  float max_distance = 600'000.0F;
  bool retain_quantized = false;
  ObserverLocation observer = {2623452.4, 1100502.2, 3415.0};
  ImageSize image = {960U, 540U};
  double vertical_field_of_view = kDefaultVerticalFieldOfView;
  CameraOrientation orientation = {0.0, 0.0, 0.0};
  TerrainPresentationSettings presentation = {
      {
          225.0 * kDegreesToRadians,
          35.0 * kDegreesToRadians,
          0.28F,
          TerrainColourSource::White,
          PresetColourmap::Viridis,
      },
      {0.0F, 600'000.0F},
      true,
  };
};

[[nodiscard]] float parse_positive_float(std::string_view value, std::string_view option) {
  const double parsed = arguments::parse_finite_double(value, option);
  if (parsed <= 0.0 || parsed > std::numeric_limits<float>::max()) {
    throw std::out_of_range(std::string(option) + " must be a positive float32 value");
  }
  return static_cast<float>(parsed);
}

/// Parse an inspector range using a stable, locale-independent syntax.
/// Commas are treated purely as digit-group separators, so "10,000" and
/// "10000" have the same value; a period is the only decimal separator.
[[nodiscard]] std::optional<double> parse_range_value(NSString *input) {
  NSString *normalised = [[input stringByReplacingOccurrencesOfString:@"," withString:@""]
      stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
  const char *characters = normalised.UTF8String;
  if (characters == nullptr || characters[0] == '\0') {
    return std::nullopt;
  }

  const std::string_view text(characters);
  double value = 0.0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value, std::chars_format::general);
  if (error != std::errc() || end != text.data() + text.size() || !std::isfinite(value)) {
    return std::nullopt;
  }
  return value;
}

/// Parse a positive image dimension with optional thousands separators.
[[nodiscard]] std::optional<uint32_t> parse_image_dimension(NSString *input) {
  NSString *normalised = [[input stringByReplacingOccurrencesOfString:@"," withString:@""]
      stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
  const char *characters = normalised.UTF8String;
  if (characters == nullptr || characters[0] == '\0') {
    return std::nullopt;
  }

  const std::string_view text(characters);
  uint32_t value = 0U;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size() || value == 0U) {
    return std::nullopt;
  }
  return value;
}

/// Avoid populating the editor through NSTextField.doubleValue, whose
/// formatting follows the user's locale and may use commas as decimal marks.
[[nodiscard]] NSString *format_range_value(double value) {
  return [NSString stringWithFormat:@"%.9g", value];
}

void print_usage(const char *program) {
  std::printf(
      "usage: %s [options]\n"
      "  --tile-dir DIR        prepared level-0 tile directory\n"
      "  --tile-cache-mib N    resident terrain-cache budget (default: 128)\n"
      "  --workers N           tile preparation workers (default: 8)\n"
      "  --max-distance M      horizontal range in metres (default: 600000)\n"
      "  --retain-quantized    keep uint16 terrain quantized in the GPU atlas\n"
      "  --easting M           fixed observer easting (default: 2623452.4)\n"
      "  --northing M          fixed observer northing (default: 1100502.2)\n"
      "  --elevation M         fixed observer elevation (default: 3415)\n"
      "  --image-width N       internal render width (default: 960)\n"
      "  --image-height N      internal render height (default: 540)\n"
      "  --vertical-fov D      vertical camera field of view in degrees (default: 70)\n"
      "  --heading D           initial heading clockwise from north (default: 0)\n"
      "  --pitch D             initial pitch above the horizon (default: 0)\n"
      "  --help                show this message\n"
      "\n"
      "Drag with the mouse or use WASD/arrow keys to look around; scroll to zoom.\n",
      program
  );
}

[[nodiscard]] ViewerSettings parse_arguments(int argc, const char *argv[]) {
  ViewerSettings settings;
  for (int index = 1; index < argc; index++) {
    const std::string_view option = argv[index];
    if (option == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (option == "--retain-quantized") {
      settings.retain_quantized = true;
      continue;
    }
    const std::string_view value = arguments::option_value(argc, argv, index, option);
    if (option == "--tile-dir") {
      settings.tile_dir = value;
    } else if (option == "--tile-cache-mib") {
      const uint64_t size = arguments::parse_uint64(value, option);
      if (size == 0U || size > std::numeric_limits<uint64_t>::max() / kBytesPerMiB) {
        throw std::out_of_range("Tile cache is outside the supported byte range");
      }
      settings.tile_cache_size_bytes = size * kBytesPerMiB;
    } else if (option == "--workers") {
      settings.workers = arguments::parse_uint32(value, option, true);
    } else if (option == "--max-distance") {
      settings.max_distance = parse_positive_float(value, option);
    } else if (option == "--easting") {
      settings.observer.easting = arguments::parse_finite_double(value, option);
    } else if (option == "--northing") {
      settings.observer.northing = arguments::parse_finite_double(value, option);
    } else if (option == "--elevation") {
      settings.observer.elevation = arguments::parse_finite_double(value, option);
    } else if (option == "--image-width") {
      settings.image.width = arguments::parse_uint32(value, option, false);
    } else if (option == "--image-height") {
      settings.image.height = arguments::parse_uint32(value, option, false);
    } else if (option == "--vertical-fov") {
      const double degrees = arguments::parse_finite_double(value, option);
      if (degrees <= 0.0 || degrees >= 180.0) {
        throw std::out_of_range("Vertical field of view must be between 0 and 180 degrees");
      }
      settings.vertical_field_of_view = degrees * kDegreesToRadians;
    } else if (option == "--heading") {
      settings.orientation.heading =
          arguments::parse_finite_double(value, option) * kDegreesToRadians;
    } else if (option == "--pitch") {
      const double degrees = arguments::parse_finite_double(value, option);
      if (degrees < -85.0 || degrees > 85.0) {
        throw std::out_of_range("Pitch must be between -85 and 85 degrees");
      }
      settings.orientation.pitch = degrees * kDegreesToRadians;
    } else {
      throw std::invalid_argument("Unknown option: " + std::string(option));
    }
  }
  const uint64_t pixels = static_cast<uint64_t>(settings.image.width) * settings.image.height;
  if (pixels == 0U || pixels > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("Viewer image dimensions exceed the Metal ray-index range");
  }
  // The interactive palette defaults to the same useful interval as tracing.
  settings.presentation.colour_range.maximum = settings.max_distance;
  return settings;
}

[[nodiscard]] RayField
make_view(ImageSize image, CameraOrientation orientation, double vertical_field_of_view) {
  return make_camera_ray_field(
      image,
      {
          orientation,
          CameraIntrinsics::from_vertical_field_of_view(image, vertical_field_of_view),
          NoDistortion{},
      }
  );
}

/// One output pixel selected in the top-left-origin ray image.
struct InspectionPixel {
  uint32_t x;
  uint32_t y;
};

/// Immutable values sampled from one completed raytrace revision.
struct PointInspection {
  InspectionPixel pixel;
  uint64_t revision;
  bool hit;
  float distance;
  float elevation;
  double easting;
  double northing;
  float slope_degrees;
  float aspect_degrees;
};

/// Screen-space position of a locked world point in the current camera view.
/// Off-screen projections retain a direction from the image centre so the UI
/// can place a directional marker at the nearest edge.
struct LockedPointProjection {
  bool onscreen;
  double pixel_x;
  double pixel_y;
  double direction_x;
  double direction_y;
};

/// Reproject a sampled terrain location through the viewer's ideal pinhole
/// camera. The curvature adjustment reconstructs the apparent vertical ray
/// displacement used when the original terrain collision was recorded.
[[nodiscard]] LockedPointProjection project_locked_point(
    const PointInspection &point,
    ObserverLocation observer,
    ImageSize image,
    double vertical_field_of_view,
    CameraOrientation orientation
) {
  const double east = point.easting - observer.easting;
  const double north = point.northing - observer.northing;
  const double horizontal_distance = std::hypot(east, north);
  const double up = point.elevation - observer.elevation -
                    kCurvatureCoefficient * horizontal_distance * horizontal_distance;

  const double sin_heading = std::sin(orientation.heading);
  const double cos_heading = std::cos(orientation.heading);
  const double sin_pitch = std::sin(orientation.pitch);
  const double cos_pitch = std::cos(orientation.pitch);
  const double sin_roll = std::sin(orientation.roll);
  const double cos_roll = std::cos(orientation.roll);

  const double forward_east = cos_pitch * sin_heading;
  const double forward_north = cos_pitch * cos_heading;
  const double forward_up = sin_pitch;
  const double pitched_up_east = -sin_pitch * sin_heading;
  const double pitched_up_north = -sin_pitch * cos_heading;
  const double right_east = cos_roll * cos_heading + sin_roll * pitched_up_east;
  const double right_north = -cos_roll * sin_heading + sin_roll * pitched_up_north;
  const double right_up = sin_roll * cos_pitch;
  const double camera_up_east = -sin_roll * cos_heading + cos_roll * pitched_up_east;
  const double camera_up_north = sin_roll * sin_heading + cos_roll * pitched_up_north;
  const double camera_up_up = cos_roll * cos_pitch;

  const double forward = east * forward_east + north * forward_north + up * forward_up;
  const double right = east * right_east + north * right_north + up * right_up;
  const double down = -(east * camera_up_east + north * camera_up_north + up * camera_up_up);
  const CameraIntrinsics intrinsics =
      CameraIntrinsics::from_vertical_field_of_view(image, vertical_field_of_view);

  double pixel_x = intrinsics.principal_x;
  double pixel_y = intrinsics.principal_y;
  if (forward > 1e-9) {
    pixel_x += intrinsics.focal_x * right / forward;
    pixel_y += intrinsics.focal_y * down / forward;
  }
  const bool finite = std::isfinite(pixel_x) && std::isfinite(pixel_y);
  const bool onscreen = finite && forward > 0.0 && pixel_x >= 0.0 && pixel_y >= 0.0 &&
                        pixel_x < image.width && pixel_y < image.height;
  if (forward > 1e-9 && finite) {
    return {
        onscreen,
        pixel_x,
        pixel_y,
        pixel_x - intrinsics.principal_x,
        pixel_y - intrinsics.principal_y,
    };
  }

  // A point behind the image plane has no finite pinhole coordinate. Camera
  // right/down components still give a useful direction in which to turn.
  double direction_x = right;
  double direction_y = down;
  if (!std::isfinite(direction_x) || !std::isfinite(direction_y) ||
      std::hypot(direction_x, direction_y) < 1e-12) {
    direction_x = 1.0;
    direction_y = 0.0;
  }
  return {false, pixel_x, pixel_y, direction_x, direction_y};
}

struct PresentedFrame {
  id<MTLTexture> texture;
  ImageSize image;
  CameraOrientation orientation;
  double vertical_field_of_view;
  uint64_t revision;
  double milliseconds;
  std::string error;
  std::optional<PointInspection> inspection;
  uint64_t inspection_sequence;
  uint64_t inspection_request_token;
};

/// Decode one IEEE float16 value emitted by Metal without depending on a SIMD
/// vector ABI shared between C++ and Metal.
[[nodiscard]] float float_from_half_bits(uint16_t bits) {
  _Float16 value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<float>(value);
}

/// Serial background renderer which coalesces input to the latest camera view.
class ViewerRenderer {
public:
  explicit ViewerRenderer(ViewerSettings settings)
      : settings_(std::move(settings)), requested_orientation_(settings_.orientation),
        requested_vertical_field_of_view_(settings_.vertical_field_of_view),
        requested_image_(settings_.image), requested_presentation_(settings_.presentation),
        presented_vertical_field_of_view_(settings_.vertical_field_of_view),
        presented_image_(settings_.image) {
    RayField initial_field =
        make_view(settings_.image, settings_.orientation, settings_.vertical_field_of_view);
    const RaytraceConfig config = {
        settings_.tile_dir,
        settings_.observer,
        settings_.max_distance,
        0U,
        settings_.tile_cache_size_bytes,
        settings_.workers,
        settings_.retain_quantized,
    };
    trace_ = std::make_unique<TerrainTraceSession>(
        config,
        initial_field,
        GpuTraceOutputRequirements{true, true}
    );
    presentation_ = std::make_unique<GpuImageRenderer>(
        trace_->device(),
        trace_->command_queue(),
        trace_->library(),
        settings_.image,
        GpuPresentationRequirements{false, false, true, true, false},
        MTLPixelFormatBGRA8Unorm
    );
    current_field_ = std::move(initial_field);
    worker_ = std::thread([this] { render_loop(); });
    request_view(settings_.orientation, settings_.vertical_field_of_view, settings_.image);
  }

  ViewerRenderer(const ViewerRenderer &) = delete;
  ViewerRenderer &operator=(const ViewerRenderer &) = delete;

  ~ViewerRenderer() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  /// Request a new camera trace; intermediate input events are coalesced.
  void request_view(CameraOrientation orientation, double vertical_field_of_view, ImageSize image) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_orientation_ = orientation;
      requested_vertical_field_of_view_ = vertical_field_of_view;
      requested_image_ = image;
      requested_revision_++;
      trace_pending_ = true;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Re-present the completed trace with new appearance settings.
  void request_presentation(TerrainPresentationSettings presentation) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_presentation_ = presentation;
      requested_revision_++;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Coalesce hover events to the latest output pixel. A missing pixel clears
  /// the published sample when inspection is disabled or leaves the image.
  uint64_t request_inspection(std::optional<InspectionPixel> pixel) {
    uint64_t token = 0U;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_inspection_ = pixel;
      requested_inspection_token_++;
      token = requested_inspection_token_;
      inspection_pending_ = true;
    }
    changed_.notify_one();
    return token;
  }

  [[nodiscard]] PresentedFrame presented_frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        presented_texture_,
        presented_image_,
        presented_orientation_,
        presented_vertical_field_of_view_,
        presented_revision_,
        frame_ms_,
        error_,
        presented_inspection_,
        presented_inspection_sequence_,
        presented_inspection_token_,
    };
  }

  [[nodiscard]] id<MTLDevice> device() const { return trace_->device(); }
  [[nodiscard]] id<MTLCommandQueue> command_queue() const { return trace_->command_queue(); }
  [[nodiscard]] ImageSize image() const { return settings_.image; }
  [[nodiscard]] ObserverLocation observer() const { return settings_.observer; }
  [[nodiscard]] double initial_vertical_field_of_view() const {
    return settings_.vertical_field_of_view;
  }
  [[nodiscard]] float max_distance() const { return settings_.max_distance; }
  [[nodiscard]] CameraOrientation initial_orientation() const { return settings_.orientation; }
  [[nodiscard]] TerrainPresentationSettings initial_presentation() const {
    return settings_.presentation;
  }

private:
  [[nodiscard]] PointInspection inspect_pixel(InspectionPixel pixel, uint64_t revision) const {
    if (pixel.x >= current_field_.image.width || pixel.y >= current_field_.image.height) {
      throw std::out_of_range("Inspection pixel lies outside the ray image");
    }
    const size_t index =
        static_cast<size_t>(pixel.y) * static_cast<size_t>(current_field_.image.width) + pixel.x;
    const auto *distances = static_cast<const float *>(trace_->distances().contents);
    const auto *elevations = static_cast<const float *>(trace_->elevations().contents);
    const auto *gradients = static_cast<const uint32_t *>(trace_->surface_gradients().contents);
    if (distances == nullptr || elevations == nullptr || gradients == nullptr ||
        index >= current_field_.rays.size()) {
      throw std::runtime_error("Could not map point-inspection buffers");
    }

    PointInspection result = {pixel, revision, false, 0.0F, 0.0F, 0.0, 0.0, 0.0F, 0.0F};
    const float distance = distances[index];
    if (!(distance > 0.0F) || !std::isfinite(distance)) {
      return result;
    }

    const RayDirection &ray = current_field_.rays[index];
    const uint32_t packed_gradients = gradients[index];
    const float east_gradient =
        float_from_half_bits(static_cast<uint16_t>(packed_gradients & 0xffffU));
    const float north_gradient =
        float_from_half_bits(static_cast<uint16_t>(packed_gradients >> 16U));
    const float slope = std::atan(std::hypot(east_gradient, north_gradient));
    double aspect = std::atan2(-east_gradient, -north_gradient) * kRadiansToDegrees;
    if (aspect < 0.0) {
      aspect += 360.0;
    }
    result.hit = true;
    result.distance = distance;
    result.elevation = elevations[index];
    result.easting =
        settings_.observer.easting + static_cast<double>(distance) * static_cast<double>(ray.x);
    result.northing =
        settings_.observer.northing + static_cast<double>(distance) * static_cast<double>(ray.y);
    result.slope_degrees = slope * static_cast<float>(kRadiansToDegrees);
    result.aspect_degrees = static_cast<float>(aspect);
    return result;
  }

  void render_loop() {
    while (true) {
      CameraOrientation orientation = {};
      double vertical_field_of_view = 0.0;
      ImageSize image = {};
      TerrainPresentationSettings presentation = {};
      uint64_t revision = 0U;
      bool trace_requested = false;
      bool presentation_requested = false;
      bool inspection_requested = false;
      std::optional<InspectionPixel> inspection_pixel;
      uint64_t inspection_token = 0U;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] {
          return stopping_ || trace_pending_ || presentation_pending_ || inspection_pending_;
        });
        if (stopping_) {
          return;
        }
        orientation = requested_orientation_;
        vertical_field_of_view = requested_vertical_field_of_view_;
        image = requested_image_;
        presentation = requested_presentation_;
        revision = requested_revision_;
        trace_requested = trace_pending_;
        presentation_requested = presentation_pending_;
        inspection_requested = inspection_pending_;
        inspection_pixel = requested_inspection_;
        inspection_token = requested_inspection_token_;
        trace_pending_ = false;
        presentation_pending_ = false;
        inspection_pending_ = false;
      }

      try {
        @autoreleasepool {
          const auto started = std::chrono::steady_clock::now();
          if (trace_requested) {
            RayField field = make_view(image, orientation, vertical_field_of_view);
            trace_->trace(field);
            current_field_ = std::move(field);
          }

          if (presentation_requested) {
            Timer timer("GPU presentation");
            presentation_->resize(current_field_.image);
            const id<MTLBuffer> colour_values =
                presentation.appearance.colour_source == TerrainColourSource::Elevation
                    ? trace_->elevations()
                    : trace_->distances();
            presentation_->render_synthetic(
                trace_->surface_gradients(),
                trace_->distances(),
                colour_values,
                presentation.appearance,
                presentation.colour_range,
                presentation.use_surface_normals,
                timer
            );
          }
          const double milliseconds =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                  .count();
          const bool publish_inspection =
              inspection_requested || (inspection_pixel.has_value() && presentation_requested);
          std::optional<PointInspection> inspection;
          if (publish_inspection && inspection_pixel.has_value()) {
            inspection = inspect_pixel(*inspection_pixel, revision);
          }

          std::lock_guard<std::mutex> lock(mutex_);
          if (presentation_requested) {
            presented_texture_ = presentation_->texture();
            presented_image_ = current_field_.image;
            presented_orientation_ = orientation;
            presented_vertical_field_of_view_ = vertical_field_of_view;
            presented_revision_ = revision;
            // The title reports camera-update throughput. A cheap appearance-only
            // pass should not replace it with a misleadingly high frame rate.
            if (trace_requested) {
              frame_ms_ = milliseconds;
            }
          }
          if (publish_inspection) {
            presented_inspection_ = inspection;
            presented_inspection_sequence_++;
            presented_inspection_token_ = inspection_token;
          }
        }
      } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = exception.what();
        return;
      }
    }
  }

  ViewerSettings settings_;
  std::unique_ptr<TerrainTraceSession> trace_;
  std::unique_ptr<GpuImageRenderer> presentation_;
  RayField current_field_;
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  CameraOrientation requested_orientation_ = {};
  double requested_vertical_field_of_view_ = 0.0;
  ImageSize requested_image_ = {};
  TerrainPresentationSettings requested_presentation_ = {};
  std::optional<InspectionPixel> requested_inspection_;
  CameraOrientation presented_orientation_ = {};
  double presented_vertical_field_of_view_ = 0.0;
  ImageSize presented_image_ = {};
  std::optional<PointInspection> presented_inspection_;
  id<MTLTexture> presented_texture_;
  uint64_t requested_revision_ = 0U;
  uint64_t requested_inspection_token_ = 0U;
  uint64_t presented_revision_ = 0U;
  uint64_t presented_inspection_sequence_ = 0U;
  uint64_t presented_inspection_token_ = 0U;
  double frame_ms_ = 0.0;
  std::string error_;
  bool trace_pending_ = false;
  bool presentation_pending_ = false;
  bool inspection_pending_ = false;
  bool stopping_ = false;
};

} // namespace
} // namespace panorama::app

@class PanoramaController;
@class ViewerOverlayView;
@class AspectFitContainerView;

/// Non-interactive symbol layered over the Metal view for a locked point.
@interface LockedPointMarkerView : NSImageView
@end

@implementation LockedPointMarkerView
- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}
@end

@interface PanoramaView : MTKView {
@private
  NSPoint _lastMouseLocation;
  NSTrackingArea *_inspectionTrackingArea;
  LockedPointMarkerView *_lockedPointMarker;
  double _lockedPointPixelX;
  double _lockedPointPixelY;
  double _lockedPointDirectionX;
  double _lockedPointDirectionY;
  bool _pointInspectionEnabled;
  bool _lockedPointIndicatorActive;
  bool _lockedPointOnscreen;
}
@property(nonatomic, weak) PanoramaController *panoramaController;
- (void)setPointInspectionEnabled:(bool)enabled;
- (void)setLockedPointIndicator:(std::optional<panorama::app::LockedPointProjection>)projection;
@end

@interface PanoramaController : NSObject <MTKViewDelegate, NSTextFieldDelegate> {
@private
  panorama::app::ViewerRenderer *_renderer;
  __weak NSWindow *_window;
  __weak PanoramaView *_panoramaView;
  __weak ViewerOverlayView *_overlayView;
  __weak AspectFitContainerView *_aspectFitView;
  panorama::CameraOrientation _orientation;
  double _verticalFieldOfView;
  panorama::ImageSize _image;
  panorama::TerrainPresentationSettings _presentation;
  std::optional<panorama::app::PointInspection> _lockedPoint;
  NSPopUpButton *_colourSourceControl;
  NSPopUpButton *_colourmapControl;
  NSTextField *_minimumControl;
  NSTextField *_maximumControl;
  NSSlider *_zoomControl;
  NSTextField *_zoomValueLabel;
  NSSlider *_panningSensitivityControl;
  NSTextField *_panningSensitivityLabel;
  NSTextField *_imageWidthControl;
  NSTextField *_imageHeightControl;
  NSButton *_aspectLockControl;
  NSButton *_matchWindowControl;
  NSButton *_invertMousePanningControl;
  NSButton *_normalLightingControl;
  NSTextField *_debugInfoLabel;
  NSTextField *_pointInfoHeading;
  NSTextField *_pointInfoLabel;
  uint64_t _displayedRevision;
  uint64_t _displayedInspectionSequence;
  uint64_t _pointLockRequestToken;
  double _lockedAspectRatio;
  double _panningSensitivity;
  bool _pointInspectionEnabled;
  bool _pointInspectionLocked;
  bool _pointLockPending;
  bool _invertMousePanning;
  bool _updatingResolutionControls;
}
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)rotateForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)panForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)zoomWithScrollDelta:(double)delta precise:(bool)precise;
- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView
             aspectFitView:(AspectFitContainerView *)aspectFitView;
- (void)inspectPixelX:(uint32_t)x y:(uint32_t)y;
- (void)togglePointLockAtPixelX:(uint32_t)x y:(uint32_t)y;
- (void)clearPointInspection;
- (void)togglePointInspection:(id)sender;
- (NSViewController *)makeSettingsViewController;
- (NSViewController *)makeDebugViewController;
- (NSViewController *)makePointInfoViewController;
@end

@implementation PanoramaView

/// Position the marker using the current displayed view bounds. Keeping the
/// projection in image coordinates lets ordinary AppKit layout handle window
/// resizing without retracing or resampling the locked point.
- (void)layoutLockedPointIndicator {
  if (!_lockedPointIndicatorActive || _lockedPointMarker == nil) {
    _lockedPointMarker.hidden = YES;
    return;
  }
  const NSRect bounds = self.bounds;
  const double image_width = self.drawableSize.width;
  const double image_height = self.drawableSize.height;
  constexpr CGFloat kMarkerSize = 26.0;
  constexpr CGFloat kEdgeInset = 18.0;
  if (bounds.size.width <= kMarkerSize || bounds.size.height <= kMarkerSize || image_width <= 0.0 ||
      image_height <= 0.0) {
    _lockedPointMarker.hidden = YES;
    return;
  }

  NSPoint centre = {};
  if (_lockedPointOnscreen) {
    centre.x =
        NSMinX(bounds) + static_cast<CGFloat>(_lockedPointPixelX / image_width) * bounds.size.width;
    centre.y = NSMaxY(bounds) -
               static_cast<CGFloat>(_lockedPointPixelY / image_height) * bounds.size.height;
    _lockedPointMarker.image = [NSImage imageWithSystemSymbolName:@"scope"
                                         accessibilityDescription:@"Locked terrain point"];
    [_lockedPointMarker.layer setAffineTransform:CGAffineTransformIdentity];
    _lockedPointMarker.toolTip = @"Locked terrain point";
  } else {
    // Convert the image's downward-positive direction to AppKit's upward-
    // positive coordinates, then intersect it with an inset view rectangle.
    double direction_x = _lockedPointDirectionX;
    double direction_y = -_lockedPointDirectionY;
    const double length = std::hypot(direction_x, direction_y);
    if (!std::isfinite(length) || length < 1e-12) {
      direction_x = 1.0;
      direction_y = 0.0;
    }
    const CGFloat half_width = std::max(0.0, bounds.size.width * 0.5 - kEdgeInset);
    const CGFloat half_height = std::max(0.0, bounds.size.height * 0.5 - kEdgeInset);
    const double horizontal_scale = std::abs(direction_x) > 1e-12
                                        ? half_width / std::abs(direction_x)
                                        : std::numeric_limits<double>::infinity();
    const double vertical_scale = std::abs(direction_y) > 1e-12
                                      ? half_height / std::abs(direction_y)
                                      : std::numeric_limits<double>::infinity();
    const double scale = std::min(horizontal_scale, vertical_scale);
    centre = NSMakePoint(
        NSMidX(bounds) + static_cast<CGFloat>(direction_x * scale),
        NSMidY(bounds) + static_cast<CGFloat>(direction_y * scale)
    );
    _lockedPointMarker.image =
        [NSImage imageWithSystemSymbolName:@"arrow.up.circle.fill"
                  accessibilityDescription:@"Locked terrain point is outside the view"];
    const CGFloat rotation = static_cast<CGFloat>(std::atan2(-direction_x, direction_y));
    [_lockedPointMarker.layer setAffineTransform:CGAffineTransformMakeRotation(rotation)];
    _lockedPointMarker.toolTip = @"Locked terrain point is outside the view";
  }

  _lockedPointMarker.frame = NSMakeRect(
      centre.x - kMarkerSize * 0.5,
      centre.y - kMarkerSize * 0.5,
      kMarkerSize,
      kMarkerSize
  );
  _lockedPointMarker.hidden = NO;
}

- (void)setLockedPointIndicator:(std::optional<panorama::app::LockedPointProjection>)projection {
  _lockedPointIndicatorActive = projection.has_value();
  if (!_lockedPointIndicatorActive) {
    _lockedPointMarker.hidden = YES;
    return;
  }
  if (_lockedPointMarker == nil) {
    _lockedPointMarker = [[LockedPointMarkerView alloc] initWithFrame:NSZeroRect];
    _lockedPointMarker.imageScaling = NSImageScaleProportionallyUpOrDown;
    _lockedPointMarker.contentTintColor = NSColor.systemOrangeColor;
    _lockedPointMarker.wantsLayer = YES;
    _lockedPointMarker.layer.shadowColor = NSColor.blackColor.CGColor;
    _lockedPointMarker.layer.shadowOpacity = 0.9F;
    _lockedPointMarker.layer.shadowRadius = 2.0;
    _lockedPointMarker.layer.shadowOffset = CGSizeZero;
    [self addSubview:_lockedPointMarker];
  }
  _lockedPointOnscreen = projection->onscreen;
  _lockedPointPixelX = projection->pixel_x;
  _lockedPointPixelY = projection->pixel_y;
  _lockedPointDirectionX = projection->direction_x;
  _lockedPointDirectionY = projection->direction_y;
  [self layoutLockedPointIndicator];
}

- (void)layout {
  [super layout];
  [self layoutLockedPointIndicator];
}

- (void)updateTrackingAreas {
  [super updateTrackingAreas];
  if (_inspectionTrackingArea != nil) {
    [self removeTrackingArea:_inspectionTrackingArea];
    _inspectionTrackingArea = nil;
  }
  if (!_pointInspectionEnabled) {
    return;
  }
  _inspectionTrackingArea =
      [[NSTrackingArea alloc] initWithRect:NSZeroRect
                                   options:NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited |
                                           NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect
                                     owner:self
                                  userInfo:nil];
  [self addTrackingArea:_inspectionTrackingArea];
}

- (void)resetCursorRects {
  [super resetCursorRects];
  if (_pointInspectionEnabled) {
    [self addCursorRect:self.bounds cursor:NSCursor.crosshairCursor];
  }
}

- (void)setPointInspectionEnabled:(bool)enabled {
  _pointInspectionEnabled = enabled;
  self.window.acceptsMouseMovedEvents = enabled;
  [self updateTrackingAreas];
  [self.window invalidateCursorRectsForView:self];
}

/// Convert an AppKit event location into the current top-left-origin ray image.
- (BOOL)inspectionPixelForEvent:(NSEvent *)event x:(uint32_t *)x y:(uint32_t *)y {
  const NSRect bounds = self.bounds;
  const NSPoint location = [self convertPoint:event.locationInWindow fromView:nil];
  const uint32_t width = static_cast<uint32_t>(self.drawableSize.width);
  const uint32_t height = static_cast<uint32_t>(self.drawableSize.height);
  if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0 || width == 0U || height == 0U) {
    return NO;
  }

  // AppKit view coordinates rise from the bottom-left, whereas RayField rows
  // use image coordinates from the top-left.
  const double normalised_x =
      std::clamp((location.x - NSMinX(bounds)) / bounds.size.width, 0.0, 1.0);
  const double normalised_y =
      std::clamp((NSMaxY(bounds) - location.y) / bounds.size.height, 0.0, 1.0);
  *x = std::min(width - 1U, static_cast<uint32_t>(normalised_x * static_cast<double>(width)));
  *y = std::min(height - 1U, static_cast<uint32_t>(normalised_y * static_cast<double>(height)));
  return YES;
}

- (void)mouseMoved:(NSEvent *)event {
  if (!_pointInspectionEnabled) {
    [super mouseMoved:event];
    return;
  }
  uint32_t x = 0U;
  uint32_t y = 0U;
  if (![self inspectionPixelForEvent:event x:&x y:&y]) {
    return;
  }
  [self.panoramaController inspectPixelX:x y:y];
}

/// Right-click locks the current terrain sample without consuming the
/// left-button drag gesture used to rotate the camera.
- (void)rightMouseDown:(NSEvent *)event {
  if (!_pointInspectionEnabled) {
    [super rightMouseDown:event];
    return;
  }
  uint32_t x = 0U;
  uint32_t y = 0U;
  if ([self inspectionPixelForEvent:event x:&x y:&y]) {
    [self.panoramaController togglePointLockAtPixelX:x y:y];
  }
}

- (void)mouseExited:(NSEvent *)event {
  (void)event;
  if (_pointInspectionEnabled) {
    [self.panoramaController clearPointInspection];
  }
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (void)mouseDown:(NSEvent *)event {
  _lastMouseLocation = [self convertPoint:event.locationInWindow fromView:nil];
}

- (void)mouseDragged:(NSEvent *)event {
  const NSPoint location = [self convertPoint:event.locationInWindow fromView:nil];
  const double heading = (location.x - _lastMouseLocation.x) * 0.003;
  const double pitch = (location.y - _lastMouseLocation.y) * 0.003;
  _lastMouseLocation = location;
  [self.panoramaController panForCurrentZoomHeading:heading pitch:pitch];
}

- (void)scrollWheel:(NSEvent *)event {
  [self.panoramaController zoomWithScrollDelta:event.scrollingDeltaY
                                       precise:event.hasPreciseScrollingDeltas];
}

- (void)keyDown:(NSEvent *)event {
  constexpr double kStep = 2.0 * std::numbers::pi / 180.0;
  switch (event.keyCode) {
  case 123: // Left arrow.
    [self.panoramaController rotateForCurrentZoomHeading:-kStep pitch:0.0];
    break;
  case 124: // Right arrow.
    [self.panoramaController rotateForCurrentZoomHeading:kStep pitch:0.0];
    break;
  case 125: // Down arrow.
    [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:-kStep];
    break;
  case 126: // Up arrow.
    [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:kStep];
    break;
  default: {
    const NSString *characters = event.charactersIgnoringModifiers.lowercaseString;
    if ([characters isEqualToString:@"a"]) {
      [self.panoramaController rotateForCurrentZoomHeading:-kStep pitch:0.0];
    } else if ([characters isEqualToString:@"d"]) {
      [self.panoramaController rotateForCurrentZoomHeading:kStep pitch:0.0];
    } else if ([characters isEqualToString:@"s"]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:-kStep];
    } else if ([characters isEqualToString:@"w"]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:kStep];
    } else {
      [super keyDown:event];
    }
    break;
  }
  }
}

@end

/// Letterbox the render view at its current output aspect ratio without
/// imposing a fitting size on its parent. The parent can therefore resize
/// freely, and a resolution change can update the ratio without distorting the
/// Metal output.
@interface AspectFitContainerView : NSView {
@private
  NSView *_renderView;
  CGFloat _aspectRatio;
}
- (instancetype)initWithFrame:(NSRect)frame
                   renderView:(NSView *)renderView
                  aspectRatio:(CGFloat)aspectRatio;
- (void)setAspectRatio:(CGFloat)aspectRatio;
@end

@implementation AspectFitContainerView

- (instancetype)initWithFrame:(NSRect)frame
                   renderView:(NSView *)renderView
                  aspectRatio:(CGFloat)aspectRatio {
  self = [super initWithFrame:frame];
  if (self != nil) {
    _renderView = renderView;
    _aspectRatio = aspectRatio;
    self.wantsLayer = YES;
    self.layer.backgroundColor = NSColor.blackColor.CGColor;
    [self addSubview:_renderView];
  }
  return self;
}

- (void)setAspectRatio:(CGFloat)aspectRatio {
  if (aspectRatio <= 0.0 || std::abs(aspectRatio - _aspectRatio) <= 1e-9) {
    return;
  }
  _aspectRatio = aspectRatio;
  [self setNeedsLayout:YES];
}

- (void)layout {
  [super layout];
  const NSRect bounds = self.bounds;
  CGFloat width = bounds.size.width;
  CGFloat height = width / _aspectRatio;
  if (height > bounds.size.height) {
    height = bounds.size.height;
    width = height * _aspectRatio;
  }
  _renderView.frame = NSMakeRect(
      bounds.origin.x + (bounds.size.width - width) * 0.5,
      bounds.origin.y + (bounds.size.height - height) * 0.5,
      width,
      height
  );
}

@end

/// Wrap custom overlay content in the current platform's native translucent
/// material. The returned view owns `contentView` through either the modern
/// Liquid Glass API or the pre-macOS 26 visual-effect fallback.
static NSView *makeOverlayPanel(NSView *contentView) {
  NSView *panel = nil;
  if (@available(macOS 26.0, *)) {
    NSGlassEffectView *glass = [[NSGlassEffectView alloc] initWithFrame:NSZeroRect];
    glass.style = NSGlassEffectViewStyleRegular;
    glass.cornerRadius = 18.0;
    glass.contentView = contentView;
    panel = glass;
  } else {
    NSVisualEffectView *material = [[NSVisualEffectView alloc] initWithFrame:NSZeroRect];
    material.material = NSVisualEffectMaterialSidebar;
    material.blendingMode = NSVisualEffectBlendingModeWithinWindow;
    material.state = NSVisualEffectStateActive;
    material.wantsLayer = YES;
    material.layer.cornerRadius = 18.0;
    material.layer.masksToBounds = YES;
    [material addSubview:contentView];
    panel = material;
  }
  contentView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  return panel;
}

/// Keep the render at the full window size and position auxiliary panels above
/// it. Frame-based layout is deliberate: overlapping children do not define a
/// useful Auto Layout fitting size for an NSWindow content view.
@interface ViewerOverlayView : NSView {
@private
  NSView *_contentView;
  NSView *_inspectorView;
  NSView *_settingsView;
  NSView *_debugView;
  NSView *_debugContentView;
  NSView *_pointInfoView;
  NSView *_pointInfoContentView;
  CGFloat _inspectorWidth;
  NSSize _debugSize;
  NSSize _pointInfoSize;
  CGFloat _panelMargin;
  bool _inspectorVisible;
  bool _debugVisible;
  bool _pointInfoVisible;
}
- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth
                    debugView:(NSView *)debugView
                    debugSize:(NSSize)debugSize
                pointInfoView:(NSView *)pointInfoView
                pointInfoSize:(NSSize)pointInfoSize;
- (void)toggleInspector:(id)sender;
- (void)toggleDebugOverlay:(id)sender;
- (void)setPointInfoVisible:(bool)visible;
@end

@implementation ViewerOverlayView

- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth
                    debugView:(NSView *)debugView
                    debugSize:(NSSize)debugSize
                pointInfoView:(NSView *)pointInfoView
                pointInfoSize:(NSSize)pointInfoSize {
  self = [super initWithFrame:frame];
  if (self != nil) {
    _contentView = contentView;
    _settingsView = settingsView;
    _debugContentView = debugView;
    _pointInfoContentView = pointInfoView;
    _inspectorWidth = inspectorWidth;
    _debugSize = debugSize;
    _pointInfoSize = pointInfoSize;
    _panelMargin = 12.0;
    _inspectorVisible = true;
    _debugVisible = false;
    _pointInfoVisible = false;
    self.wantsLayer = YES;
    self.layer.masksToBounds = YES;
    [self addSubview:_contentView];

    _inspectorView = makeOverlayPanel(_settingsView);
    _debugView = makeOverlayPanel(_debugContentView);
    _pointInfoView = makeOverlayPanel(_pointInfoContentView);
    [self addSubview:_inspectorView];
    [self addSubview:_debugView];
    [self addSubview:_pointInfoView];
  }
  return self;
}

- (NSRect)inspectorFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  // Full-size window content extends beneath the titlebar so the native
  // toolbar can float over it. Keep the inspector inside AppKit's safe area,
  // clear of the toolbar and window controls, while allowing the terrain view
  // itself to fill the window.
  const NSEdgeInsets safeArea = self.safeAreaInsets;
  const CGFloat x = visible ? NSMaxX(bounds) - safeArea.right - _inspectorWidth - _panelMargin
                            : NSMaxX(bounds) + _panelMargin;
  const CGFloat bottom = safeArea.bottom + _panelMargin;
  const CGFloat top = safeArea.top + _panelMargin;
  const CGFloat height = std::max(0.0, bounds.size.height - bottom - top);
  return NSMakeRect(x, bounds.origin.y + bottom, _inspectorWidth, height);
}

- (NSRect)debugFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  const NSEdgeInsets safeArea = self.safeAreaInsets;
  const CGFloat availableHeight =
      std::max(0.0, bounds.size.height - safeArea.top - safeArea.bottom - 2.0 * _panelMargin);
  const CGFloat height = std::min(_debugSize.height, availableHeight);
  const CGFloat x = visible ? NSMinX(bounds) + safeArea.left + _panelMargin
                            : NSMinX(bounds) - _debugSize.width - _panelMargin;
  const CGFloat y = NSMaxY(bounds) - safeArea.top - _panelMargin - height;
  return NSMakeRect(x, y, _debugSize.width, height);
}

- (NSRect)pointInfoFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  const NSEdgeInsets safeArea = self.safeAreaInsets;
  const CGFloat x = visible ? NSMinX(bounds) + safeArea.left + _panelMargin
                            : NSMinX(bounds) - _pointInfoSize.width - _panelMargin;
  const CGFloat y = NSMinY(bounds) + safeArea.bottom + _panelMargin;
  return NSMakeRect(x, y, _pointInfoSize.width, _pointInfoSize.height);
}

- (void)layout {
  [super layout];
  _contentView.frame = self.bounds;
  _inspectorView.frame = [self inspectorFrameForVisible:_inspectorVisible];
  _settingsView.frame = _inspectorView.bounds;
  _debugView.frame = [self debugFrameForVisible:_debugVisible];
  _debugContentView.frame = _debugView.bounds;
  _pointInfoView.frame = [self pointInfoFrameForVisible:_pointInfoVisible];
  _pointInfoContentView.frame = _pointInfoView.bounds;
}

- (void)toggleInspector:(id)sender {
  (void)sender;
  _inspectorVisible = !_inspectorVisible;
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _inspectorView.animator.frame = [self inspectorFrameForVisible:_inspectorVisible];
  }];
}

- (void)toggleDebugOverlay:(id)sender {
  (void)sender;
  _debugVisible = !_debugVisible;
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _debugView.animator.frame = [self debugFrameForVisible:_debugVisible];
  }];
}

- (void)setPointInfoVisible:(bool)visible {
  if (_pointInfoVisible == visible) {
    return;
  }
  _pointInfoVisible = visible;
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _pointInfoView.animator.frame = [self pointInfoFrameForVisible:_pointInfoVisible];
  }];
}

@end

@implementation PanoramaController

- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window {
  self = [super init];
  if (self != nil) {
    _renderer = renderer;
    _window = window;
    _orientation = renderer->initial_orientation();
    _verticalFieldOfView = renderer->initial_vertical_field_of_view();
    _image = renderer->image();
    _lockedAspectRatio = static_cast<double>(_image.width) / _image.height;
    _panningSensitivity = 8.0;
    _presentation = renderer->initial_presentation();
  }
  return self;
}

- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta {
  _orientation.heading =
      std::remainder(_orientation.heading + headingDelta, 2.0 * std::numbers::pi);
  constexpr double kPitchLimit = 85.0 * std::numbers::pi / 180.0;
  _orientation.pitch = std::clamp(_orientation.pitch + pitchDelta, -kPitchLimit, kPitchLimit);
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
}

- (void)rotateForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta {
  // Mouse and keyboard deltas define their desired feel at the default FOV.
  // Scaling by the current angular extent preserves that behaviour while
  // providing proportionally finer control over a magnified view.
  const double zoom_scale = _verticalFieldOfView / panorama::app::kDefaultVerticalFieldOfView;
  constexpr double kExistingSensitivity = 8.0;
  const double sensitivity_scale = _panningSensitivity / kExistingSensitivity;
  const double scale = zoom_scale * sensitivity_scale;
  [self rotateHeading:headingDelta * scale pitch:pitchDelta * scale];
}

- (void)panForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta {
  const double direction = _invertMousePanning ? -1.0 : 1.0;
  [self rotateForCurrentZoomHeading:headingDelta * direction pitch:pitchDelta * direction];
}

- (void)updateZoomControls {
  const double degrees = _verticalFieldOfView * panorama::app::kRadiansToDegrees;
  if (_zoomControl != nil) {
    _zoomControl.doubleValue = degrees;
  }
  if (_zoomValueLabel != nil) {
    _zoomValueLabel.stringValue = [NSString stringWithFormat:@"%.1f°", degrees];
  }
}

- (void)setVerticalFieldOfViewDegrees:(double)degrees {
  constexpr double kMinimumDegrees = 5.0;
  constexpr double kMaximumDegrees = 140.0;
  const double next =
      std::clamp(degrees, kMinimumDegrees, kMaximumDegrees) * panorama::app::kDegreesToRadians;
  if (std::abs(next - _verticalFieldOfView) <= 1e-12) {
    [self updateZoomControls];
    return;
  }
  _verticalFieldOfView = next;
  [self updateZoomControls];
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
}

- (void)zoomControlChanged:(NSSlider *)sender {
  double degrees = sender.doubleValue;
  // A small detent makes the original 70-degree view easy to recover while
  // leaving the remainder of the slider continuously adjustable.
  constexpr double kDefaultDetentDegrees = 70.0;
  constexpr double kDetentRadiusDegrees = 2.0;
  if (std::abs(degrees - kDefaultDetentDegrees) <= kDetentRadiusDegrees) {
    degrees = kDefaultDetentDegrees;
  }
  [self setVerticalFieldOfViewDegrees:degrees];
}

- (void)invertMousePanningChanged:(NSButton *)sender {
  _invertMousePanning = sender.state == NSControlStateValueOn;
}

- (void)panningSensitivityChanged:(NSSlider *)sender {
  _panningSensitivity = std::round(sender.doubleValue);
  sender.doubleValue = _panningSensitivity;
  _panningSensitivityLabel.stringValue = [NSString stringWithFormat:@"%.0f", _panningSensitivity];
}

- (void)updateAspectLockAppearance {
  const BOOL locked = _aspectLockControl.state == NSControlStateValueOn;
  _aspectLockControl.image = [NSImage
      imageWithSystemSymbolName:locked ? @"lock.fill" : @"lock.open"
       accessibilityDescription:locked ? @"Aspect ratio locked" : @"Aspect ratio unlocked"];
}

- (void)aspectLockChanged:(NSButton *)sender {
  if (sender.state == NSControlStateValueOn) {
    const std::optional<uint32_t> width =
        panorama::app::parse_image_dimension(_imageWidthControl.stringValue);
    const std::optional<uint32_t> height =
        panorama::app::parse_image_dimension(_imageHeightControl.stringValue);
    if (width.has_value() && height.has_value()) {
      _lockedAspectRatio = static_cast<double>(*width) / *height;
    }
  }
  [self updateAspectLockAppearance];
}

/// Maintain the captured aspect ratio while either dimension is edited. The
/// paired field changes immediately, but GPU resources are resized only when
/// the user applies the completed settings.
- (void)controlTextDidChange:(NSNotification *)notification {
  if (_updatingResolutionControls || _aspectLockControl.state != NSControlStateValueOn) {
    return;
  }
  NSTextField *changed = notification.object;
  if (changed != _imageWidthControl && changed != _imageHeightControl) {
    return;
  }
  const std::optional<uint32_t> value = panorama::app::parse_image_dimension(changed.stringValue);
  if (!value.has_value() || !std::isfinite(_lockedAspectRatio) || _lockedAspectRatio <= 0.0) {
    return;
  }

  const double paired_value = changed == _imageWidthControl
                                  ? static_cast<double>(*value) / _lockedAspectRatio
                                  : static_cast<double>(*value) * _lockedAspectRatio;
  if (paired_value < 1.0 || paired_value > std::numeric_limits<uint32_t>::max()) {
    return;
  }
  _updatingResolutionControls = true;
  NSTextField *paired = changed == _imageWidthControl ? _imageHeightControl : _imageWidthControl;
  paired.stringValue =
      [NSString stringWithFormat:@"%u", static_cast<uint32_t>(std::llround(paired_value))];
  _updatingResolutionControls = false;
}

/// Match the render aspect to the available window content by changing only
/// its horizontal pixel count. Vertical resolution and vertical FOV remain
/// untouched.
- (void)matchWindowResolution:(id)sender {
  (void)sender;
  const NSSize available = _aspectFitView.bounds.size;
  const std::optional<uint32_t> height =
      panorama::app::parse_image_dimension(_imageHeightControl.stringValue);
  if (!height.has_value() || available.width <= 0.0 || available.height <= 0.0) {
    NSBeep();
    return;
  }
  const double width = static_cast<double>(*height) * available.width / available.height;
  if (!std::isfinite(width) || width < 1.0 || width > std::numeric_limits<uint32_t>::max()) {
    NSBeep();
    return;
  }
  const uint32_t rounded_width = static_cast<uint32_t>(std::llround(width));
  _updatingResolutionControls = true;
  _imageWidthControl.stringValue = [NSString stringWithFormat:@"%u", rounded_width];
  _updatingResolutionControls = false;
  if (_aspectLockControl.state == NSControlStateValueOn) {
    _lockedAspectRatio = static_cast<double>(rounded_width) / *height;
  }
  [self applyViewerSettings:nil];
}

- (void)zoomWithScrollDelta:(double)delta precise:(bool)precise {
  // Exponential scaling makes equal scroll motion feel proportional at wide
  // and narrow fields of view. Trackpads report much finer-grained deltas
  // than traditional mouse wheels and therefore use a gentler coefficient.
  const double sensitivity = precise ? 0.012 : 0.08;
  constexpr double kMinimumFieldOfView = 5.0 * std::numbers::pi / 180.0;
  constexpr double kMaximumFieldOfView = 140.0 * std::numbers::pi / 180.0;
  const double next = std::clamp(
      _verticalFieldOfView * std::exp(-delta * sensitivity),
      kMinimumFieldOfView,
      kMaximumFieldOfView
  );
  if (std::abs(next - _verticalFieldOfView) <= 1e-12) {
    return;
  }
  [self setVerticalFieldOfViewDegrees:next * panorama::app::kRadiansToDegrees];
}

- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView
             aspectFitView:(AspectFitContainerView *)aspectFitView {
  _panoramaView = panoramaView;
  _overlayView = overlayView;
  _aspectFitView = aspectFitView;
}

- (void)inspectPixelX:(uint32_t)x y:(uint32_t)y {
  if (_pointInspectionEnabled && !_pointInspectionLocked && !_pointLockPending) {
    _renderer->request_inspection(panorama::app::InspectionPixel{x, y});
  }
}

- (void)togglePointLockAtPixelX:(uint32_t)x y:(uint32_t)y {
  if (!_pointInspectionEnabled) {
    return;
  }
  if (_pointInspectionLocked || _pointLockPending) {
    _pointInspectionLocked = false;
    _pointLockPending = false;
    _lockedPoint.reset();
    [_panoramaView setLockedPointIndicator:std::nullopt];
    _pointInfoHeading.stringValue = @"Point Info";
    _renderer->request_inspection(panorama::app::InspectionPixel{x, y});
    return;
  }

  _pointLockPending = true;
  _pointInfoHeading.stringValue = @"Point Info — Locking…";
  _pointLockRequestToken = _renderer->request_inspection(panorama::app::InspectionPixel{x, y});
}

- (void)clearPointInspection {
  if (!_pointInspectionLocked && !_pointLockPending) {
    _renderer->request_inspection(std::nullopt);
  }
}

- (void)togglePointInspection:(id)sender {
  _pointInspectionEnabled = !_pointInspectionEnabled;
  _pointInspectionLocked = false;
  _pointLockPending = false;
  _lockedPoint.reset();
  [_panoramaView setLockedPointIndicator:std::nullopt];
  [_panoramaView setPointInspectionEnabled:_pointInspectionEnabled];
  [_overlayView setPointInfoVisible:_pointInspectionEnabled];
  _pointInfoHeading.stringValue = @"Point Info";
  if (!_pointInspectionEnabled) {
    [self clearPointInspection];
  } else {
    [self updatePointInfo:std::nullopt];
  }

  if ([sender isKindOfClass:NSToolbarItem.class]) {
    NSToolbarItem *item = sender;
    if (@available(macOS 26.0, *)) {
      item.style = _pointInspectionEnabled ? NSToolbarItemStyleProminent : NSToolbarItemStylePlain;
    }
  }
}

/// Build the controls shown in the trailing render-settings inspector.
- (NSViewController *)makeSettingsViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 300.0, 400.0)];
  viewController.view = content;

  NSTextField *heading = [NSTextField labelWithString:@"Viewer Settings"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _zoomControl = [NSSlider sliderWithValue:_verticalFieldOfView * panorama::app::kRadiansToDegrees
                                  minValue:0.0
                                  maxValue:140.0
                                    target:self
                                    action:@selector(zoomControlChanged:)];
  _zoomControl.continuous = YES;
  _zoomControl.numberOfTickMarks = 3;
  _zoomControl.allowsTickMarkValuesOnly = NO;
  _zoomValueLabel = [NSTextField labelWithString:@""];
  _zoomValueLabel.alignment = NSTextAlignmentRight;
  [_zoomValueLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *zoomSetting = [NSStackView stackViewWithViews:@[ _zoomControl, _zoomValueLabel ]];
  zoomSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  zoomSetting.alignment = NSLayoutAttributeCenterY;
  zoomSetting.spacing = 6.0;
  [self updateZoomControls];

  _panningSensitivityControl = [NSSlider sliderWithValue:_panningSensitivity
                                                minValue:1.0
                                                maxValue:10.0
                                                  target:self
                                                  action:@selector(panningSensitivityChanged:)];
  _panningSensitivityControl.continuous = YES;
  _panningSensitivityControl.numberOfTickMarks = 10;
  _panningSensitivityControl.allowsTickMarkValuesOnly = YES;
  _panningSensitivityControl.toolTip = @"Mouse and keyboard panning sensitivity";
  _panningSensitivityLabel = [NSTextField labelWithString:@"8"];
  _panningSensitivityLabel.alignment = NSTextAlignmentRight;
  [_panningSensitivityLabel.widthAnchor constraintEqualToConstant:18.0].active = YES;
  NSStackView *panningSensitivitySetting =
      [NSStackView stackViewWithViews:@[ _panningSensitivityControl, _panningSensitivityLabel ]];
  panningSensitivitySetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  panningSensitivitySetting.alignment = NSLayoutAttributeCenterY;
  panningSensitivitySetting.spacing = 6.0;

  _imageWidthControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _imageWidthControl.stringValue = [NSString stringWithFormat:@"%u", _image.width];
  _imageWidthControl.delegate = self;
  _imageHeightControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _imageHeightControl.stringValue = [NSString stringWithFormat:@"%u", _image.height];
  _imageHeightControl.delegate = self;
  [_imageWidthControl.widthAnchor constraintEqualToConstant:42.0].active = YES;
  [_imageHeightControl.widthAnchor constraintEqualToConstant:42.0].active = YES;
  NSTextField *resolutionSeparator = [NSTextField labelWithString:@"×"];

  _aspectLockControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _aspectLockControl.buttonType = NSButtonTypeToggle;
  _aspectLockControl.state = NSControlStateValueOn;
  _aspectLockControl.title = @"";
  _aspectLockControl.bordered = NO;
  _aspectLockControl.imagePosition = NSImageOnly;
  _aspectLockControl.target = self;
  _aspectLockControl.action = @selector(aspectLockChanged:);
  _aspectLockControl.toolTip = @"Keep width and height at the current aspect ratio";
  [_aspectLockControl setAccessibilityLabel:@"Lock aspect ratio"];
  [_aspectLockControl.widthAnchor constraintEqualToConstant:20.0].active = YES;
  [self updateAspectLockAppearance];

  NSStackView *resolutionSetting = [NSStackView stackViewWithViews:@[
    _imageWidthControl,
    resolutionSeparator,
    _imageHeightControl,
    _aspectLockControl,
  ]];
  resolutionSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  resolutionSetting.alignment = NSLayoutAttributeCenterY;
  resolutionSetting.spacing = 4.0;

  _matchWindowControl = [NSButton buttonWithTitle:@"Match Window"
                                           target:self
                                           action:@selector(matchWindowResolution:)];
  _matchWindowControl.image =
      [NSImage imageWithSystemSymbolName:@"arrow.left.and.right"
                accessibilityDescription:@"Match horizontal resolution to window"];
  _matchWindowControl.imagePosition = NSImageLeading;
  _matchWindowControl.toolTip =
      @"Change horizontal resolution to match the window; keep vertical resolution fixed";

  _invertMousePanningControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _invertMousePanningControl.buttonType = NSButtonTypeSwitch;
  _invertMousePanningControl.title = @"Invert click-and-drag panning";
  _invertMousePanningControl.state = NSControlStateValueOff;
  _invertMousePanningControl.target = self;
  _invertMousePanningControl.action = @selector(invertMousePanningChanged:);

  _colourSourceControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourSourceControl addItemsWithTitles:@[ @"None (white)", @"Distance", @"Elevation" ]];
  [_colourSourceControl
      selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colour_source)];

  _colourmapControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourmapControl
      addItemsWithTitles:@[ @"Viridis", @"Plasma", @"Inferno", @"Magma", @"Cividis", @"Turbo" ]];
  [_colourmapControl selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colourmap)];

  _minimumControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _minimumControl.stringValue =
      panorama::app::format_range_value(_presentation.colour_range.minimum);

  _maximumControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _maximumControl.stringValue =
      panorama::app::format_range_value(_presentation.colour_range.maximum);

  _normalLightingControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _normalLightingControl.buttonType = NSButtonTypeSwitch;
  _normalLightingControl.title = @"Shade using surface normals";
  _normalLightingControl.state =
      _presentation.use_surface_normals ? NSControlStateValueOn : NSControlStateValueOff;

  NSButton *apply = [[NSButton alloc] initWithFrame:NSZeroRect];
  apply.title = @"Apply";
  apply.bezelStyle = NSBezelStyleRounded;
  apply.keyEquivalent = @"\r";
  apply.target = self;
  apply.action = @selector(applyViewerSettings:);
  _colourSourceControl.target = self;
  _colourSourceControl.action = @selector(renderModeChanged:);

  auto make_row = [](NSString *title, NSView *control) {
    NSTextField *label = [NSTextField labelWithString:title];
    [label.widthAnchor constraintEqualToConstant:105.0].active = YES;
    // The 270-point panel has 238 points inside its horizontal margins.
    // Keep each row within that width instead of allowing controls to crowd
    // the trailing glass edge.
    [control.widthAnchor constraintGreaterThanOrEqualToConstant:125.0].active = YES;
    NSStackView *row = [NSStackView stackViewWithViews:@[ label, control ]];
    row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row.alignment = NSLayoutAttributeCenterY;
    row.spacing = 8.0;
    return row;
  };

  NSStackView *settings = [NSStackView stackViewWithViews:@[
    heading,
    make_row(@"Zoom (V. FOV)", zoomSetting),
    make_row(@"Pan sensitivity", panningSensitivitySetting),
    make_row(@"Resolution", resolutionSetting),
    _matchWindowControl,
    _invertMousePanningControl,
    make_row(@"Colour by", _colourSourceControl),
    make_row(@"Colourmap", _colourmapControl),
    make_row(@"Range minimum", _minimumControl),
    make_row(@"Range maximum", _maximumControl),
    _normalLightingControl,
    apply,
  ]];
  settings.orientation = NSUserInterfaceLayoutOrientationVertical;
  settings.alignment = NSLayoutAttributeLeading;
  settings.spacing = 12.0;
  settings.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:settings];
  [NSLayoutConstraint activateConstraints:@[
    [settings.topAnchor constraintEqualToAnchor:content.topAnchor constant:20.0],
    [settings.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:16.0],
    [settings.trailingAnchor constraintLessThanOrEqualToAnchor:content.trailingAnchor
                                                      constant:-16.0],
  ]];

  [self updateSettingsControlAvailability];
  return viewController;
}

/// Build the read-only diagnostics displayed over the leading side of the
/// rendered scene. Values are refreshed only when a completed revision becomes
/// visible, avoiding work on unchanged MetalKit redraws.
- (NSViewController *)makeDebugViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 220.0, 282.0)];
  viewController.view = content;

  NSTextField *heading = [NSTextField labelWithString:@"Viewer Debug Info"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _debugInfoLabel = [NSTextField labelWithString:@""];
  _debugInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _debugInfoLabel.maximumNumberOfLines = 0;
  _debugInfoLabel.lineBreakMode = NSLineBreakByClipping;

  NSStackView *debugInfo = [NSStackView stackViewWithViews:@[ heading, _debugInfoLabel ]];
  debugInfo.orientation = NSUserInterfaceLayoutOrientationVertical;
  debugInfo.alignment = NSLayoutAttributeLeading;
  debugInfo.spacing = 12.0;
  debugInfo.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:debugInfo];
  [NSLayoutConstraint activateConstraints:@[
    [debugInfo.topAnchor constraintEqualToAnchor:content.topAnchor constant:16.0],
    [debugInfo.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:16.0],
    [debugInfo.trailingAnchor constraintLessThanOrEqualToAnchor:content.trailingAnchor
                                                       constant:-16.0],
  ]];

  [self updateDebugInfoWithOrientation:_orientation
                   verticalFieldOfView:_verticalFieldOfView
                                 image:_image
                          milliseconds:0.0
                              revision:0U];
  return viewController;
}

- (void)updateDebugInfoWithOrientation:(panorama::CameraOrientation)orientation
                   verticalFieldOfView:(double)verticalFieldOfView
                                 image:(panorama::ImageSize)image
                          milliseconds:(double)milliseconds
                              revision:(uint64_t)revision {
  if (_debugInfoLabel == nil) {
    return;
  }

  const panorama::ObserverLocation observer = _renderer->observer();
  double heading = std::fmod(orientation.heading * panorama::app::kRadiansToDegrees, 360.0);
  if (heading < 0.0) {
    heading += 360.0;
  }
  const double fps = milliseconds > 0.0 ? 1'000.0 / milliseconds : 0.0;
  NSString *performance =
      milliseconds > 0.0
          ? [NSString
                stringWithFormat:@"FPS          %8.2f\nFrame time   %8.2f ms", fps, milliseconds]
          : @"FPS                 —\nFrame time          —";
  _debugInfoLabel.stringValue = [NSString
      stringWithFormat:@"%@\nRevision     %8llu\n\n"
                        "Easting    %11.2f m\nNorthing   %11.2f m\nElevation  %11.2f m\n\n"
                        "Heading      %8.2f°\nPitch        %8.2f°\nRoll         %8.2f°\n"
                        "V. FOV       %8.2f°\n\nResolution   %4u × %4u\nMax range  %10.0f m",
                       performance,
                       static_cast<unsigned long long>(revision),
                       observer.easting,
                       observer.northing,
                       observer.elevation,
                       heading,
                       orientation.pitch * panorama::app::kRadiansToDegrees,
                       orientation.roll * panorama::app::kRadiansToDegrees,
                       verticalFieldOfView * panorama::app::kRadiansToDegrees,
                       image.width,
                       image.height,
                       _renderer->max_distance()];
}

/// Build the compact hover readout shown while point inspection is enabled.
- (NSViewController *)makePointInfoViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 230.0, 174.0)];
  viewController.view = content;

  _pointInfoHeading = [NSTextField labelWithString:@"Point Info"];
  _pointInfoHeading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _pointInfoLabel = [NSTextField labelWithString:@"Move the pointer over the rendered terrain.\n"
                                                  "Right-click to lock a point."];
  _pointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _pointInfoLabel.maximumNumberOfLines = 0;
  _pointInfoLabel.lineBreakMode = NSLineBreakByWordWrapping;

  NSStackView *pointInfo = [NSStackView stackViewWithViews:@[ _pointInfoHeading, _pointInfoLabel ]];
  pointInfo.orientation = NSUserInterfaceLayoutOrientationVertical;
  pointInfo.alignment = NSLayoutAttributeLeading;
  pointInfo.spacing = 10.0;
  pointInfo.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:pointInfo];
  [NSLayoutConstraint activateConstraints:@[
    [pointInfo.topAnchor constraintEqualToAnchor:content.topAnchor constant:14.0],
    [pointInfo.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:14.0],
    [pointInfo.trailingAnchor constraintLessThanOrEqualToAnchor:content.trailingAnchor
                                                       constant:-14.0],
  ]];
  return viewController;
}

- (void)updatePointInfo:(std::optional<panorama::app::PointInspection>)inspection {
  if (_pointInfoLabel == nil) {
    return;
  }
  if (!inspection.has_value()) {
    _pointInfoLabel.stringValue = @"Move the pointer over the rendered terrain.\n"
                                   "Right-click to lock a point.";
    return;
  }
  const panorama::app::PointInspection &point = *inspection;
  if (!point.hit) {
    _pointInfoLabel.stringValue =
        [NSString stringWithFormat:@"Pixel      %4u, %4u\n\nNo terrain intersection",
                                   point.pixel.x,
                                   point.pixel.y];
    return;
  }
  _pointInfoLabel.stringValue =
      [NSString stringWithFormat:@"Pixel      %4u, %4u\n"
                                  "Distance   %10.1f m\nElevation  %10.1f m\n"
                                  "Easting    %10.1f m\nNorthing   %10.1f m\n"
                                  "Slope      %10.1f°\nAspect     %10.1f°",
                                 point.pixel.x,
                                 point.pixel.y,
                                 point.distance,
                                 point.elevation,
                                 point.easting,
                                 point.northing,
                                 point.slope_degrees,
                                 point.aspect_degrees];
}

/// Keep a locked world point aligned with the latest completed camera view.
/// The off-screen state is represented both by an edge arrow and in text, so
/// the lock remains unambiguous even if another overlay obscures the marker.
- (void)updateLockedPointIndicatorWithOrientation:(panorama::CameraOrientation)orientation
                              verticalFieldOfView:(double)verticalFieldOfView
                                            image:(panorama::ImageSize)image {
  if (!_pointInspectionLocked || !_lockedPoint.has_value()) {
    [_panoramaView setLockedPointIndicator:std::nullopt];
    return;
  }
  const panorama::app::LockedPointProjection projection = panorama::app::project_locked_point(
      *_lockedPoint,
      _renderer->observer(),
      image,
      verticalFieldOfView,
      orientation
  );
  [_panoramaView setLockedPointIndicator:projection];
  _pointInfoHeading.stringValue =
      projection.onscreen ? @"Point Info — Locked" : @"Point Info — Locked (Off-screen)";
}

/// Palette and range controls have no effect on the uncoloured white mode.
- (void)updateSettingsControlAvailability {
  const BOOL scalarColour = _colourSourceControl.indexOfSelectedItem != 0;
  _colourmapControl.enabled = scalarColour;
  _minimumControl.enabled = scalarColour;
  _maximumControl.enabled = scalarColour;
}

- (void)renderModeChanged:(id)sender {
  (void)sender;
  [self updateSettingsControlAvailability];
}

/// Validate and publish one coherent settings snapshot to the render worker.
- (void)applyViewerSettings:(id)sender {
  (void)sender;
  const std::optional<uint32_t> width =
      panorama::app::parse_image_dimension(_imageWidthControl.stringValue);
  const std::optional<uint32_t> height =
      panorama::app::parse_image_dimension(_imageHeightControl.stringValue);
  const uint64_t pixel_count =
      width.has_value() && height.has_value() ? static_cast<uint64_t>(*width) * *height : 0U;
  if (!width.has_value() || !height.has_value() ||
      pixel_count > std::numeric_limits<uint32_t>::max()) {
    NSBeep();
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Invalid render resolution";
    alert.informativeText =
        @"Width and height must be positive whole numbers, and their product must fit in the "
         "Metal ray-index range.";
    [alert beginSheetModalForWindow:_window completionHandler:nil];
    return;
  }

  const std::optional<double> minimum =
      panorama::app::parse_range_value(_minimumControl.stringValue);
  const std::optional<double> maximum =
      panorama::app::parse_range_value(_maximumControl.stringValue);
  if (!minimum.has_value() || !maximum.has_value() || *maximum <= *minimum ||
      *minimum < -std::numeric_limits<float>::max() ||
      *maximum > std::numeric_limits<float>::max()) {
    NSBeep();
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Invalid colour range";
    alert.informativeText =
        @"Use commas only as optional thousands separators and a period as the decimal "
         "separator. The maximum must be a finite value greater than the minimum.";
    [alert beginSheetModalForWindow:_window completionHandler:nil];
    return;
  }

  _presentation.appearance.colour_source =
      static_cast<panorama::TerrainColourSource>(_colourSourceControl.indexOfSelectedItem);
  _presentation.appearance.colourmap =
      static_cast<panorama::PresetColourmap>(_colourmapControl.indexOfSelectedItem);
  _presentation.colour_range = {
      static_cast<float>(*minimum),
      static_cast<float>(*maximum),
  };
  _presentation.use_surface_normals = _normalLightingControl.state == NSControlStateValueOn;
  _renderer->request_presentation(_presentation);

  const panorama::ImageSize next_image = {*width, *height};
  if (next_image.width != _image.width || next_image.height != _image.height) {
    _image = next_image;
    _renderer->request_inspection(std::nullopt);
    _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  }
}

- (void)drawInMTKView:(MTKView *)view {
  const panorama::app::PresentedFrame frame = _renderer->presented_frame();
  if (frame.texture != nil && (view.drawableSize.width != frame.image.width ||
                               view.drawableSize.height != frame.image.height)) {
    view.drawableSize = CGSizeMake(frame.image.width, frame.image.height);
    [_aspectFitView setAspectRatio:static_cast<CGFloat>(frame.image.width) /
                                   static_cast<CGFloat>(frame.image.height)];
  }
  id<CAMetalDrawable> drawable = view.currentDrawable;
  if (drawable == nil) {
    return;
  }
  id<MTLCommandBuffer> command = [_renderer->command_queue() commandBuffer];
  if (command == nil) {
    return;
  }

  if (frame.texture != nil) {
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:frame.texture
              sourceSlice:0U
              sourceLevel:0U
             sourceOrigin:MTLOriginMake(0U, 0U, 0U)
               sourceSize:MTLSizeMake(frame.texture.width, frame.texture.height, 1U)
                toTexture:drawable.texture
         destinationSlice:0U
         destinationLevel:0U
        destinationOrigin:MTLOriginMake(0U, 0U, 0U)];
    [blit endEncoding];
  } else {
    MTLRenderPassDescriptor *pass = view.currentRenderPassDescriptor;
    if (pass != nil) {
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
      id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
      [encoder endEncoding];
    }
  }
  [command presentDrawable:drawable];
  [command commit];

  if (frame.inspection_sequence != _displayedInspectionSequence) {
    _displayedInspectionSequence = frame.inspection_sequence;
    const bool matches_visible_frame =
        !frame.inspection.has_value() || frame.inspection->revision == frame.revision;
    if (_pointLockPending && frame.inspection_request_token == _pointLockRequestToken &&
        matches_visible_frame && frame.inspection.has_value()) {
      _pointLockPending = false;
      [self updatePointInfo:frame.inspection];
      if (frame.inspection->hit) {
        _pointInspectionLocked = true;
        _lockedPoint = frame.inspection;
        [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                    verticalFieldOfView:frame.vertical_field_of_view
                                                  image:frame.image];
        // The label now owns the immutable sampled values. Stop the renderer
        // resampling this screen pixel as subsequent camera views complete.
        _renderer->request_inspection(std::nullopt);
      } else {
        _pointInfoHeading.stringValue = @"Point Info";
      }
    } else if (_pointInspectionEnabled && !_pointInspectionLocked && !_pointLockPending) {
      [self updatePointInfo:matches_visible_frame ? frame.inspection : std::nullopt];
    }
  }

  if (!frame.error.empty()) {
    _window.title = [NSString stringWithFormat:@"panorama-app — error: %s", frame.error.c_str()];
  } else if (frame.revision != 0U && frame.revision != _displayedRevision) {
    _displayedRevision = frame.revision;
    const double fps = frame.milliseconds > 0.0 ? 1'000.0 / frame.milliseconds : 0.0;
    [self updateDebugInfoWithOrientation:frame.orientation
                     verticalFieldOfView:frame.vertical_field_of_view
                                   image:frame.image
                            milliseconds:frame.milliseconds
                                revision:frame.revision];
    [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                verticalFieldOfView:frame.vertical_field_of_view
                                              image:frame.image];
    _window.title = [NSString
        stringWithFormat:@"panorama-app — heading %.1f°, pitch %.1f° — %.1f ms (%.1f fps)",
                         frame.orientation.heading * panorama::app::kRadiansToDegrees,
                         frame.orientation.pitch * panorama::app::kRadiansToDegrees,
                         frame.milliseconds,
                         fps];
  }
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
  (void)view;
  (void)size;
}

@end

static NSToolbarItemIdentifier const kDebugToolbarItemIdentifier = @"panorama.debug-info";
static NSToolbarItemIdentifier const kPointInspectorToolbarItemIdentifier =
    @"panorama.point-inspector";

@interface PanoramaAppDelegate : NSObject <NSApplicationDelegate, NSToolbarDelegate> {
@private
  std::unique_ptr<panorama::app::ViewerRenderer> _renderer;
  NSWindow *_window;
  PanoramaController *_controller;
  ViewerOverlayView *_overlayView;
}
- (instancetype)initWithSettings:(panorama::app::ViewerSettings)settings;
@end

@implementation PanoramaAppDelegate

- (instancetype)initWithSettings:(panorama::app::ViewerSettings)settings {
  self = [super init];
  if (self != nil) {
    _renderer = std::make_unique<panorama::app::ViewerRenderer>(std::move(settings));
  }
  return self;
}

/// Put the overlay controls at the trailing edge, matching native macOS apps.
- (NSArray<NSToolbarItemIdentifier> *)toolbarDefaultItemIdentifiers:(NSToolbar *)toolbar {
  (void)toolbar;
  return @[
    NSToolbarFlexibleSpaceItemIdentifier,
    kPointInspectorToolbarItemIdentifier,
    kDebugToolbarItemIdentifier,
    NSToolbarToggleInspectorItemIdentifier,
  ];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarAllowedItemIdentifiers:(NSToolbar *)toolbar {
  (void)toolbar;
  return @[
    NSToolbarFlexibleSpaceItemIdentifier,
    NSToolbarSpaceItemIdentifier,
    kPointInspectorToolbarItemIdentifier,
    kDebugToolbarItemIdentifier,
    NSToolbarToggleInspectorItemIdentifier,
  ];
}

- (NSToolbarItem *)toolbar:(NSToolbar *)toolbar
        itemForItemIdentifier:(NSToolbarItemIdentifier)itemIdentifier
    willBeInsertedIntoToolbar:(BOOL)willBeInserted {
  (void)toolbar;
  (void)willBeInserted;
  const BOOL isInspector = [itemIdentifier isEqualToString:NSToolbarToggleInspectorItemIdentifier];
  const BOOL isDebug = [itemIdentifier isEqualToString:kDebugToolbarItemIdentifier];
  const BOOL isPointInspector =
      [itemIdentifier isEqualToString:kPointInspectorToolbarItemIdentifier];
  if (!isInspector && !isDebug && !isPointInspector) {
    return nil;
  }

  // Viewless bordered items receive the native toolbar appearance, including
  // Liquid Glass on supported macOS releases. The inspector also uses AppKit's
  // standard semantic identifier even though this delegate instantiates it.
  NSToolbarItem *item = [[NSToolbarItem alloc] initWithItemIdentifier:itemIdentifier];
  item.bordered = YES;
  if (isInspector) {
    item.target = _overlayView;
    item.label = @"Inspector";
    item.paletteLabel = @"Inspector";
    item.toolTip = @"Show or hide the render settings inspector";
    item.image = [NSImage imageWithSystemSymbolName:@"sidebar.right"
                           accessibilityDescription:@"Toggle Inspector"];
    item.action = @selector(toggleInspector:);
  } else if (isDebug) {
    item.target = _overlayView;
    item.label = @"Debug Info";
    item.paletteLabel = @"Debug Info";
    item.toolTip = @"Show or hide viewer debugging information";
    item.image = [NSImage imageWithSystemSymbolName:@"info.circle"
                           accessibilityDescription:@"Toggle Debug Info"];
    item.action = @selector(toggleDebugOverlay:);
  } else {
    item.label = @"Inspect Point";
    item.paletteLabel = @"Inspect Point";
    item.toolTip = @"Inspect terrain beneath the pointer; right-click to lock a point";
    item.image = [NSImage imageWithSystemSymbolName:@"scope"
                           accessibilityDescription:@"Toggle Point Inspection"];
    item.target = _controller;
    item.action = @selector(togglePointInspection:);
  }
  return item;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  (void)notification;
  const panorama::ImageSize image = _renderer->image();
  constexpr CGFloat kInspectorWidth = 270.0;
  constexpr NSSize kDebugSize = {220.0, 282.0};
  constexpr NSSize kPointInfoSize = {230.0, 174.0};
  const NSRect windowFrame = NSMakeRect(0.0, 0.0, image.width, image.height);
  const NSRect imageFrame = NSMakeRect(0.0, 0.0, image.width, image.height);
  _window = [[NSWindow alloc]
      initWithContentRect:windowFrame
                styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                          NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable |
                          NSWindowStyleMaskFullSizeContentView
                  backing:NSBackingStoreBuffered
                    defer:NO];
  _window.title = @"panorama-app — drag or use WASD/arrow keys to look around; scroll to zoom";

  PanoramaView *view = [[PanoramaView alloc] initWithFrame:imageFrame device:_renderer->device()];
  view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
  view.framebufferOnly = NO;
  view.autoResizeDrawable = NO;
  view.drawableSize = CGSizeMake(image.width, image.height);
  view.preferredFramesPerSecond = 30;
  view.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
  _controller = [[PanoramaController alloc] initWithRenderer:_renderer.get() window:_window];

  // Keep the traced ray field and Metal drawable at the same aspect ratio when
  // the inspector, window, or requested resolution changes; otherwise AppKit
  // scales the drawable non-uniformly and distorts terrain.
  const CGFloat imageAspect =
      static_cast<CGFloat>(image.width) / static_cast<CGFloat>(image.height);
  AspectFitContainerView *imageContainer =
      [[AspectFitContainerView alloc] initWithFrame:imageFrame
                                         renderView:view
                                        aspectRatio:imageAspect];

  NSViewController *settingsController = [_controller makeSettingsViewController];
  NSViewController *debugController = [_controller makeDebugViewController];
  NSViewController *pointInfoController = [_controller makePointInfoViewController];
  _overlayView = [[ViewerOverlayView alloc] initWithFrame:imageFrame
                                              contentView:imageContainer
                                             settingsView:settingsController.view
                                           inspectorWidth:kInspectorWidth
                                                debugView:debugController.view
                                                debugSize:kDebugSize
                                            pointInfoView:pointInfoController.view
                                            pointInfoSize:kPointInfoSize];
  [_controller attachPanoramaView:view overlayView:_overlayView aspectFitView:imageContainer];

  NSToolbar *toolbar = [[NSToolbar alloc] initWithIdentifier:@"panorama.toolbar"];
  toolbar.delegate = self;
  toolbar.displayMode = NSToolbarDisplayModeIconOnly;
  toolbar.allowsUserCustomization = NO;
  _window.toolbar = toolbar;
  // Standard AppKit chrome adopts Liquid Glass on current macOS releases.
  // Let the terrain extend beneath it instead of drawing an opaque titlebar
  // background that visually separates the toolbar from the scene.
  _window.toolbarStyle = NSWindowToolbarStyleAutomatic;
  _window.titleVisibility = NSWindowTitleHidden;
  _window.titlebarAppearsTransparent = YES;
  _window.titlebarSeparatorStyle = NSTitlebarSeparatorStyleNone;

  // This source-only AppKit application has no nib to supply its main menu.
  NSMenu *mainMenu = [[NSMenu alloc] initWithTitle:@""];
  NSMenuItem *applicationItem = [[NSMenuItem alloc] initWithTitle:@"" action:nil keyEquivalent:@""];
  [mainMenu addItem:applicationItem];
  NSMenu *applicationMenu = [[NSMenu alloc] initWithTitle:@"panorama-app"];
  [applicationMenu addItem:[[NSMenuItem alloc] initWithTitle:@"Quit panorama-app"
                                                      action:@selector(terminate:)
                                               keyEquivalent:@"q"]];
  applicationItem.submenu = applicationMenu;

  NSApp.mainMenu = mainMenu;

  view.panoramaController = _controller;
  view.delegate = _controller;
  NSView *windowContent = _overlayView;
  windowContent.translatesAutoresizingMaskIntoConstraints = YES;
  windowContent.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  _window.contentView = windowContent;
  [_window makeKeyAndOrderFront:nil];
  // Adding the unified toolbar settles its final content geometry when the
  // window is first shown. Apply the requested render size after that step.
  [_window setContentSize:imageFrame.size];
  [_window center];
  [_window makeFirstResponder:view];
  [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
  (void)sender;
  return YES;
}

@end

int main(int argc, const char *argv[]) {
  try {
    @autoreleasepool {
      panorama::app::ViewerSettings settings = panorama::app::parse_arguments(argc, argv);
      NSApplication *application = NSApplication.sharedApplication;
      application.activationPolicy = NSApplicationActivationPolicyRegular;
      PanoramaAppDelegate *delegate =
          [[PanoramaAppDelegate alloc] initWithSettings:std::move(settings)];
      application.delegate = delegate;
      [application run];
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "%s\n", exception.what());
    return EXIT_FAILURE;
  }
}
