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

struct ViewerSettings {
  std::filesystem::path tile_dir = "data/swissalti3d-10-level-0";
  uint64_t tile_cache_size_bytes = 128ULL * kBytesPerMiB;
  uint32_t workers = 8U;
  float max_distance = 600'000.0F;
  bool retain_quantized = false;
  ObserverLocation observer = {2623452.4, 1100502.2, 3415.0};
  ImageSize image = {960U, 540U};
  double horizontal_field_of_view = 70.0 * kDegreesToRadians;
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
      "  --horizontal-fov D    camera field of view in degrees (default: 70)\n"
      "  --heading D           initial heading clockwise from north (default: 0)\n"
      "  --pitch D             initial pitch above the horizon (default: 0)\n"
      "  --help                show this message\n"
      "\n"
      "Drag with the mouse or use WASD/arrow keys to look around.\n",
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
    } else if (option == "--horizontal-fov") {
      const double degrees = arguments::parse_finite_double(value, option);
      if (degrees <= 0.0 || degrees >= 180.0) {
        throw std::out_of_range("Horizontal field of view must be between 0 and 180 degrees");
      }
      settings.horizontal_field_of_view = degrees * kDegreesToRadians;
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

[[nodiscard]] RayField make_view(const ViewerSettings &settings, CameraOrientation orientation) {
  return make_camera_ray_field(
      settings.image,
      {
          orientation,
          CameraIntrinsics::from_horizontal_field_of_view(
              settings.image,
              settings.horizontal_field_of_view
          ),
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

struct PresentedFrame {
  id<MTLTexture> texture;
  CameraOrientation orientation;
  uint64_t revision;
  double milliseconds;
  std::string error;
  std::optional<PointInspection> inspection;
  uint64_t inspection_sequence;
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
        requested_presentation_(settings_.presentation) {
    RayField initial_field = make_view(settings_, settings_.orientation);
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
    request_view(settings_.orientation);
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
  void request_view(CameraOrientation orientation) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_orientation_ = orientation;
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
  void request_inspection(std::optional<InspectionPixel> pixel) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_inspection_ = pixel;
      inspection_pending_ = true;
    }
    changed_.notify_one();
  }

  [[nodiscard]] PresentedFrame presented_frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        presented_texture_,
        presented_orientation_,
        presented_revision_,
        frame_ms_,
        error_,
        presented_inspection_,
        presented_inspection_sequence_,
    };
  }

  [[nodiscard]] id<MTLDevice> device() const { return trace_->device(); }
  [[nodiscard]] id<MTLCommandQueue> command_queue() const { return trace_->command_queue(); }
  [[nodiscard]] ImageSize image() const { return settings_.image; }
  [[nodiscard]] ObserverLocation observer() const { return settings_.observer; }
  [[nodiscard]] double horizontal_field_of_view() const {
    return settings_.horizontal_field_of_view;
  }
  [[nodiscard]] float max_distance() const { return settings_.max_distance; }
  [[nodiscard]] CameraOrientation initial_orientation() const { return settings_.orientation; }
  [[nodiscard]] TerrainPresentationSettings initial_presentation() const {
    return settings_.presentation;
  }

private:
  [[nodiscard]] PointInspection inspect_pixel(InspectionPixel pixel, uint64_t revision) const {
    if (pixel.x >= settings_.image.width || pixel.y >= settings_.image.height) {
      throw std::out_of_range("Inspection pixel lies outside the ray image");
    }
    const size_t index =
        static_cast<size_t>(pixel.y) * static_cast<size_t>(settings_.image.width) + pixel.x;
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
      TerrainPresentationSettings presentation = {};
      uint64_t revision = 0U;
      bool trace_requested = false;
      bool presentation_requested = false;
      bool inspection_requested = false;
      std::optional<InspectionPixel> inspection_pixel;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] {
          return stopping_ || trace_pending_ || presentation_pending_ || inspection_pending_;
        });
        if (stopping_) {
          return;
        }
        orientation = requested_orientation_;
        presentation = requested_presentation_;
        revision = requested_revision_;
        trace_requested = trace_pending_;
        presentation_requested = presentation_pending_;
        inspection_requested = inspection_pending_;
        inspection_pixel = requested_inspection_;
        trace_pending_ = false;
        presentation_pending_ = false;
        inspection_pending_ = false;
      }

      try {
        @autoreleasepool {
          const auto started = std::chrono::steady_clock::now();
          if (trace_requested) {
            RayField field = make_view(settings_, orientation);
            trace_->trace(field);
            current_field_ = std::move(field);
          }

          if (presentation_requested) {
            Timer timer("GPU presentation");
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
            presented_orientation_ = orientation;
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
  TerrainPresentationSettings requested_presentation_ = {};
  std::optional<InspectionPixel> requested_inspection_;
  CameraOrientation presented_orientation_ = {};
  std::optional<PointInspection> presented_inspection_;
  id<MTLTexture> presented_texture_;
  uint64_t requested_revision_ = 0U;
  uint64_t presented_revision_ = 0U;
  uint64_t presented_inspection_sequence_ = 0U;
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

@interface PanoramaView : MTKView {
@private
  NSPoint _lastMouseLocation;
  NSTrackingArea *_inspectionTrackingArea;
  bool _pointInspectionEnabled;
}
@property(nonatomic, weak) PanoramaController *panoramaController;
- (void)setPointInspectionEnabled:(bool)enabled;
@end

@interface PanoramaController : NSObject <MTKViewDelegate> {
@private
  panorama::app::ViewerRenderer *_renderer;
  __weak NSWindow *_window;
  __weak PanoramaView *_panoramaView;
  __weak ViewerOverlayView *_overlayView;
  panorama::CameraOrientation _orientation;
  panorama::TerrainPresentationSettings _presentation;
  NSPopUpButton *_colourSourceControl;
  NSPopUpButton *_colourmapControl;
  NSTextField *_minimumControl;
  NSTextField *_maximumControl;
  NSButton *_normalLightingControl;
  NSTextField *_debugInfoLabel;
  NSTextField *_pointInfoLabel;
  uint64_t _displayedRevision;
  uint64_t _displayedInspectionSequence;
  bool _pointInspectionEnabled;
}
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView;
- (void)inspectPixelX:(uint32_t)x y:(uint32_t)y;
- (void)clearPointInspection;
- (void)togglePointInspection:(id)sender;
- (NSViewController *)makeSettingsViewController;
- (NSViewController *)makeDebugViewController;
- (NSViewController *)makePointInfoViewController;
@end

@implementation PanoramaView

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

- (void)mouseMoved:(NSEvent *)event {
  if (!_pointInspectionEnabled) {
    [super mouseMoved:event];
    return;
  }
  const NSRect bounds = self.bounds;
  const NSPoint location = [self convertPoint:event.locationInWindow fromView:nil];
  const uint32_t width = static_cast<uint32_t>(self.drawableSize.width);
  const uint32_t height = static_cast<uint32_t>(self.drawableSize.height);
  if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0 || width == 0U || height == 0U) {
    return;
  }

  // AppKit view coordinates rise from the bottom-left, whereas RayField rows
  // use image coordinates from the top-left.
  const double normalised_x =
      std::clamp((location.x - NSMinX(bounds)) / bounds.size.width, 0.0, 1.0);
  const double normalised_y =
      std::clamp((NSMaxY(bounds) - location.y) / bounds.size.height, 0.0, 1.0);
  const uint32_t x =
      std::min(width - 1U, static_cast<uint32_t>(normalised_x * static_cast<double>(width)));
  const uint32_t y =
      std::min(height - 1U, static_cast<uint32_t>(normalised_y * static_cast<double>(height)));
  [self.panoramaController inspectPixelX:x y:y];
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
  [self.panoramaController rotateHeading:heading pitch:pitch];
}

- (void)keyDown:(NSEvent *)event {
  constexpr double kStep = 2.0 * std::numbers::pi / 180.0;
  switch (event.keyCode) {
  case 123: // Left arrow.
    [self.panoramaController rotateHeading:-kStep pitch:0.0];
    break;
  case 124: // Right arrow.
    [self.panoramaController rotateHeading:kStep pitch:0.0];
    break;
  case 125: // Down arrow.
    [self.panoramaController rotateHeading:0.0 pitch:-kStep];
    break;
  case 126: // Up arrow.
    [self.panoramaController rotateHeading:0.0 pitch:kStep];
    break;
  default: {
    const NSString *characters = event.charactersIgnoringModifiers.lowercaseString;
    if ([characters isEqualToString:@"a"]) {
      [self.panoramaController rotateHeading:-kStep pitch:0.0];
    } else if ([characters isEqualToString:@"d"]) {
      [self.panoramaController rotateHeading:kStep pitch:0.0];
    } else if ([characters isEqualToString:@"s"]) {
      [self.panoramaController rotateHeading:0.0 pitch:-kStep];
    } else if ([characters isEqualToString:@"w"]) {
      [self.panoramaController rotateHeading:0.0 pitch:kStep];
    } else {
      [super keyDown:event];
    }
    break;
  }
  }
}

@end

/// Letterbox one fixed-aspect render view without imposing a fitting size on
/// its parent. The parent can therefore resize freely without distorting the
/// Metal output or acquiring a preferred size from the render aspect ratio.
@interface AspectFitContainerView : NSView {
@private
  NSView *_renderView;
  CGFloat _aspectRatio;
}
- (instancetype)initWithFrame:(NSRect)frame
                   renderView:(NSView *)renderView
                  aspectRatio:(CGFloat)aspectRatio;
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
    _presentation = renderer->initial_presentation();
  }
  return self;
}

- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta {
  _orientation.heading =
      std::remainder(_orientation.heading + headingDelta, 2.0 * std::numbers::pi);
  constexpr double kPitchLimit = 85.0 * std::numbers::pi / 180.0;
  _orientation.pitch = std::clamp(_orientation.pitch + pitchDelta, -kPitchLimit, kPitchLimit);
  _renderer->request_view(_orientation);
}

- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView {
  _panoramaView = panoramaView;
  _overlayView = overlayView;
}

- (void)inspectPixelX:(uint32_t)x y:(uint32_t)y {
  if (_pointInspectionEnabled) {
    _renderer->request_inspection(panorama::app::InspectionPixel{x, y});
  }
}

- (void)clearPointInspection {
  _renderer->request_inspection(std::nullopt);
}

- (void)togglePointInspection:(id)sender {
  _pointInspectionEnabled = !_pointInspectionEnabled;
  [_panoramaView setPointInspectionEnabled:_pointInspectionEnabled];
  [_overlayView setPointInfoVisible:_pointInspectionEnabled];
  if (!_pointInspectionEnabled) {
    [self clearPointInspection];
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

  NSTextField *heading = [NSTextField labelWithString:@"Render Settings"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

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
  apply.action = @selector(applyRenderSettings:);
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

  [self updateDebugInfoWithOrientation:_orientation milliseconds:0.0 revision:0U];
  return viewController;
}

- (void)updateDebugInfoWithOrientation:(panorama::CameraOrientation)orientation
                          milliseconds:(double)milliseconds
                              revision:(uint64_t)revision {
  if (_debugInfoLabel == nil) {
    return;
  }

  const panorama::ObserverLocation observer = _renderer->observer();
  const panorama::ImageSize image = _renderer->image();
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
                        "H. FOV       %8.2f°\n\nResolution   %4u × %4u\nMax range  %10.0f m",
                       performance,
                       static_cast<unsigned long long>(revision),
                       observer.easting,
                       observer.northing,
                       observer.elevation,
                       heading,
                       orientation.pitch * panorama::app::kRadiansToDegrees,
                       orientation.roll * panorama::app::kRadiansToDegrees,
                       _renderer->horizontal_field_of_view() * panorama::app::kRadiansToDegrees,
                       image.width,
                       image.height,
                       _renderer->max_distance()];
}

/// Build the compact hover readout shown while point inspection is enabled.
- (NSViewController *)makePointInfoViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 230.0, 174.0)];
  viewController.view = content;

  NSTextField *heading = [NSTextField labelWithString:@"Point Info"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _pointInfoLabel = [NSTextField labelWithString:@"Move the pointer over the rendered terrain."];
  _pointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _pointInfoLabel.maximumNumberOfLines = 0;
  _pointInfoLabel.lineBreakMode = NSLineBreakByWordWrapping;

  NSStackView *pointInfo = [NSStackView stackViewWithViews:@[ heading, _pointInfoLabel ]];
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
    _pointInfoLabel.stringValue = @"Move the pointer over the rendered terrain.";
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
- (void)applyRenderSettings:(id)sender {
  (void)sender;
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
}

- (void)drawInMTKView:(MTKView *)view {
  const panorama::app::PresentedFrame frame = _renderer->presented_frame();
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
    [self updatePointInfo:matches_visible_frame ? frame.inspection : std::nullopt];
  }

  if (!frame.error.empty()) {
    _window.title = [NSString stringWithFormat:@"panorama-app — error: %s", frame.error.c_str()];
  } else if (frame.revision != 0U && frame.revision != _displayedRevision) {
    _displayedRevision = frame.revision;
    const double fps = frame.milliseconds > 0.0 ? 1'000.0 / frame.milliseconds : 0.0;
    [self updateDebugInfoWithOrientation:frame.orientation
                            milliseconds:frame.milliseconds
                                revision:frame.revision];
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
    item.toolTip = @"Inspect terrain beneath the pointer";
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
  _window.title = @"panorama-app — drag or use WASD/arrow keys to look around";

  PanoramaView *view = [[PanoramaView alloc] initWithFrame:imageFrame device:_renderer->device()];
  view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
  view.framebufferOnly = NO;
  view.autoResizeDrawable = NO;
  view.drawableSize = CGSizeMake(image.width, image.height);
  view.preferredFramesPerSecond = 30;
  view.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
  _controller = [[PanoramaController alloc] initWithRenderer:_renderer.get() window:_window];

  // The traced ray field and Metal drawable have a fixed aspect ratio. Keep
  // that ratio when the inspector or window changes the content-pane shape;
  // otherwise AppKit scales the drawable non-uniformly and distorts terrain.
  const CGFloat imageAspect =
      static_cast<CGFloat>(image.width) / static_cast<CGFloat>(image.height);
  NSView *imageContainer = [[AspectFitContainerView alloc] initWithFrame:imageFrame
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
  [_controller attachPanoramaView:view overlayView:_overlayView];

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
