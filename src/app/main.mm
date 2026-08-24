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
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
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

struct PresentedFrame {
  id<MTLTexture> texture;
  CameraOrientation orientation;
  uint64_t revision;
  double milliseconds;
  std::string error;
};

/// Serial background renderer which coalesces input to the latest camera view.
class ViewerRenderer {
public:
  explicit ViewerRenderer(ViewerSettings settings)
      : settings_(std::move(settings)), requested_orientation_(settings_.orientation),
        requested_presentation_(settings_.presentation) {
    const RayField initial_field = make_view(settings_, settings_.orientation);
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

  [[nodiscard]] PresentedFrame presented_frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {presented_texture_, presented_orientation_, presented_revision_, frame_ms_, error_};
  }

  [[nodiscard]] id<MTLDevice> device() const { return trace_->device(); }
  [[nodiscard]] id<MTLCommandQueue> command_queue() const { return trace_->command_queue(); }
  [[nodiscard]] ImageSize image() const { return settings_.image; }
  [[nodiscard]] CameraOrientation initial_orientation() const { return settings_.orientation; }
  [[nodiscard]] TerrainPresentationSettings initial_presentation() const {
    return settings_.presentation;
  }

private:
  void render_loop() {
    while (true) {
      CameraOrientation orientation = {};
      TerrainPresentationSettings presentation = {};
      uint64_t revision = 0U;
      bool trace_requested = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] {
          return stopping_ || trace_pending_ || presentation_pending_;
        });
        if (stopping_) {
          return;
        }
        orientation = requested_orientation_;
        presentation = requested_presentation_;
        revision = requested_revision_;
        trace_requested = trace_pending_;
        trace_pending_ = false;
        presentation_pending_ = false;
      }

      try {
        @autoreleasepool {
          const auto started = std::chrono::steady_clock::now();
          if (trace_requested) {
            const RayField field = make_view(settings_, orientation);
            trace_->trace(field);
          }

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
          const double milliseconds =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                  .count();

          std::lock_guard<std::mutex> lock(mutex_);
          presented_texture_ = presentation_->texture();
          presented_orientation_ = orientation;
          presented_revision_ = revision;
          // The title reports camera-update throughput. A cheap appearance-only
          // pass should not replace it with a misleadingly high frame rate.
          if (trace_requested) {
            frame_ms_ = milliseconds;
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
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  CameraOrientation requested_orientation_ = {};
  TerrainPresentationSettings requested_presentation_ = {};
  CameraOrientation presented_orientation_ = {};
  id<MTLTexture> presented_texture_;
  uint64_t requested_revision_ = 0U;
  uint64_t presented_revision_ = 0U;
  double frame_ms_ = 0.0;
  std::string error_;
  bool trace_pending_ = false;
  bool presentation_pending_ = false;
  bool stopping_ = false;
};

} // namespace
} // namespace panorama::app

@class PanoramaController;

@interface PanoramaView : MTKView {
@private
  NSPoint _lastMouseLocation;
}
@property(nonatomic, weak) PanoramaController *panoramaController;
@end

@interface PanoramaController : NSObject <MTKViewDelegate> {
@private
  panorama::app::ViewerRenderer *_renderer;
  __weak NSWindow *_window;
  panorama::CameraOrientation _orientation;
  panorama::TerrainPresentationSettings _presentation;
  NSPopUpButton *_colourSourceControl;
  NSPopUpButton *_colourmapControl;
  NSTextField *_minimumControl;
  NSTextField *_maximumControl;
  NSButton *_normalLightingControl;
  uint64_t _displayedRevision;
}
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta;
- (NSViewController *)makeSettingsViewController;
@end

@implementation PanoramaView

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

/// Keep the render at the full window size and position the inspector above its
/// trailing edge. Frame-based layout is deliberate: overlapping children do
/// not define a useful Auto Layout fitting size for an NSWindow content view.
@interface InspectorOverlayView : NSView {
@private
  NSView *_contentView;
  NSVisualEffectView *_inspectorView;
  CGFloat _inspectorWidth;
  bool _inspectorVisible;
}
- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth;
- (void)toggleInspector:(id)sender;
@end

@implementation InspectorOverlayView

- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth {
  self = [super initWithFrame:frame];
  if (self != nil) {
    _contentView = contentView;
    _inspectorWidth = inspectorWidth;
    _inspectorVisible = true;
    self.wantsLayer = YES;
    self.layer.masksToBounds = YES;
    [self addSubview:_contentView];

    _inspectorView = [[NSVisualEffectView alloc] initWithFrame:NSZeroRect];
    _inspectorView.material = NSVisualEffectMaterialSidebar;
    _inspectorView.blendingMode = NSVisualEffectBlendingModeWithinWindow;
    _inspectorView.state = NSVisualEffectStateActive;
    [self addSubview:_inspectorView];

    settingsView.frame = _inspectorView.bounds;
    settingsView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [_inspectorView addSubview:settingsView];
    NSBox *separator = [[NSBox alloc] initWithFrame:NSMakeRect(0.0, 0.0, 1.0, frame.size.height)];
    separator.boxType = NSBoxSeparator;
    separator.autoresizingMask = NSViewHeightSizable;
    [_inspectorView addSubview:separator];
  }
  return self;
}

- (NSRect)inspectorFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  const CGFloat x = NSMaxX(bounds) - (visible ? _inspectorWidth : 0.0);
  return NSMakeRect(x, bounds.origin.y, _inspectorWidth, bounds.size.height);
}

- (void)layout {
  [super layout];
  _contentView.frame = self.bounds;
  _inspectorView.frame = [self inspectorFrameForVisible:_inspectorVisible];
}

- (void)toggleInspector:(id)sender {
  (void)sender;
  _inspectorVisible = !_inspectorVisible;
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _inspectorView.animator.frame = [self inspectorFrameForVisible:_inspectorVisible];
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
  _minimumControl.doubleValue = _presentation.colour_range.minimum;

  _maximumControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _maximumControl.doubleValue = _presentation.colour_range.maximum;

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
    [control.widthAnchor constraintGreaterThanOrEqualToConstant:145.0].active = YES;
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
  const double minimum = _minimumControl.doubleValue;
  const double maximum = _maximumControl.doubleValue;
  if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum <= minimum ||
      minimum < -std::numeric_limits<float>::max() || maximum > std::numeric_limits<float>::max()) {
    NSBeep();
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Invalid colour range";
    alert.informativeText = @"The maximum must be a finite value greater than the minimum.";
    [alert beginSheetModalForWindow:_window completionHandler:nil];
    return;
  }

  _presentation.appearance.colour_source =
      static_cast<panorama::TerrainColourSource>(_colourSourceControl.indexOfSelectedItem);
  _presentation.appearance.colourmap =
      static_cast<panorama::PresetColourmap>(_colourmapControl.indexOfSelectedItem);
  _presentation.colour_range = {
      static_cast<float>(minimum),
      static_cast<float>(maximum),
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

  if (!frame.error.empty()) {
    _window.title = [NSString stringWithFormat:@"panorama-app — error: %s", frame.error.c_str()];
  } else if (frame.revision != 0U && frame.revision != _displayedRevision) {
    _displayedRevision = frame.revision;
    const double fps = frame.milliseconds > 0.0 ? 1'000.0 / frame.milliseconds : 0.0;
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

static NSString *const kInspectorToolbarItemIdentifier = @"panorama.inspector";

@interface PanoramaAppDelegate : NSObject <NSApplicationDelegate, NSToolbarDelegate> {
@private
  std::unique_ptr<panorama::app::ViewerRenderer> _renderer;
  NSWindow *_window;
  PanoramaController *_controller;
  InspectorOverlayView *_overlayView;
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

/// Put the inspector control at the trailing edge, matching native macOS apps.
- (NSArray<NSToolbarItemIdentifier> *)toolbarDefaultItemIdentifiers:(NSToolbar *)toolbar {
  (void)toolbar;
  return @[ NSToolbarFlexibleSpaceItemIdentifier, kInspectorToolbarItemIdentifier ];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarAllowedItemIdentifiers:(NSToolbar *)toolbar {
  (void)toolbar;
  return @[
    NSToolbarFlexibleSpaceItemIdentifier,
    NSToolbarSpaceItemIdentifier,
    kInspectorToolbarItemIdentifier,
  ];
}

- (NSToolbarItem *)toolbar:(NSToolbar *)toolbar
        itemForItemIdentifier:(NSToolbarItemIdentifier)itemIdentifier
    willBeInsertedIntoToolbar:(BOOL)willBeInserted {
  (void)toolbar;
  (void)willBeInserted;
  if (![itemIdentifier isEqualToString:kInspectorToolbarItemIdentifier]) {
    return nil;
  }
  NSToolbarItem *item = [[NSToolbarItem alloc] initWithItemIdentifier:itemIdentifier];
  item.label = @"Inspector";
  item.paletteLabel = @"Inspector";
  item.toolTip = @"Show or hide the render settings inspector";
  item.image = [NSImage imageWithSystemSymbolName:@"sidebar.right"
                         accessibilityDescription:@"Toggle Inspector"];
  item.target = _overlayView;
  item.action = @selector(toggleInspector:);
  return item;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  (void)notification;
  const panorama::ImageSize image = _renderer->image();
  constexpr CGFloat kInspectorWidth = 270.0;
  const NSRect windowFrame = NSMakeRect(0.0, 0.0, image.width, image.height);
  const NSRect imageFrame = NSMakeRect(0.0, 0.0, image.width, image.height);
  _window = [[NSWindow alloc]
      initWithContentRect:windowFrame
                styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                          NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
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
  _overlayView = [[InspectorOverlayView alloc] initWithFrame:imageFrame
                                                 contentView:imageContainer
                                                settingsView:settingsController.view
                                              inspectorWidth:kInspectorWidth];

  NSToolbar *toolbar = [[NSToolbar alloc] initWithIdentifier:@"panorama.toolbar"];
  toolbar.delegate = self;
  toolbar.displayMode = NSToolbarDisplayModeIconOnly;
  toolbar.allowsUserCustomization = NO;
  _window.toolbar = toolbar;
  _window.toolbarStyle = NSWindowToolbarStyleUnified;
  _window.titleVisibility = NSWindowTitleHidden;

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
