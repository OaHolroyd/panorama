#include "arguments.h"
#include "gpu_image_renderer.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "synthetic_render_options.h"
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
      : settings_(std::move(settings)), requested_orientation_(settings_.orientation) {
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
        GpuTraceOutputRequirements{false, true}
    );
    presentation_ = std::make_unique<GpuImageRenderer>(
        trace_->device(),
        trace_->command_queue(),
        trace_->library(),
        settings_.image,
        GpuPresentationRequirements{false, false, true, false, false},
        MTLPixelFormatBGRA8Unorm
    );
    worker_ = std::thread([this] { render_loop(); });
    request(settings_.orientation);
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

  void request(CameraOrientation orientation) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_orientation_ = orientation;
      requested_revision_++;
      request_pending_ = true;
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

private:
  void render_loop() {
    while (true) {
      CameraOrientation orientation = {};
      uint64_t revision = 0U;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] { return stopping_ || request_pending_; });
        if (stopping_) {
          return;
        }
        orientation = requested_orientation_;
        revision = requested_revision_;
        request_pending_ = false;
      }

      try {
        @autoreleasepool {
          const auto started = std::chrono::steady_clock::now();
          const RayField field = make_view(settings_, orientation);
          trace_->trace(field);

          Timer timer("GPU presentation");
          const SyntheticRenderOptions options = {
              225.0 * kDegreesToRadians,
              35.0 * kDegreesToRadians,
              0.28F,
              TerrainColourSource::White,
              PresetColourmap::Viridis,
          };
          presentation_->render_synthetic(
              trace_->surface_gradients(),
              trace_->distances(),
              trace_->distances(),
              options,
              {0.0F, settings_.max_distance},
              timer
          );
          const double milliseconds =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                  .count();

          std::lock_guard<std::mutex> lock(mutex_);
          presented_texture_ = presentation_->texture();
          presented_orientation_ = orientation;
          presented_revision_ = revision;
          frame_ms_ = milliseconds;
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
  CameraOrientation presented_orientation_ = {};
  id<MTLTexture> presented_texture_;
  uint64_t requested_revision_ = 0U;
  uint64_t presented_revision_ = 0U;
  double frame_ms_ = 0.0;
  std::string error_;
  bool request_pending_ = false;
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
  NSPoint _lastMouseLocation;
  uint64_t _displayedRevision;
}
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta;
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

@implementation PanoramaController

- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window {
  self = [super init];
  if (self != nil) {
    _renderer = renderer;
    _window = window;
    _orientation = renderer->initial_orientation();
  }
  return self;
}

- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta {
  _orientation.heading =
      std::remainder(_orientation.heading + headingDelta, 2.0 * std::numbers::pi);
  constexpr double kPitchLimit = 85.0 * std::numbers::pi / 180.0;
  _orientation.pitch = std::clamp(_orientation.pitch + pitchDelta, -kPitchLimit, kPitchLimit);
  _renderer->request(_orientation);
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

@interface PanoramaAppDelegate : NSObject <NSApplicationDelegate> {
@private
  std::unique_ptr<panorama::app::ViewerRenderer> _renderer;
  NSWindow *_window;
  PanoramaController *_controller;
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

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  (void)notification;
  const panorama::ImageSize image = _renderer->image();
  const NSRect frame = NSMakeRect(0.0, 0.0, image.width, image.height);
  _window =
      [[NSWindow alloc] initWithContentRect:frame
                                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                            NSWindowStyleMaskMiniaturizable
                                    backing:NSBackingStoreBuffered
                                      defer:NO];
  _window.title = @"panorama-app — drag or use WASD/arrow keys to look around";

  PanoramaView *view = [[PanoramaView alloc] initWithFrame:frame device:_renderer->device()];
  view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
  view.framebufferOnly = NO;
  view.autoResizeDrawable = NO;
  view.drawableSize = CGSizeMake(image.width, image.height);
  view.preferredFramesPerSecond = 30;
  view.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
  _controller = [[PanoramaController alloc] initWithRenderer:_renderer.get() window:_window];
  view.panoramaController = _controller;
  view.delegate = _controller;
  _window.contentView = view;
  [_window center];
  [_window makeKeyAndOrderFront:nil];
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
