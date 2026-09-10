#include "viewer_renderer.h"
#include "arguments.h"
#include "coordinate_input.h"
#include "gpu_image_renderer.h"
#include "gpu_terrain_frame.h"
#include "inspection_coordinates.h"
#include "metalfx_upscaler.h"
#include "minimap.h"
#include "peak_catalogue.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "solar_position.h"
#include "synthetic_render_options.h"
#include "terrain_catalogue.h"
#include "terrain_presentation_settings.h"
#include "terrain_trace_session.h"
#include "timer.h"
#include "trace_diagnostics.h"
#include "visibility_mask.h"

#import <AppKit/AppKit.h>
#import <MapKit/MapKit.h>
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
#include <vector>

namespace panorama::app {
namespace {
constexpr double kFallbackEyeHeight = 2.0;
}

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

[[nodiscard]] NSString *format_clock_minutes(double value) {
  const int total = std::clamp(static_cast<int>(std::lround(value)), 0, 1439);
  return [NSString stringWithFormat:@"%02d:%02d", total / 60, total % 60];
}

/// Interpret calendar fields in the observer's time zone, then return the UTC
/// fields consumed by the solar-position calculation. The round trip rejects
/// local times which do not exist when daylight saving advances the clock.
[[nodiscard]] std::optional<CalendarDateTime>
local_date_time_to_utc(CalendarDateTime local, NSTimeZone *timeZone) {
  NSCalendar *localCalendar =
      [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
  localCalendar.timeZone = timeZone;
  NSDateComponents *localComponents = [[NSDateComponents alloc] init];
  localComponents.year = local.year;
  localComponents.month = local.month;
  localComponents.day = local.day;
  localComponents.hour = local.hour;
  localComponents.minute = local.minute;
  localComponents.timeZone = timeZone;
  NSDate *instant = [localCalendar dateFromComponents:localComponents];
  if (instant == nil) {
    return std::nullopt;
  }

  constexpr NSCalendarUnit fields = NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay |
                                    NSCalendarUnitHour | NSCalendarUnitMinute;
  NSDateComponents *roundTrip = [localCalendar components:fields fromDate:instant];
  if (roundTrip.year != local.year || roundTrip.month != local.month ||
      roundTrip.day != local.day || roundTrip.hour != local.hour ||
      roundTrip.minute != local.minute) {
    return std::nullopt;
  }

  NSCalendar *utcCalendar =
      [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
  utcCalendar.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
  NSDateComponents *utc = [utcCalendar components:fields fromDate:instant];
  return CalendarDateTime{
      static_cast<int>(utc.year),
      static_cast<int>(utc.month),
      static_cast<int>(utc.day),
      static_cast<int>(utc.hour),
      static_cast<int>(utc.minute),
  };
}

/// Convert UTC minutes relative to the supplied Gregorian date into an
/// observer-local clock label. Minutes may cross a UTC day boundary.
[[nodiscard]] NSString *
format_local_daylight_time(CalendarDateTime date, double utcMinutes, NSTimeZone *timeZone) {
  NSCalendar *utcCalendar =
      [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
  utcCalendar.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
  NSDateComponents *midnightComponents = [[NSDateComponents alloc] init];
  midnightComponents.year = date.year;
  midnightComponents.month = date.month;
  midnightComponents.day = date.day;
  midnightComponents.timeZone = utcCalendar.timeZone;
  NSDate *midnight = [utcCalendar dateFromComponents:midnightComponents];
  NSDate *instant = [midnight dateByAddingTimeInterval:utcMinutes * 60.0];

  NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
  formatter.locale = [[NSLocale alloc] initWithLocaleIdentifier:@"en_US_POSIX"];
  formatter.timeZone = timeZone;
  formatter.dateFormat = @"HH:mm";
  return [formatter stringFromDate:instant];
}

/// Describe the civil-time offset at the selected instant. Using the instant,
/// rather than the zone's current abbreviation, keeps historical and future
/// dates on the correct side of daylight-saving transitions.
[[nodiscard]] NSString *format_time_zone_summary(NSTimeZone *timeZone, CalendarDateTime utc) {
  NSCalendar *utcCalendar =
      [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
  utcCalendar.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
  NSDateComponents *components = [[NSDateComponents alloc] init];
  components.year = utc.year;
  components.month = utc.month;
  components.day = utc.day;
  components.hour = utc.hour;
  components.minute = utc.minute;
  components.timeZone = utcCalendar.timeZone;
  NSDate *instant = [utcCalendar dateFromComponents:components];

  const NSInteger offsetMinutes = [timeZone secondsFromGMTForDate:instant] / 60;
  const NSInteger absoluteMinutes = std::abs(offsetMinutes);
  NSString *offset =
      absoluteMinutes % 60 == 0
          ? [NSString
                stringWithFormat:@"UTC%c%ld", offsetMinutes < 0 ? '-' : '+', absoluteMinutes / 60]
          : [NSString stringWithFormat:@"UTC%c%ld:%02ld",
                                       offsetMinutes < 0 ? '-' : '+',
                                       absoluteMinutes / 60,
                                       absoluteMinutes % 60];
  NSString *abbreviation = [timeZone abbreviationForDate:instant];
  if (abbreviation == nil) {
    abbreviation = timeZone.name;
  }
  return [NSString stringWithFormat:@"%@ • %@", abbreviation, offset];
}

void print_usage(const char *program) {
  std::printf(
      "usage: %s [options]\n"
      "\n"
      "Interactively explore a prepared DTM using real-time persistent GPU terrain\n"
      "tracing. The viewer includes a minimap, appearance and camera settings, and\n"
      "interactive collision inspection.\n"
      "\n"
      "Input and camera options:\n"
      "  --raytracer MODE     metal-bvh (default) or software\n"
      "  --bvh-block-cells N  cells per BVH block axis (default: 4)\n"
      "  --bvh-cache-mib N    BVH cache and build budget (default: 2048)\n"
      "  --tile-dir DIR        prepared level-0 tile directory\n"
      "  --peak-gazetteer CSV  peak labels dataset (default: data/gazetteers/Alps589.csv)\n"
      "  --tile-cache-mib N    resident terrain-cache budget (default: 128)\n"
      "  --workers N           tile preparation workers (default: 8)\n"
      "  --max-distance M      horizontal range in metres (default: 600000)\n"
      "  --lod-scale V         terrain cell footprint multiplier; 0 keeps full detail\n"
      "                        (default: 1.5)\n"
      "  --discard-quantized   expand uint16 terrain to Float32 in the GPU atlas\n"
      "                        (default: retain uint16)\n"
      "  --trace-diagnostics   log frame timing, BVH cache, memory and display progress\n"
      "  --easting M           fixed observer easting (default: 2623452.4)\n"
      "  --northing M          fixed observer northing (default: 1100502.2)\n"
      "  --elevation M         fixed observer elevation (default: 3415)\n"
      "  --image-width N       internal render width (default: 1600)\n"
      "  --image-height N      internal render height (default: 900)\n"
      "  --vertical-fov D      vertical camera field of view in degrees (default: 70)\n"
      "  --heading D           initial heading clockwise from north (default: 0)\n"
      "  --pitch D             initial pitch above the horizon (default: 0)\n"
      "  --help                show this message\n"
      "\n"
      "Viewer controls:\n"
      "  Browse: drag or use WASD/arrow keys to look around; scroll to zoom.\n"
      "  Roam: use WASD to move; turn with arrow keys or mouse motion.\n"
      "        Configure movement and turning in the Position tab.\n"
      "  Cruise: move continuously; cursor displacement steers and W/S changes speed.\n"
      "          Optional Aircraft dynamics adds banked turns and energy exchange.\n"
      "  Press Space to pause or resume interactive viewer movement.\n"
      "  Use the toolbar to show the settings inspector, debug data, or minimap.\n",
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
    if (option == "--discard-quantized") {
      settings.discard_quantized = true;
      continue;
    }
    if (option == "--trace-diagnostics") {
      settings.trace_diagnostics = true;
      continue;
    }
    const std::string_view value = arguments::option_value(argc, argv, index, option);
    if (option == "--tile-dir") {
      settings.tile_dir = value;
    } else if (option == "--peak-gazetteer") {
      settings.peak_gazetteer = value;
    } else if (option == "--raytracer") {
      settings.raytracer = arguments::parse_raytracer(value);
    } else if (option == "--bvh-block-cells") {
      settings.bvh_block_cells = arguments::parse_uint32(value, option, false);
    } else if (option == "--bvh-cache-mib") {
      const uint64_t size = arguments::parse_uint64(value, option);
      if (size == 0U || size > std::numeric_limits<uint64_t>::max() / kBytesPerMiB)
        throw std::out_of_range("BVH cache is outside the supported byte range");
      settings.bvh_cache_size_bytes = size * kBytesPerMiB;
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
    } else if (option == "--lod-scale") {
      const double scale = arguments::parse_finite_double(value, option);
      if (scale < 0.0 || scale > std::numeric_limits<float>::max()) {
        throw std::out_of_range("LOD scale must be a nonnegative float32 value");
      }
      settings.lod_scale = static_cast<float>(scale);
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
  return settings;
}

[[nodiscard]] RayFieldRequest
make_view(ImageSize image, CameraOrientation orientation, double vertical_field_of_view) {
  return RayFieldRequest{
      image,
      CameraProjection{
          orientation,
          CameraIntrinsics::from_vertical_field_of_view(image, vertical_field_of_view),
          NoDistortion{},
      }};
}

/// Reproject a sampled terrain location through the viewer's ideal pinhole
/// camera. The curvature adjustment reconstructs the apparent vertical ray
/// displacement used when the original terrain collision was recorded.
[[nodiscard]] LockedPointProjection project_locked_point(
    TerrainPoint point,
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

/// Decode one IEEE float16 value emitted by Metal without depending on a SIMD
/// vector ABI shared between C++ and Metal.
[[nodiscard]] float float_from_half_bits(uint16_t bits) {
  _Float16 value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<float>(value);
}

/// A sampling presentation pass permits upscaled, custom-sized, and retained
/// frames to fill the current drawable without CPU-side texture copies.
class FullscreenPresentation {
public:
  FullscreenPresentation(id<MTLDevice> device, id<MTLLibrary> library) {
    id<MTLFunction> vertex = [library newFunctionWithName:@"fullscreen_presentation_vertex"];
    id<MTLFunction> fragment = [library newFunctionWithName:@"fullscreen_presentation_fragment"];
    MTLRenderPipelineDescriptor *descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.vertexFunction = vertex;
    descriptor.fragmentFunction = fragment;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    NSError *error = nil;
    pipeline_ = vertex == nil || fragment == nil
                    ? nil
                    : [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (pipeline_ == nil)
      throw std::runtime_error("Could not create fullscreen presentation pipeline");
  }
  void
  encode(id<MTLCommandBuffer> command, id<MTLTexture> source, id<MTLTexture> destination) const {
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = destination;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
    if (encoder == nil)
      throw std::runtime_error("Could not create fullscreen presentation encoder");
    [encoder setRenderPipelineState:pipeline_];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
  }

private:
  id<MTLRenderPipelineState> pipeline_;
};

/// Serial background renderer which coalesces input to the latest camera view.
class ViewerRendererImpl final : public ViewerRenderer {
public:
  explicit ViewerRendererImpl(ViewerSettings settings)
      : settings_(std::move(settings)), requested_orientation_(settings_.orientation),
        requested_vertical_field_of_view_(settings_.vertical_field_of_view),
        requested_image_(settings_.image), requested_presentation_(settings_.presentation),
        presented_vertical_field_of_view_(settings_.vertical_field_of_view),
        presented_image_(settings_.image) {
    RayFieldRequest initial_field =
        make_view(settings_.image, settings_.orientation, settings_.vertical_field_of_view);
    const auto traceConfig =
        [&](ObserverLocation observer, bool allowFallback, bool bilinear, bool c1Normals) {
          return RaytraceConfig{
              settings_.tile_dir,
              observer,
              settings_.max_distance,
              0U,
              settings_.tile_cache_size_bytes,
              settings_.workers,
              !settings_.discard_quantized,
              bilinear,
              c1Normals,
              allowFallback,
              settings_.lod_scale,
              settings_.raytracer,
              settings_.bvh_block_cells,
              settings_.bvh_cache_size_bytes,
          };
        };
    const ObserverLocation requestedObserver = settings_.observer;
    trace_ = std::make_unique<TerrainTraceSession>(
        traceConfig(requestedObserver, true, settings_.bilinear_collisions, settings_.c1_normals),
        initial_field,
        GpuTraceOutputRequirements{
            .surface_gradients = true,
            .elevations = true,
            .debugging_info = true,
        }
    );
    settings_.observer = trace_->observer();
    try {
      peak_catalogue_ = PeakCatalogue::load(settings_.peak_gazetteer, trace_->crs());
    } catch (const std::exception &error) {
      std::fprintf(stderr, "Peak labels disabled: %s\n", error.what());
    }
    observer_fallback_used_ = settings_.observer.easting != requestedObserver.easting ||
                              settings_.observer.northing != requestedObserver.northing;
    device_ = trace_->device();
    // Ground queries share the trace session's catalogue and resident atlas;
    // constructing a second app-local tile cache would duplicate I/O.
    if (const std::optional<float> ground =
            trace_->sample_terrain(settings_.observer.easting, settings_.observer.northing)) {
      if (observer_fallback_used_) {
        settings_.observer.elevation = static_cast<double>(*ground) + kFallbackEyeHeight;
        trace_ = std::make_unique<TerrainTraceSession>(
            traceConfig(
                settings_.observer,
                false,
                settings_.bilinear_collisions,
                settings_.c1_normals
            ),
            initial_field,
            GpuTraceOutputRequirements{
                .surface_gradients = true,
                .elevations = true,
                .debugging_info = true,
            }
        );
        if (trace_->device() != device_) {
          throw std::runtime_error("Observer fallback selected a different Metal device");
        }
        observer_ground_clearance_ = kFallbackEyeHeight;
      } else {
        observer_ground_clearance_ = std::max(0.0, settings_.observer.elevation - *ground);
      }
    }
    if (observer_fallback_used_) {
      std::fprintf(
          stderr,
          "Requested observer is outside the prepared terrain; starting at (%.3f, %.3f, %.1f).\n",
          settings_.observer.easting,
          settings_.observer.northing,
          settings_.observer.elevation
      );
    }
    device_ = trace_->device();
    display_queue_ = trace_->command_queue();
    library_ = trace_->library();
    if (display_queue_ == nil)
      throw std::runtime_error("Could not create viewer display command queue");
    presentation_ = std::make_unique<GpuImageRenderer>(
        device_,
        display_queue_,
        library_,
        settings_.image,
        GpuPresentationRequirements{
            .scalar_diagnostics = false,
            .normal_diagnostics = false,
            .debugging_diagnostics = false,
            .white_synthetic = true,
            .synthetic_scalar_colour = true,
            .host_readback = false,
        },
        MTLPixelFormatBGRA8Unorm
    );
    metalfx_ = std::make_unique<MetalFxUpscaler>(device_, MTLPixelFormatBGRA8Unorm);
    fullscreen_presentation_ = std::make_unique<FullscreenPresentation>(device_, library_);
    visibility_ = std::make_unique<GpuVisibilityPointProjector>(device_, library_);
    current_field_ = std::move(initial_field);
    current_output_image_ = settings_.image;
    current_observer_ = settings_.observer;
    current_orientation_ = settings_.orientation;
    current_vertical_field_of_view_ = settings_.vertical_field_of_view;
    requested_observer_ = settings_.observer;
    requested_lod_scale_ = settings_.lod_scale;
    requested_metalfx_ = {settings_.metalfx_activation, settings_.metalfx_preset, false};
    requested_raytracer_ = settings_.raytracer;
    requested_bilinear_collisions_ = settings_.bilinear_collisions;
    requested_c1_normals_ = settings_.c1_normals;
    presented_observer_ = settings_.observer;
    if (settings_.trace_diagnostics)
      diagnostics_ = std::make_unique<diagnostics::Monitor>(device_);
    worker_ = std::thread([this] { render_loop(); });
    request_view(settings_.orientation, settings_.vertical_field_of_view, settings_.image);
  }

  ViewerRendererImpl(const ViewerRendererImpl &) = delete;
  ViewerRendererImpl &operator=(const ViewerRendererImpl &) = delete;

  ~ViewerRendererImpl() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  /// Hidden maps neither produce nor retain full-frame visibility snapshots.
  /// A generation prevents an already encoded producer from publishing after
  /// hide/show. Showing requests one fresh trace, even with a stationary camera.
  void request_minimap_enabled(bool enabled) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (minimap_enabled_ == enabled)
        return;
      minimap_enabled_ = enabled;
      ++minimap_generation_;
      minimap_changed_ = true;
      presented_visibility_points_ = nil;
      if (enabled) {
        ++requested_revision_;
        trace_pending_ = true;
        presentation_pending_ = true;
      }
    }
    changed_.notify_one();
  }

  void request_peak_labels_enabled(bool enabled) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (peak_labels_enabled_ == enabled)
        return;
      peak_labels_enabled_ = enabled && peak_catalogue_.has_value();
      ++peak_labels_generation_;
      peak_labels_changed_ = true;
      if (!peak_labels_enabled_)
        presented_peak_labels_.reset();
    }
    changed_.notify_one();
  }

  /// Request a new camera trace; intermediate input events are coalesced.
  void request_view(
      CameraOrientation orientation,
      double vertical_field_of_view,
      ImageSize image
  ) override {
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

  void
  request_metalfx(MetalFxActivation activation, MetalFxPreset preset, bool interacting) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const MetalFxSelection requested = {activation, preset, interacting};
      if (requested_metalfx_ == requested)
        return;
      requested_metalfx_ = requested;
      requested_revision_++;
      trace_pending_ = true;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Switch primary tracing on the render worker, coalescing rapid selection changes.
  void request_raytracer(Raytracer raytracer) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (requested_raytracer_ == raytracer)
        return;
      requested_raytracer_ = raytracer;
      requested_revision_++;
      trace_pending_ = true;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  [[nodiscard]] Raytracer requested_raytracer() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return requested_raytracer_;
  }

  /// Re-present the completed trace with new appearance settings.
  void request_presentation(TerrainPresentationSettings presentation) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_presentation_ = presentation;
      requested_revision_++;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Re-trace with a new per-tile terrain LOD policy. The render worker owns
  /// the session, so the UI only publishes the newest scale here.
  void request_lod_scale(float lodScale) override {
    if (!std::isfinite(lodScale) || lodScale < 0.0F) {
      throw std::invalid_argument("Terrain LOD scale must be finite and nonnegative");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (requested_lod_scale_ == lodScale)
        return;
      requested_lod_scale_ = lodScale;
      requested_revision_++;
      lod_scale_pending_ = true;
      trace_pending_ = true;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Switch among the precompiled collision and normal-interpolation pipelines.
  /// Terrain residency remains valid, but the current view must be traced again.
  void request_collision_settings(bool bilinear, bool c1Normals) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (requested_bilinear_collisions_ == bilinear && requested_c1_normals_ == c1Normals)
        return;
      requested_bilinear_collisions_ = bilinear;
      requested_c1_normals_ = c1Normals;
      requested_revision_++;
      collision_settings_pending_ = true;
      trace_pending_ = true;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Coalesce hover events to the latest view-relative location. A missing
  /// location clears the sample when inspection is disabled or leaves the image.
  uint64_t request_inspection(std::optional<InspectionLocation> location) override {
    uint64_t token = 0U;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_inspection_ = location;
      requested_inspection_token_++;
      token = requested_inspection_token_;
      inspection_pending_ = true;
    }
    changed_.notify_one();
    return token;
  }

  /// Sample the actual terrain under a minimap coordinate on the render worker.
  uint64_t request_map_point(MapCoordinate coordinate) override {
    uint64_t token = 0U;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_map_coordinate_ = coordinate;
      requested_map_point_token_++;
      token = requested_map_point_token_;
      map_point_pending_ = true;
    }
    changed_.notify_one();
    return token;
  }

  /// Track one locked world point and depth-test it after every camera trace.
  uint64_t request_target_visibility(std::optional<TerrainPoint> point) override {
    uint64_t token = 0U;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_target_ = point;
      requested_target_token_++;
      token = requested_target_token_;
      target_pending_ = true;
    }
    changed_.notify_one();
    return token;
  }

  /// Rebuild spatial tracing around a sampled terrain point at the requested
  /// eye height. Supplying the clearance with the destination keeps map jumps
  /// and height edits ordered when requests are coalesced.
  void request_observer_at(TerrainPoint point, double groundClearance) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      observer_ground_clearance_ = groundClearance;
      requested_observer_ = {
          point.easting,
          point.northing,
          static_cast<double>(point.elevation) + groundClearance,
      };
      requested_revision_++;
      // A discrete destination supersedes a queued movement sample. An
      // already-running sample may still finish, but this request is processed
      // immediately afterwards and becomes the published observer.
      roam_pending_ = false;
      observer_pending_ = true;
      trace_pending_ = true;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Change height above the ground beneath the latest requested position.
  /// This avoids resampling the current location and also behaves correctly if
  /// a destination request has not yet completed its trace.
  void request_ground_clearance(double groundClearance) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_observer_.elevation += groundClearance - observer_ground_clearance_;
      observer_ground_clearance_ = groundClearance;
      requested_revision_++;
      roam_pending_ = false;
      observer_pending_ = true;
      trace_pending_ = true;
      presentation_pending_ = true;
    }
    changed_.notify_one();
  }

  /// Coalesce continuous movement to the newest requested horizontal point.
  /// Ground sampling happens on the render worker so the main thread never
  /// blocks on tile I/O while keys are held.
  uint64_t
  request_roam(MapCoordinate coordinate, RoamAltitudeMode altitudeMode, double height) override {
    uint64_t token = 0U;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_roam_coordinate_ = coordinate;
      requested_roam_altitude_mode_ = altitudeMode;
      requested_roam_height_ = height;
      requested_roam_token_++;
      token = requested_roam_token_;
      requested_revision_++;
      roam_pending_ = true;
    }
    changed_.notify_one();
    return token;
  }

  [[nodiscard]] PresentedFrame presented_frame() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return presented_frame_locked();
  }

private:
  /// Caller holds mutex_ so image resources and UI metadata describe one frame.
  [[nodiscard]] PresentedFrame presented_frame_locked() const {
    return {
        .texture = presented_texture_,
        .visibility_points = presented_visibility_points_,
        .image = presented_image_,
        .output_image = presented_output_image_,
        .metalfx_enabled = presented_metalfx_enabled_,
        .metalfx_preset = presented_metalfx_preset_,
        .orientation = presented_orientation_,
        .vertical_field_of_view = presented_vertical_field_of_view_,
        .revision = presented_revision_,
        .milliseconds = frame_ms_,
        .gpu_milliseconds = frame_gpu_ms_,
        .streamed = frame_streamed_,
        .error = error_,
        .inspection = presented_inspection_,
        .inspection_sequence = presented_inspection_sequence_,
        .inspection_request_token = presented_inspection_token_,
        .observer = presented_observer_,
        .map_point = presented_map_point_,
        .map_point_sequence = presented_map_point_sequence_,
        .map_point_request_token = presented_map_point_token_,
        .target_visibility = presented_target_visibility_,
        .target_visibility_sequence = presented_target_visibility_sequence_,
        .peak_labels = presented_peak_labels_,
        .roam_result = presented_roam_result_,
        .roam_result_sequence = presented_roam_result_sequence_,
    };
  }

public:
  /// Select the latest frame after drawable acquisition, then encode and commit
  /// while publication is locked. Never abandon an encoded drawable because a
  /// newer frame arrived during acquisition. Queue ordering keeps this blit
  /// ahead of the source texture's next write after the publication lock opens.
  [[nodiscard]] bool submit_presentation(
      id<MTLCommandBuffer> command,
      PresentedFrame &frame,
      id<CAMetalDrawable> drawable
  ) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame.revision != presented_revision_ || frame.texture != presented_texture_) {
      if (diagnostics::enabled)
        ++diagnostics::display.refreshed;
    }
    frame = presented_frame_locked();
    diagnostics::display.mark("encode");
    if (frame.texture != nil) {
      fullscreen_presentation_->encode(command, frame.texture, drawable.texture);
    } else {
      MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = drawable.texture;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
      id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
      if (encoder == nil)
        return false;
      [encoder endEncoding];
    }
    diagnostics::display.mark("submit");
    diagnostics::track_submission(diagnostics::display, command, drawable);
    [command presentDrawable:drawable];
    [command commit];
    return true;
  }

  [[nodiscard]] id<MTLDevice> device() const override { return device_; }
  [[nodiscard]] id<MTLCommandQueue> command_queue() const override { return display_queue_; }
  [[nodiscard]] id<MTLLibrary> library() const override { return library_; }
  [[nodiscard]] ImageSize initial_image() const override { return settings_.image; }
  [[nodiscard]] ObserverLocation observer() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return presented_observer_;
  }
  [[nodiscard]] Crs terrain_crs() const override { return trace_->crs(); }
  [[nodiscard]] const TerrainCoverage &terrain_coverage() const override {
    return trace_->terrain_coverage();
  }
  [[nodiscard]] bool observer_used_fallback() const override { return observer_fallback_used_; }
  [[nodiscard]] double ground_clearance() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return observer_ground_clearance_;
  }
  [[nodiscard]] double initial_vertical_field_of_view() const override {
    return settings_.vertical_field_of_view;
  }
  [[nodiscard]] float max_distance() const override { return settings_.max_distance; }
  [[nodiscard]] float initial_lod_scale() const override { return settings_.lod_scale; }
  [[nodiscard]] MetalFxActivation initial_metalfx_activation() const override {
    return settings_.metalfx_activation;
  }
  [[nodiscard]] MetalFxPreset initial_metalfx_preset() const override {
    return settings_.metalfx_preset;
  }
  [[nodiscard]] bool metalfx_supported() const override { return metalfx_->supported(); }
  [[nodiscard]] bool peak_labels_available() const override { return peak_catalogue_.has_value(); }
  [[nodiscard]] bool initial_bilinear_collisions() const override {
    return settings_.bilinear_collisions;
  }
  [[nodiscard]] bool initial_c1_normals() const override { return settings_.c1_normals; }
  [[nodiscard]] CameraOrientation initial_orientation() const override {
    return settings_.orientation;
  }
  [[nodiscard]] TerrainPresentationSettings initial_presentation() const override {
    return settings_.presentation;
  }

private:
  [[nodiscard]] PeakLabelFrame visible_peaks() const {
    PeakLabelFrame result = {.revision = current_revision_,
                             .output_image = current_output_image_,
                             .peaks = {}};
    if (!peak_catalogue_.has_value() || current_field_.image.width == 0U ||
        current_field_.image.height == 0U)
      return result;
    const auto *distances = static_cast<const float *>(trace_->distances().contents);
    if (distances == nullptr)
      return result;
    size_t within_range = 0U;
    size_t onscreen = 0U;
    size_t sampled = 0U;
    double best_margin = -std::numeric_limits<double>::infinity();
    for (const PeakRecord &peak : peak_catalogue_->peaks()) {
      const double east = peak.easting - current_observer_.easting;
      const double north = peak.northing - current_observer_.northing;
      const double horizontal = std::hypot(east, north);
      if (horizontal > settings_.max_distance)
        continue;
      ++within_range;
      const LockedPointProjection projection = project_locked_point(
          {peak.easting, peak.northing, peak.elevation},
          current_observer_,
          current_output_image_,
          current_vertical_field_of_view_,
          current_orientation_
      );
      if (!projection.onscreen)
        continue;
      ++onscreen;
      const auto pixel = inspection_pixel(
          {projection.pixel_x / current_output_image_.width,
           projection.pixel_y / current_output_image_.height},
          current_field_.image
      );
      if (!pixel)
        continue;
      ++sampled;
      const double angular_pixel = current_vertical_field_of_view_ / current_field_.image.height;
      // Gazetteer summits and the rendered DEM need not identify the same
      // horizontal sample, especially once LOD coarsening is active. Preserve
      // a modest metre-scale allowance when high output resolution makes the
      // angular pixel footprint very small.
      const double tolerance = std::max(50.0, 2.0 * horizontal * std::tan(angular_pixel));
      float farthest = 0.0F;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int x = static_cast<int>(pixel->x) + dx;
          const int y = static_cast<int>(pixel->y) + dy;
          if (x < 0 || y < 0 || x >= static_cast<int>(current_field_.image.width) ||
              y >= static_cast<int>(current_field_.image.height))
            continue;
          const float distance = distances
              [static_cast<size_t>(y) * current_field_.image.width + static_cast<size_t>(x)];
          if (std::isfinite(distance))
            farthest = std::max(farthest, distance);
        }
      }
      best_margin = std::max(best_margin, static_cast<double>(farthest) + tolerance - horizontal);
      if (farthest + tolerance < horizontal)
        continue;
      result.peaks.push_back(
          {peak.id,
           projection.pixel_x,
           projection.pixel_y,
           horizontal,
           peak.prominence,
           peak.elevation,
           peak.name}
      );
    }
    std::ranges::sort(result.peaks, [](const VisiblePeak &left, const VisiblePeak &right) {
      if (left.prominence != right.prominence)
        return left.prominence > right.prominence;
      if (left.distance != right.distance)
        return left.distance < right.distance;
      return left.peak_id < right.peak_id;
    });
    if (result.peaks.size() > 100U)
      result.peaks.resize(100U);
    if (settings_.trace_diagnostics) {
      std::printf(
          "Peak labels: %zu/%zu visible, %zu onscreen, %zu within range, best margin %.1f m\n",
          result.peaks.size(),
          sampled,
          onscreen,
          within_range,
          best_margin
      );
    }
    return result;
  }

  [[nodiscard]] bool target_is_occluded(
      TerrainPoint point,
      CameraOrientation orientation,
      double vertical_field_of_view
  ) const {
    const LockedPointProjection projection = project_locked_point(
        point,
        current_observer_,
        current_output_image_,
        vertical_field_of_view,
        orientation
    );
    if (!projection.onscreen) {
      return false;
    }
    const double target_distance = std::hypot(
        point.easting - current_observer_.easting,
        point.northing - current_observer_.northing
    );
    const double angular_pixel = vertical_field_of_view / current_field_.image.height;
    const double tolerance = std::max(5.0, 2.0 * target_distance * std::tan(angular_pixel));
    if (target_distance > static_cast<double>(settings_.max_distance) + tolerance) {
      return true;
    }
    const auto *distances = static_cast<const float *>(trace_->distances().contents);
    if (distances == nullptr || current_field_.image.width == 0U ||
        current_field_.image.height == 0U) {
      throw std::runtime_error("Could not map target-visibility distance buffer");
    }
    const auto pixel = inspection_pixel(
        {projection.pixel_x / current_output_image_.width,
         projection.pixel_y / current_output_image_.height},
        current_field_.image
    );
    if (!pixel)
      return false;
    const size_t index = static_cast<size_t>(pixel->y) * current_field_.image.width + pixel->x;
    const float collision_distance = distances[index];
    if (!(collision_distance > 0.0F) || !std::isfinite(collision_distance)) {
      return false;
    }
    return static_cast<double>(collision_distance) + tolerance < target_distance;
  }

  [[nodiscard]] std::optional<PointInspection> inspect_location(InspectionLocation location) const {
    const auto mapped_pixel = inspection_pixel(location, current_field_.image);
    if (!mapped_pixel)
      return std::nullopt;
    const InspectionPixel pixel = *mapped_pixel;
    const size_t index =
        static_cast<size_t>(pixel.y) * static_cast<size_t>(current_field_.image.width) + pixel.x;
    const auto *distances = static_cast<const float *>(trace_->distances().contents);
    const auto *elevations = static_cast<const float *>(trace_->elevations().contents);
    const auto *gradients = static_cast<const uint32_t *>(trace_->surface_gradients().contents);
    if (distances == nullptr || elevations == nullptr || gradients == nullptr ||
        index >= trace_->ray_directions().length / sizeof(RayDirection)) {
      throw std::runtime_error("Could not map point-inspection buffers");
    }

    PointInspection result = {
        .pixel = pixel,
        .revision = current_revision_,
        .hit = false,
        .distance = 0.0F,
        .elevation = 0.0F,
        .easting = 0.0,
        .northing = 0.0,
        .slope_degrees = 0.0F,
        .aspect_degrees = 0.0F,
        .map_selected = false,
    };
    const float distance = distances[index];
    if (!(distance > 0.0F) || !std::isfinite(distance)) {
      return result;
    }

    const RayDirection &ray =
        static_cast<const RayDirection *>(trace_->ray_directions().contents)[index];
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
        current_observer_.easting + static_cast<double>(distance) * static_cast<double>(ray.x);
    result.northing =
        current_observer_.northing + static_cast<double>(distance) * static_cast<double>(ray.y);
    result.slope_degrees = slope * static_cast<float>(kRadiansToDegrees);
    result.aspect_degrees = static_cast<float>(aspect);
    return result;
  }

  void render_loop() {
    while (true) {
      CameraOrientation orientation = {};
      double vertical_field_of_view = 0.0;
      ImageSize image = {};
      MetalFxSelection metalfx_selection;
      TerrainPresentationSettings presentation = {};
      uint64_t revision = 0U;
      uint64_t minimap_generation = 0;
      bool minimap_enabled = false;
      bool peak_labels_enabled = false;
      bool peak_labels_requested = false;
      uint64_t peak_labels_generation = 0U;
      bool trace_requested = false;
      bool presentation_requested = false;
      bool inspection_requested = false;
      bool observer_requested = false;
      bool map_point_requested = false;
      bool target_requested = false;
      bool roam_requested = false;
      bool lod_scale_requested = false;
      bool collision_settings_requested = false;
      bool bilinear_collisions = false;
      bool c1_normals = false;
      Raytracer raytracer = Raytracer::Software;
      float lod_scale = 0.0F;
      std::optional<InspectionLocation> inspection_location;
      uint64_t inspection_token = 0U;
      ObserverLocation observer = {};
      MapCoordinate map_coordinate = {};
      uint64_t map_point_token = 0U;
      MapCoordinate roam_coordinate = {};
      RoamAltitudeMode roam_altitude_mode = RoamAltitudeMode::FollowTerrain;
      double roam_height = 0.0;
      uint64_t roam_token = 0U;
      double current_ground_clearance = 0.0;
      std::optional<TerrainPoint> target;
      uint64_t target_token = 0U;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] {
          return stopping_ || trace_pending_ || presentation_pending_ || inspection_pending_ ||
                 observer_pending_ || map_point_pending_ || target_pending_ || roam_pending_ ||
                 minimap_changed_ || peak_labels_changed_;
        });
        if (stopping_) {
          return;
        }
        minimap_enabled = minimap_enabled_;
        minimap_generation = minimap_generation_;
        minimap_changed_ = false;
        peak_labels_enabled = peak_labels_enabled_;
        peak_labels_generation = peak_labels_generation_;
        peak_labels_requested = peak_labels_changed_ || (peak_labels_enabled && trace_pending_);
        peak_labels_changed_ = false;
        if (!minimap_enabled)
          current_visibility_points_ = nil;
        orientation = requested_orientation_;
        vertical_field_of_view = requested_vertical_field_of_view_;
        image = requested_image_;
        metalfx_selection = requested_metalfx_;
        presentation = requested_presentation_;
        revision = requested_revision_;
        trace_requested = trace_pending_;
        presentation_requested = presentation_pending_;
        inspection_requested = inspection_pending_;
        inspection_location = requested_inspection_;
        inspection_token = requested_inspection_token_;
        observer_requested = observer_pending_;
        observer = requested_observer_;
        map_point_requested = map_point_pending_;
        map_coordinate = requested_map_coordinate_;
        map_point_token = requested_map_point_token_;
        roam_requested = roam_pending_;
        roam_coordinate = requested_roam_coordinate_;
        roam_altitude_mode = requested_roam_altitude_mode_;
        roam_height = requested_roam_height_;
        roam_token = requested_roam_token_;
        current_ground_clearance = observer_ground_clearance_;
        lod_scale_requested = lod_scale_pending_;
        lod_scale = requested_lod_scale_;
        collision_settings_requested = collision_settings_pending_;
        bilinear_collisions = requested_bilinear_collisions_;
        c1_normals = requested_c1_normals_;
        raytracer = requested_raytracer_;
        target_requested = target_pending_ || trace_pending_ || presentation_pending_;
        target = requested_target_;
        target_token = requested_target_token_;
        trace_pending_ = false;
        presentation_pending_ = false;
        inspection_pending_ = false;
        observer_pending_ = false;
        map_point_pending_ = false;
        target_pending_ = false;
        roam_pending_ = false;
        lod_scale_pending_ = false;
        collision_settings_pending_ = false;
      }

      bool unpublished_frame = false;
      bool upscale_frame_started = false;
      diagnostics::Scope diagnostic_scope(diagnostics::worker);
      trace_activity::Binding diagnostic_binding(
          settings_.trace_diagnostics ? &diagnostics::terrain : nullptr
      );
      trace_activity::Scope terrain_activity("worker iteration");
      try {
        @autoreleasepool {
          diagnostics::worker.mark("prepare");
          const auto started = std::chrono::steady_clock::now();
          auto bvh_before =
              settings_.trace_diagnostics ? trace_->bvh_statistics() : MetalBvhStatistics{};
          auto tiles_before =
              settings_.trace_diagnostics ? trace_->tile_statistics() : TileManagerStatistics{};
          const ObserverLocation previous_observer = current_observer_;
          bool session_replaced = false;
          std::optional<RoamResult> roam_result;
          double next_ground_clearance = current_ground_clearance;
          if (roam_requested) {
            // Resolve terrain on the render worker through TileManager so
            // continuous input never blocks the AppKit event thread.
            const std::optional<float> ground =
                trace_->sample_terrain(roam_coordinate.easting, roam_coordinate.northing);
            const bool terrain_clear =
                ground.has_value() && (roam_altitude_mode == RoamAltitudeMode::FollowTerrain ||
                                       roam_height >= static_cast<double>(*ground) + 0.5);
            roam_result = RoamResult{
                .request_token = roam_token,
                .accepted = terrain_clear,
                .ground_elevation =
                    ground.has_value() ? *ground : std::numeric_limits<float>::quiet_NaN(),
            };
            if (terrain_clear) {
              observer = {
                  roam_coordinate.easting,
                  roam_coordinate.northing,
                  roam_altitude_mode == RoamAltitudeMode::FollowTerrain
                      ? static_cast<double>(*ground) + roam_height
                      : roam_height,
              };
              next_ground_clearance = observer.elevation - *ground;
              observer_requested = true;
              trace_requested = true;
              presentation_requested = true;
              target_requested = true;
            }
          }
          peak_labels_requested = peak_labels_requested || (peak_labels_enabled && trace_requested);
          GpuTerrainFrameTiming producer_timing;
          if (trace_requested) {
            MetalFxResolution metalfx_resolution =
                panorama::app::metalfx_resolution(image, metalfx_selection, metalfx_->supported());
            // Configure before creating the GPU request so failure traces at
            // native resolution rather than stretching a reduced ray image.
            if (metalfx_resolution.enabled && !metalfx_->configure(metalfx_resolution.trace, image))
              metalfx_resolution = {image, false};
            RayFieldRequest field = metalfx_ray_request(
                make_view(image, orientation, vertical_field_of_view),
                metalfx_resolution.trace
            );
            trace_->set_raytracer(raytracer);
            if (lod_scale_requested) {
              trace_->set_lod_scale(lod_scale);
            }
            if (collision_settings_requested) {
              trace_->set_collision_options(bilinear_collisions, c1_normals);
            }
            if (observer_requested) {
              if (!trace_->relocate_observer(observer)) {
                trace_activity::Scope activity("terrain session replacement");
                const RaytraceConfig config = {
                    settings_.tile_dir,
                    observer,
                    settings_.max_distance,
                    0U,
                    settings_.tile_cache_size_bytes,
                    settings_.workers,
                    !settings_.discard_quantized,
                    bilinear_collisions,
                    c1_normals,
                    false,
                    lod_scale,
                    raytracer,
                    settings_.bvh_block_cells,
                    settings_.bvh_cache_size_bytes,
                };
                auto replacement = std::make_unique<TerrainTraceSession>(
                    config,
                    field,
                    GpuTraceOutputRequirements{
                        .surface_gradients = true,
                        .elevations = true,
                        .debugging_info = true,
                    },
                    display_queue_
                );
                trace_ = std::move(replacement);
                session_replaced = true;
                bvh_before = {};
                tiles_before = {};
              }
              current_observer_ = observer;
            }
            current_field_ = std::move(field);
            current_output_image_ = image;
            current_metalfx_enabled_ = metalfx_resolution.enabled;
            presentation_->set_output_texture_usage(
                current_metalfx_enabled_ ? metalfx_->input_texture_usage() : MTLTextureUsageUnknown
            );
            current_orientation_ = orientation;
            current_vertical_field_of_view_ = vertical_field_of_view;
            current_revision_ = revision;
          }

          if (presentation_requested) {
            diagnostics::worker.mark("render");
            trace_activity::Scope activity("terrain frame");
            id<MTLBuffer> next_visibility_points = current_visibility_points_;
            // Select one unpublished output for the whole producer, including
            // any streaming repair pass. The callback can run more than once.
            if (current_metalfx_enabled_) {
              metalfx_->begin_frame();
              upscale_frame_started = true;
            }
            producer_timing = render_terrain_frame(
                *trace_,
                trace_requested ? &current_field_ : nullptr,
                *presentation_,
                presentation,
                [&](id<MTLCommandBuffer> command) {
                  if (current_metalfx_enabled_) {
                    metalfx_->encode(command, presentation_->texture());
                  }
                  // Recheck at encoding time: hiding may have arrived while
                  // the trace was preparing. The producer's repair callback can
                  // run again; only its final snapshot is published.
                  std::lock_guard<std::mutex> lock(mutex_);
                  if (trace_requested && minimap_enabled_ &&
                      minimap_generation == minimap_generation_)
                    next_visibility_points = visibility_->project(
                        trace_->ray_directions(),
                        trace_->distances(),
                        current_field_.image,
                        command
                    );
                }
            );
            unpublished_frame = true;
            current_visibility_points_ = next_visibility_points;
            current_revision_ = revision;
          }
          const double milliseconds =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                  .count();
          if (settings_.trace_diagnostics && presentation_requested) {
            diagnostics::worker.mark("log");
            const auto bvh = trace_->bvh_statistics();
            const auto tiles = trace_->tile_statistics();
            std::printf(
                "Frame %llu: wall %.3f ms, GPU producer %.3f ms, %u producer submission(s), %s; "
                "t=%.3f, BVH resident/budget %.1f/%.1f MiB, "
                "builds/hits/evictions=%llu/%llu/%llu, scene builds=%llu\n",
                static_cast<unsigned long long>(revision),
                milliseconds,
                producer_timing.gpu_milliseconds,
                producer_timing.producer_submissions,
                producer_timing.streamed ? "streaming work excluded from GPU producer timing"
                                         : "resident trace and presentation combined",
                diagnostics::seconds(),
                double(bvh.resident_bytes) / 1048576.0,
                double(bvh.budget_bytes) / 1048576.0,
                static_cast<unsigned long long>(bvh.builds),
                static_cast<unsigned long long>(bvh.cache_hits),
                static_cast<unsigned long long>(bvh.evictions),
                static_cast<unsigned long long>(bvh.scene_builds)
            );
            std::printf(
                "Frame phases %llu: pre-render %.3f ms, prepare %.3f ms, primary repair %.3f ms, "
                "shadow repair %.3f ms, producer wait %.3f ms; "
                "observer=%.2f/%.2f/%.2f, moved=%.2f m, roam=%d, replaced=%d, "
                "output=%ux%u, trace=%ux%u, MetalFX=%s, LOD=%.3f, shadows=%d, minimap=%d\n",
                static_cast<unsigned long long>(revision),
                milliseconds - producer_timing.wall_milliseconds,
                producer_timing.preparation_milliseconds,
                producer_timing.primary_repair_milliseconds,
                producer_timing.shadow_repair_milliseconds,
                producer_timing.producer_wait_milliseconds,
                current_observer_.easting,
                current_observer_.northing,
                current_observer_.elevation,
                std::hypot(
                    current_observer_.easting - previous_observer.easting,
                    current_observer_.northing - previous_observer.northing
                ),
                int(roam_requested),
                int(session_replaced),
                current_output_image_.width,
                current_output_image_.height,
                current_field_.image.width,
                current_field_.image.height,
                current_metalfx_enabled_ ? metalfx_preset_name(metalfx_selection.preset) : "Native",
                double(lod_scale),
                int(presentation.use_surface_normals && presentation.appearance.raytraced_shadows),
                int(minimap_enabled)
            );
            const auto delta = [](uint64_t after, uint64_t before) {
              return static_cast<unsigned long long>(after >= before ? after - before : after);
            };
            std::printf(
                "Frame work %llu: BVH built/hit/evicted=%llu/%llu/%llu, "
                "catalogue/instance/scene builds=%llu/%llu/%llu, "
                "cached/scene tiles=%llu/%llu, scene %.1f MiB, "
                "selection/detail passes=%llu/%llu, fallback ray attempts=%llu, "
                "GPU build/selection/detail=%.3f/%.3f/%.3f ms, CPU grouping=%.3f ms; "
                "atlas installed/evicted=%llu/%llu, I/O=%.3f MiB, resident=%u/%u\n",
                static_cast<unsigned long long>(revision),
                delta(bvh.builds, bvh_before.builds),
                delta(bvh.cache_hits, bvh_before.cache_hits),
                delta(bvh.evictions, bvh_before.evictions),
                delta(bvh.catalogue_builds, bvh_before.catalogue_builds),
                delta(bvh.instance_builds, bvh_before.instance_builds),
                delta(bvh.scene_builds, bvh_before.scene_builds),
                static_cast<unsigned long long>(bvh.cached_tiles),
                static_cast<unsigned long long>(bvh.scene_tiles),
                double(bvh.scene_bytes) / 1048576.0,
                delta(bvh.selection_passes, bvh_before.selection_passes),
                delta(bvh.trace_passes, bvh_before.trace_passes),
                delta(bvh.scene_fallback_rays, bvh_before.scene_fallback_rays),
                bvh.build_gpu_ms - bvh_before.build_gpu_ms,
                bvh.selection_gpu_ms - bvh_before.selection_gpu_ms,
                bvh.trace_gpu_ms - bvh_before.trace_gpu_ms,
                bvh.grouping_cpu_ms - bvh_before.grouping_cpu_ms,
                delta(tiles.installations, tiles_before.installations),
                delta(tiles.evictions, tiles_before.evictions),
                double(tiles.bytes_loaded_with_metal_io - tiles_before.bytes_loaded_with_metal_io) /
                    1048576.0,
                tiles.resident_tiles,
                tiles.slot_capacity
            );
            std::printf(
                "Frame shadows %llu: BVH caster builds=%llu, repair passes=%llu, "
                "repair GPU=%.3f ms, capacity fallbacks=%llu\n",
                static_cast<unsigned long long>(revision),
                delta(bvh.shadow_tiles_built, bvh_before.shadow_tiles_built),
                delta(bvh.shadow_passes, bvh_before.shadow_passes),
                bvh.shadow_gpu_ms - bvh_before.shadow_gpu_ms,
                delta(bvh.shadow_cache_fallbacks, bvh_before.shadow_cache_fallbacks)
            );
            std::fflush(stdout);
          }
          if (settings_.trace_diagnostics && trace_requested) {
            const auto camera = trace_->camera_statistics();
            std::printf(
                "Camera preparation: wall %.3f ms, GPU LOD %.3f ms, plans/footprints=%llu/%llu\n",
                camera.preparation_wall_ms,
                camera.preparation_gpu_ms,
                static_cast<unsigned long long>(camera.plan_updates),
                static_cast<unsigned long long>(camera.footprint_updates)
            );
          }
          diagnostics::worker.mark("inspection");
          const bool publish_inspection =
              inspection_requested || (inspection_location.has_value() && presentation_requested);
          std::optional<PointInspection> inspection;
          if (publish_inspection && inspection_location.has_value()) {
            inspection = inspect_location(*inspection_location);
          }
          std::optional<TerrainPoint> map_point;
          if (map_point_requested) {
            // Map inspection uses exact LOD-1 sampling even when the current
            // render selected a coarser terrain variant for this source.
            if (const std::optional<float> elevation =
                    trace_->sample_terrain(map_coordinate.easting, map_coordinate.northing)) {
              map_point = TerrainPoint{
                  map_coordinate.easting,
                  map_coordinate.northing,
                  *elevation,
              };
            }
          }
          std::optional<TargetVisibility> target_visibility;
          if (target_requested && target.has_value()) {
            target_visibility = TargetVisibility{
                .revision = current_revision_,
                .request_token = target_token,
                .occluded = target_is_occluded(
                    *target,
                    current_orientation_,
                    current_vertical_field_of_view_
                ),
            };
          }
          std::optional<PeakLabelFrame> peak_labels;
          if (peak_labels_enabled && peak_labels_requested)
            peak_labels = visible_peaks();

          diagnostics::worker.mark("publish");
          std::lock_guard<std::mutex> lock(mutex_);
          if (trace_requested)
            error_.clear();
          if (presentation_requested) {
            presented_texture_ =
                current_metalfx_enabled_ ? metalfx_->texture() : presentation_->texture();
            if (!minimap_enabled_ || minimap_generation != minimap_generation_)
              current_visibility_points_ = nil;
            presented_visibility_points_ = current_visibility_points_;
            presented_image_ = current_field_.image;
            presented_output_image_ = current_output_image_;
            presented_metalfx_enabled_ = current_metalfx_enabled_;
            presented_metalfx_preset_ = metalfx_selection.preset;
            presented_orientation_ = orientation;
            presented_vertical_field_of_view_ = vertical_field_of_view;
            presented_revision_ = revision;
            if (diagnostics::enabled)
              diagnostics::worker.revision = revision;
            presented_observer_ = current_observer_;
            unpublished_frame = false;
            upscale_frame_started = false;
            // The title reports camera-update throughput. A cheap appearance-only
            // pass should not replace it with a misleadingly high frame rate.
            if (trace_requested) {
              frame_ms_ = milliseconds;
              frame_gpu_ms_ = producer_timing.gpu_milliseconds;
              frame_streamed_ = producer_timing.streamed;
            }
          }
          if (publish_inspection) {
            presented_inspection_ = inspection;
            presented_inspection_sequence_++;
            presented_inspection_token_ = inspection_token;
          }
          if (map_point_requested) {
            presented_map_point_ = map_point;
            presented_map_point_sequence_++;
            presented_map_point_token_ = map_point_token;
          }
          if (target_requested) {
            presented_target_visibility_ = target_visibility;
            presented_target_visibility_sequence_++;
          }
          if (peak_labels_requested && peak_labels_generation == peak_labels_generation_)
            presented_peak_labels_ = peak_labels_enabled_ ? std::move(peak_labels) : std::nullopt;
          if (roam_requested) {
            presented_roam_result_ = roam_result;
            presented_roam_result_sequence_++;
            if (roam_result->accepted) {
              observer_ground_clearance_ = next_ground_clearance;
              // Once the latest roam request settles, make it the base for
              // subsequent discrete height changes.  Do not overwrite a newer
              // destination which arrived while this frame was tracing.
              if (requested_roam_token_ == roam_token && !observer_pending_) {
                requested_observer_ = current_observer_;
              }
            }
          }
          diagnostics::worker.mark("pool-drain");
        }
      } catch (const std::exception &exception) {
        if (unpublished_frame) {
          presentation_->cancel_frame();
        }
        if (upscale_frame_started)
          metalfx_->cancel_frame();
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = exception.what();
        printf("ERROR: %s\n", error_.c_str());
      }
    }
  }

  ViewerSettings settings_;
  std::unique_ptr<TerrainTraceSession> trace_;
  std::unique_ptr<GpuImageRenderer> presentation_;
  std::unique_ptr<MetalFxUpscaler> metalfx_;
  std::unique_ptr<FullscreenPresentation> fullscreen_presentation_;
  std::unique_ptr<GpuVisibilityPointProjector> visibility_;
  id<MTLDevice> device_;
  id<MTLCommandQueue> display_queue_;
  id<MTLLibrary> library_;
  std::unique_ptr<diagnostics::Monitor> diagnostics_;
  RayFieldRequest current_field_;
  ObserverLocation current_observer_ = {};
  CameraOrientation current_orientation_ = {};
  double current_vertical_field_of_view_ = 0.0;
  uint64_t current_revision_ = 0U;
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  CameraOrientation requested_orientation_ = {};
  double requested_vertical_field_of_view_ = 0.0;
  ImageSize requested_image_ = {};
  MetalFxSelection requested_metalfx_ = {};
  TerrainPresentationSettings requested_presentation_ = {};
  std::optional<InspectionLocation> requested_inspection_;
  MapCoordinate requested_map_coordinate_ = {};
  MapCoordinate requested_roam_coordinate_ = {};
  RoamAltitudeMode requested_roam_altitude_mode_ = RoamAltitudeMode::FollowTerrain;
  double requested_roam_height_ = 0.0;
  ObserverLocation requested_observer_ = {};
  std::optional<TerrainPoint> requested_target_;
  CameraOrientation presented_orientation_ = {};
  double presented_vertical_field_of_view_ = 0.0;
  ImageSize presented_image_ = {};
  ImageSize current_output_image_ = {};
  ImageSize presented_output_image_ = {};
  bool current_metalfx_enabled_ = false;
  bool presented_metalfx_enabled_ = false;
  MetalFxPreset presented_metalfx_preset_ = MetalFxPreset::Off;
  std::optional<PointInspection> presented_inspection_;
  std::optional<TerrainPoint> presented_map_point_;
  ObserverLocation presented_observer_ = {};
  std::optional<TargetVisibility> presented_target_visibility_;
  std::optional<PeakLabelFrame> presented_peak_labels_;
  std::optional<RoamResult> presented_roam_result_;
  id<MTLTexture> presented_texture_;
  id<MTLBuffer> current_visibility_points_;
  id<MTLBuffer> presented_visibility_points_;
  uint64_t requested_revision_ = 0U;
  uint64_t requested_inspection_token_ = 0U;
  uint64_t presented_revision_ = 0U;
  uint64_t presented_inspection_sequence_ = 0U;
  uint64_t presented_inspection_token_ = 0U;
  uint64_t requested_map_point_token_ = 0U;
  uint64_t presented_map_point_sequence_ = 0U;
  uint64_t presented_map_point_token_ = 0U;
  uint64_t requested_target_token_ = 0U;
  uint64_t presented_target_visibility_sequence_ = 0U;
  uint64_t requested_roam_token_ = 0U;
  uint64_t presented_roam_result_sequence_ = 0U;
  double observer_ground_clearance_ = 0.0;
  double frame_ms_ = 0.0;
  double frame_gpu_ms_ = 0.0;
  bool frame_streamed_ = false;
  float requested_lod_scale_ = 0.0F;
  Raytracer requested_raytracer_ = Raytracer::MetalBvh;
  bool requested_bilinear_collisions_ = false;
  bool requested_c1_normals_ = false;
  std::string error_;
  bool minimap_enabled_ = false;
  bool minimap_changed_ = false;
  bool peak_labels_enabled_ = false;
  bool peak_labels_changed_ = false;
  uint64_t peak_labels_generation_ = 0U;
  std::optional<PeakCatalogue> peak_catalogue_;
  uint64_t minimap_generation_ = 0;
  bool trace_pending_ = false;
  bool presentation_pending_ = false;
  bool inspection_pending_ = false;
  bool observer_pending_ = false;
  bool map_point_pending_ = false;
  bool target_pending_ = false;
  bool roam_pending_ = false;
  bool lod_scale_pending_ = false;
  bool collision_settings_pending_ = false;
  bool observer_fallback_used_ = false;
  bool stopping_ = false;
};

std::unique_ptr<ViewerRenderer> make_viewer_renderer(ViewerSettings settings) {
  return std::make_unique<ViewerRendererImpl>(std::move(settings));
}

} // namespace panorama::app
