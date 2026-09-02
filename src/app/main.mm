#include "arguments.h"
#include "coordinate_input.h"
#include "gpu_image_renderer.h"
#include "loaded_tile.h"
#include "metal_tile.h"
#include "minimap.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "solar_position.h"
#include "synthetic_render_options.h"
#include "terrain_catalogue.h"
#include "terrain_presentation_settings.h"
#include "terrain_trace_session.h"
#include "timer.h"

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

constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
constexpr double kDegreesToRadians = std::numbers::pi / 180.0;
constexpr double kRadiansToDegrees = 180.0 / std::numbers::pi;
constexpr double kDefaultVerticalFieldOfView = 70.0 * kDegreesToRadians;
constexpr double kDefaultSunAzimuthDegrees = 225.0;
constexpr double kDefaultSunAltitudeDegrees = 35.0;
constexpr double kMinimumMovementSpeed = 1.0;
constexpr double kMaximumRoamSpeed = 200.0;
constexpr double kMaximumCruiseSpeed = 10'000.0;
constexpr double kDefaultRoamSpeed = 20.0;
// 50 m/s is 180 km/h: representative of a light training aircraft while
// remaining manageable when Cruise first resumes.
constexpr double kDefaultCruiseSpeed = 50.0;
constexpr double kCruiseSteeringDeadZone = 0.06;
constexpr double kCruiseSteeringExponent = 1.5;
constexpr double kCruiseMaximumYawRate = 90.0 * kDegreesToRadians;
constexpr double kCruiseMaximumPitchRate = 60.0 * kDegreesToRadians;
constexpr double kAircraftMaximumBank = 55.0 * kDegreesToRadians;
constexpr double kAircraftBankResponseSeconds = 0.5;
constexpr double kAircraftTrimResponseSeconds = 8.0;
constexpr double kGravity = 9.80665;
constexpr float kDefaultSkyStrength = 0.28F;
constexpr float kDefaultSkyDetail = 0.65F;
constexpr float kDefaultDiffusivity = 1.0F;
constexpr double kFallbackEyeHeight = 2.0;

struct ViewerSettings {
  std::filesystem::path tile_dir = "data/swissalti3d-10-level-0";
  uint64_t tile_cache_size_bytes = 128ULL * kBytesPerMiB;
  uint32_t workers = 8U;
  float max_distance = 600'000.0F;
  float lod_scale = 0.0F;
  bool discard_quantized = false;
  ObserverLocation observer = {2623452.4, 1100502.2, 3415.0};
  ImageSize image = {1600U, 900U};
  double vertical_field_of_view = kDefaultVerticalFieldOfView;
  CameraOrientation orientation = {0.0, 0.0, 0.0};
  TerrainPresentationSettings presentation = {
      .appearance =
          {
              .sun_azimuth = kDefaultSunAzimuthDegrees * kDegreesToRadians,
              .sun_elevation = kDefaultSunAltitudeDegrees * kDegreesToRadians,
              .ambient_light = kDefaultSkyStrength,
              .ambient_detail = kDefaultSkyDetail,
              .diffusivity = kDefaultDiffusivity,
              .colour_source = TerrainColourSource::Distance,
              .colourmap = PresetColourmap::Viewfinder,
          },
      .colour_range = {0.0F, 100'000.0F},
      .use_surface_normals = true,
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
      "  --tile-dir DIR        prepared level-0 tile directory\n"
      "  --tile-cache-mib N    resident terrain-cache budget (default: 128)\n"
      "  --workers N           tile preparation workers (default: 8)\n"
      "  --max-distance M      horizontal range in metres (default: 600000)\n"
      "  --lod-scale V         terrain cell footprint multiplier; 0 keeps full detail\n"
      "                        (default: 0)\n"
      "  --discard-quantized   expand uint16 terrain to Float32 in the GPU atlas\n"
      "                        (default: retain uint16)\n"
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
  /// Map samples have no source pixel, slope, or aspect value.
  bool map_selected;
};

/// Projected horizontal coordinate awaiting terrain-elevation sampling.
struct MapCoordinate {
  double easting;
  double northing;
};

/// Minimal world-space terrain sample shared by map movement and visibility.
struct TerrainPoint {
  double easting;
  double northing;
  float elevation;
};

/// Operation to complete when the latest asynchronous map sample arrives.
enum class MapPointAction : uint8_t {
  None,
  Hover,
  Look,
  MoveObserver,
};

enum class RoamKey : uint8_t {
  Forward,
  Backward,
  Left,
  Right,
};

enum class RoamAltitudeMode : uint8_t {
  FollowTerrain,
  HoldAltitude,
};

struct RoamResult {
  uint64_t request_token;
  bool accepted;
  float ground_elevation;
};

/// Overlay region currently responsible for point-inspection hover state.
enum class PointerOwner : uint8_t {
  None,
  Panorama,
  Minimap,
  Overlay,
};

/// Occlusion result tied to both a rendered revision and request generation.
struct TargetVisibility {
  uint64_t revision;
  uint64_t request_token;
  bool occluded;
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

/// One mutex-consistent snapshot consumed by the main-thread Metal view.
struct PresentedFrame {
  id<MTLTexture> texture;
  id<MTLBuffer> visibility_points;
  ImageSize image;
  CameraOrientation orientation;
  double vertical_field_of_view;
  uint64_t revision;
  double milliseconds;
  std::string error;
  std::optional<PointInspection> inspection;
  uint64_t inspection_sequence;
  uint64_t inspection_request_token;
  ObserverLocation observer;
  std::optional<TerrainPoint> map_point;
  uint64_t map_point_sequence;
  uint64_t map_point_request_token;
  std::optional<TargetVisibility> target_visibility;
  uint64_t target_visibility_sequence;
  std::optional<RoamResult> roam_result;
  uint64_t roam_result_sequence;
};

/// Decode one IEEE float16 value emitted by Metal without depending on a SIMD
/// vector ABI shared between C++ and Metal.
[[nodiscard]] float float_from_half_bits(uint16_t bits) {
  _Float16 value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<float>(value);
}

/// GPU-only conversion from reusable trace buffers to an immutable buffer of
/// observer-relative east/north collision points. Publishing a fresh buffer
/// per trace prevents continuous camera input from exposing the minimap to
/// ray storage which the next trace is already clearing or replacing.
class GpuVisibilityPointProjector {
public:
  GpuVisibilityPointProjector(
      id<MTLDevice> device,
      id<MTLCommandQueue> queue,
      id<MTLLibrary> library
  )
      : device_(device), queue_(queue) {
    if (device_ == nil || queue_ == nil || library == nil) {
      throw std::invalid_argument("Visibility projection requires valid Metal resources");
    }
    id<MTLFunction> function = [library newFunctionWithName:@"visibility_collision_points"];
    if (function == nil) {
      throw std::runtime_error("Visibility collision-point kernel is missing");
    }
    NSError *error = nil;
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (pipeline_ == nil) {
      const char *detail = error == nil ? "unknown error" : error.localizedDescription.UTF8String;
      throw std::runtime_error(
          "Could not create visibility collision-point pipeline: " + std::string(detail)
      );
    }
  }

  [[nodiscard]] id<MTLBuffer>
  project(id<MTLBuffer> rays, id<MTLBuffer> distances, ImageSize image) const {
    const uint64_t count64 = static_cast<uint64_t>(image.width) * image.height;
    const uint64_t rayBytes = count64 * sizeof(RayDirection);
    const uint64_t distanceBytes = count64 * sizeof(float);
    if (count64 == 0U || count64 > std::numeric_limits<uint32_t>::max() || rays == nil ||
        distances == nil || rays.length < rayBytes || distances.length < distanceBytes) {
      throw std::invalid_argument("Visibility projection requires valid trace buffers");
    }
    const uint32_t count = static_cast<uint32_t>(count64);
    const NSUInteger length = static_cast<NSUInteger>(count64 * 2U * sizeof(float));
    id<MTLBuffer> points = [device_ newBufferWithLength:length
                                                options:MTLResourceStorageModePrivate];
    if (points == nil) {
      throw std::runtime_error("Could not allocate visibility collision-point buffer");
    }
    points.label = @"Minimap visibility collision points";

    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("Could not create visibility collision-point command");
    }
    command.label = @"Project minimap visibility collisions";
    encoder.label = @"visibility_collision_points";
    [encoder setComputePipelineState:pipeline_];
    [encoder setBuffer:rays offset:0 atIndex:0];
    [encoder setBuffer:distances offset:0 atIndex:1];
    [encoder setBuffer:points offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    const NSUInteger groupWidth =
        std::min<NSUInteger>(256U, pipeline_.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1U, 1U)
        threadsPerThreadgroup:MTLSizeMake(groupWidth, 1U, 1U)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      const char *detail =
          command.error == nil ? "unknown error" : command.error.localizedDescription.UTF8String;
      throw std::runtime_error(
          "GPU visibility collision-point projection failed: " + std::string(detail)
      );
    }
    return points;
  }

private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> pipeline_;
};

/// Small, one-tile CPU cache used for elevation-correct minimap interaction.
/// Custom terrain stays on its normal compressed Metal-I/O path; only the
/// tile under the pointer is copied into shared memory for bilinear sampling.
class TerrainPointSampler {
public:
  TerrainPointSampler(
      id<MTLDevice> device,
      const std::filesystem::path &tile_dir,
      ObserverLocation observer,
      float max_distance
  )
      : device_(device), tile_dir_(tile_dir), max_distance_(max_distance),
        catalogue_(TerrainCatalogue::discover(tile_dir, observer, max_distance, 0U)),
        io_queue_(make_metal_io_queue(device)) {}

  void recenter(ObserverLocation observer) {
    catalogue_ = TerrainCatalogue::discover(tile_dir_, observer, max_distance_, 0U);
    cached_vertices_.clear();
    cached_key_.reset();
  }

  [[nodiscard]] std::optional<TerrainPoint> sample(MapCoordinate coordinate) {
    const TileKey key = tile_key_at(catalogue_.grid(), coordinate.easting, coordinate.northing);
    const std::optional<uint32_t> source_index = catalogue_.find_source(key);
    if (!source_index.has_value()) {
      return std::nullopt;
    }
    if (!cached_key_.has_value() || !(*cached_key_ == key)) {
      load(catalogue_.sources()[*source_index]);
      cached_key_ = key;
    }

    const double x = (coordinate.easting - cached_lower_left_x_) / cached_cell_size_;
    const double y = (coordinate.northing - cached_lower_left_y_) / cached_cell_size_;
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 || y < 0.0 || x > cached_cell_count_ ||
        y > cached_cell_count_) {
      return std::nullopt;
    }
    const uint32_t x0 = std::min(cached_cell_count_ - 1U, static_cast<uint32_t>(std::floor(x)));
    const uint32_t y0 = std::min(cached_cell_count_ - 1U, static_cast<uint32_t>(std::floor(y)));
    const double tx = std::clamp(x - x0, 0.0, 1.0);
    const double ty = std::clamp(y - y0, 0.0, 1.0);
    const size_t side = static_cast<size_t>(cached_cell_count_) + 1U;
    const auto vertex = [&](uint32_t column, uint32_t row) {
      return static_cast<double>(cached_vertices_[static_cast<size_t>(row) * side + column]);
    };
    const double south = std::lerp(vertex(x0, y0), vertex(x0 + 1U, y0), tx);
    const double north = std::lerp(vertex(x0, y0 + 1U), vertex(x0 + 1U, y0 + 1U), tx);
    const double elevation = std::lerp(south, north, ty);
    return TerrainPoint{coordinate.easting, coordinate.northing, static_cast<float>(elevation)};
  }

private:
  void load(const TerrainSource &source) {
    const MetalTileHeader header = read_metal_tile_header(source.path);
    if (header.vertex_byte_count > std::numeric_limits<NSUInteger>::max()) {
      throw std::overflow_error("Terrain point tile is too large for Metal");
    }
    id<MTLBuffer> payload =
        [device_ newBufferWithLength:static_cast<NSUInteger>(header.vertex_byte_count)
                             options:MTLResourceStorageModeShared];
    if (payload == nil || payload.contents == nullptr) {
      throw std::runtime_error("Could not allocate terrain point sample buffer");
    }
    const MetalTileBufferLoad load = {
        source.path,
        0U,
        nil,
        header.vertex_offset,
        header.vertex_byte_count,
    };
    load_metal_tiles_into_buffer(
        device_,
        io_queue_,
        std::span<const MetalTileBufferLoad>(&load, 1U),
        payload,
        payload.length
    );
    const size_t count = (static_cast<size_t>(header.cell_count) + 1U) *
                         (static_cast<size_t>(header.cell_count) + 1U);
    cached_vertices_.resize(count);
    if (header.sample_type == MetalTileSampleType::Float32) {
      const auto *source_values = static_cast<const float *>(payload.contents);
      std::copy_n(source_values, count, cached_vertices_.begin());
    } else {
      const auto *source_values = static_cast<const uint16_t *>(payload.contents);
      const float base = static_cast<float>(header.elevation_base_decimeters) / 10.0F;
      for (size_t index = 0U; index < count; index++) {
        cached_vertices_[index] = base + static_cast<float>(source_values[index]) / 10.0F;
      }
    }
    cached_cell_count_ = header.cell_count;
    cached_cell_size_ = header.cell_size;
    cached_lower_left_x_ = header.lower_left_x;
    cached_lower_left_y_ = header.lower_left_y;
  }

  id<MTLDevice> device_;
  std::filesystem::path tile_dir_;
  float max_distance_;
  TerrainCatalogue catalogue_;
  id<MTLIOCommandQueue> io_queue_;
  std::optional<TileKey> cached_key_;
  std::vector<float> cached_vertices_;
  uint32_t cached_cell_count_ = 0U;
  double cached_cell_size_ = 0.0;
  double cached_lower_left_x_ = 0.0;
  double cached_lower_left_y_ = 0.0;
};

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
    const auto traceConfig = [&](ObserverLocation observer, bool allowFallback) {
      return RaytraceConfig{
          settings_.tile_dir,
          observer,
          settings_.max_distance,
          0U,
          settings_.tile_cache_size_bytes,
          settings_.workers,
          !settings_.discard_quantized,
          allowFallback,
          settings_.lod_scale,
      };
    };
    const ObserverLocation requestedObserver = settings_.observer;
    trace_ = std::make_unique<TerrainTraceSession>(
        traceConfig(requestedObserver, true),
        initial_field,
        GpuTraceOutputRequirements{.elevations = true, .surface_gradients = true}
    );
    settings_.observer = trace_->observer();
    observer_fallback_used_ = settings_.observer.easting != requestedObserver.easting ||
                              settings_.observer.northing != requestedObserver.northing;
    device_ = trace_->device();
    sampler_ = std::make_unique<TerrainPointSampler>(
        device_,
        settings_.tile_dir,
        settings_.observer,
        settings_.max_distance
    );
    if (const std::optional<TerrainPoint> ground =
            sampler_->sample({settings_.observer.easting, settings_.observer.northing})) {
      if (observer_fallback_used_) {
        settings_.observer.elevation = static_cast<double>(ground->elevation) + kFallbackEyeHeight;
        trace_ = std::make_unique<TerrainTraceSession>(
            traceConfig(settings_.observer, false),
            initial_field,
            GpuTraceOutputRequirements{.elevations = true, .surface_gradients = true}
        );
        if (trace_->device() != device_) {
          throw std::runtime_error("Observer fallback selected a different Metal device");
        }
        observer_ground_clearance_ = kFallbackEyeHeight;
      } else {
        observer_ground_clearance_ =
            std::max(0.0, settings_.observer.elevation - ground->elevation);
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
    display_queue_ = [device_ newCommandQueue];
    library_ = trace_->library();
    if (display_queue_ == nil) {
      throw std::runtime_error("Could not create viewer display command queue");
    }
    presentation_ = std::make_unique<GpuImageRenderer>(
        device_,
        display_queue_,
        library_,
        settings_.image,
        GpuPresentationRequirements{
            .scalar_diagnostics = false,
            .normal_diagnostics = false,
            .white_synthetic = true,
            .synthetic_scalar_colour = true,
            .host_readback = false,
        },
        MTLPixelFormatBGRA8Unorm
    );
    visibility_ = std::make_unique<GpuVisibilityPointProjector>(device_, display_queue_, library_);
    current_field_ = std::move(initial_field);
    current_observer_ = settings_.observer;
    current_orientation_ = settings_.orientation;
    current_vertical_field_of_view_ = settings_.vertical_field_of_view;
    requested_observer_ = settings_.observer;
    requested_lod_scale_ = settings_.lod_scale;
    presented_observer_ = settings_.observer;
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

  /// Re-trace with a new per-tile terrain LOD policy. The render worker owns
  /// the session, so the UI only publishes the newest scale here.
  void request_lod_scale(float lodScale) {
    if (!std::isfinite(lodScale) || lodScale < 0.0F) {
      throw std::invalid_argument("Terrain LOD scale must be finite and nonnegative");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_lod_scale_ = lodScale;
      requested_revision_++;
      lod_scale_pending_ = true;
      trace_pending_ = true;
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

  /// Sample the actual terrain under a minimap coordinate on the render worker.
  uint64_t request_map_point(MapCoordinate coordinate) {
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
  uint64_t request_target_visibility(std::optional<TerrainPoint> point) {
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
  void request_observer_at(TerrainPoint point, double groundClearance) {
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
  void request_ground_clearance(double groundClearance) {
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
  uint64_t request_roam(MapCoordinate coordinate, RoamAltitudeMode altitudeMode, double height) {
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

  [[nodiscard]] PresentedFrame presented_frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        .texture = presented_texture_,
        .visibility_points = presented_visibility_points_,
        .image = presented_image_,
        .orientation = presented_orientation_,
        .vertical_field_of_view = presented_vertical_field_of_view_,
        .revision = presented_revision_,
        .milliseconds = frame_ms_,
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
        .roam_result = presented_roam_result_,
        .roam_result_sequence = presented_roam_result_sequence_,
    };
  }

  [[nodiscard]] id<MTLDevice> device() const { return device_; }
  [[nodiscard]] id<MTLCommandQueue> command_queue() const { return display_queue_; }
  [[nodiscard]] id<MTLLibrary> library() const { return library_; }
  [[nodiscard]] ImageSize initial_image() const { return settings_.image; }
  [[nodiscard]] ObserverLocation observer() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return presented_observer_;
  }
  [[nodiscard]] Crs terrain_crs() const { return trace_->crs(); }
  [[nodiscard]] const TerrainCoverage &terrain_coverage() const {
    return trace_->terrain_coverage();
  }
  [[nodiscard]] bool observer_used_fallback() const { return observer_fallback_used_; }
  [[nodiscard]] double ground_clearance() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return observer_ground_clearance_;
  }
  [[nodiscard]] double initial_vertical_field_of_view() const {
    return settings_.vertical_field_of_view;
  }
  [[nodiscard]] float max_distance() const { return settings_.max_distance; }
  [[nodiscard]] float initial_lod_scale() const { return settings_.lod_scale; }
  [[nodiscard]] CameraOrientation initial_orientation() const { return settings_.orientation; }
  [[nodiscard]] TerrainPresentationSettings initial_presentation() const {
    return settings_.presentation;
  }

private:
  [[nodiscard]] bool target_is_occluded(
      TerrainPoint point,
      CameraOrientation orientation,
      double vertical_field_of_view
  ) const {
    const LockedPointProjection projection = project_locked_point(
        point,
        current_observer_,
        current_field_.image,
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
    const uint32_t x = std::min(
        current_field_.image.width - 1U,
        static_cast<uint32_t>(std::llround(projection.pixel_x))
    );
    const uint32_t y = std::min(
        current_field_.image.height - 1U,
        static_cast<uint32_t>(std::llround(projection.pixel_y))
    );
    const size_t index =
        static_cast<size_t>(y) * static_cast<size_t>(current_field_.image.width) + x;
    const float collision_distance = distances[index];
    if (!(collision_distance > 0.0F) || !std::isfinite(collision_distance)) {
      return false;
    }
    return static_cast<double>(collision_distance) + tolerance < target_distance;
  }

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

    PointInspection result = {
        .pixel = pixel,
        .revision = revision,
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
      TerrainPresentationSettings presentation = {};
      uint64_t revision = 0U;
      bool trace_requested = false;
      bool presentation_requested = false;
      bool inspection_requested = false;
      bool observer_requested = false;
      bool map_point_requested = false;
      bool target_requested = false;
      bool roam_requested = false;
      bool lod_scale_requested = false;
      float lod_scale = 0.0F;
      std::optional<InspectionPixel> inspection_pixel;
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
                 observer_pending_ || map_point_pending_ || target_pending_ || roam_pending_;
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
      }

      try {
        @autoreleasepool {
          const auto started = std::chrono::steady_clock::now();
          std::optional<RoamResult> roam_result;
          double next_ground_clearance = current_ground_clearance;
          if (roam_requested) {
            const std::optional<TerrainPoint> ground = sampler_->sample(roam_coordinate);
            const bool terrain_clear =
                ground.has_value() && (roam_altitude_mode == RoamAltitudeMode::FollowTerrain ||
                                       roam_height >= static_cast<double>(ground->elevation) + 0.5);
            roam_result = RoamResult{
                .request_token = roam_token,
                .accepted = terrain_clear,
                .ground_elevation = ground.has_value() ? ground->elevation
                                                       : std::numeric_limits<float>::quiet_NaN(),
            };
            if (terrain_clear) {
              observer = {
                  roam_coordinate.easting,
                  roam_coordinate.northing,
                  roam_altitude_mode == RoamAltitudeMode::FollowTerrain
                      ? static_cast<double>(ground->elevation) + roam_height
                      : roam_height,
              };
              next_ground_clearance = observer.elevation - ground->elevation;
              observer_requested = true;
              trace_requested = true;
              presentation_requested = true;
              target_requested = true;
            }
          }
          if (trace_requested) {
            RayField field = make_view(image, orientation, vertical_field_of_view);
            if (lod_scale_requested) {
              trace_->set_lod_scale(lod_scale);
            }
            if (observer_requested) {
              if (!trace_->relocate_observer(observer)) {
                const RaytraceConfig config = {
                    settings_.tile_dir,
                    observer,
                    settings_.max_distance,
                    0U,
                    settings_.tile_cache_size_bytes,
                    settings_.workers,
                    !settings_.discard_quantized,
                    false,
                    lod_scale,
                };
                auto replacement = std::make_unique<TerrainTraceSession>(
                    config,
                    field,
                    GpuTraceOutputRequirements{.elevations = true, .surface_gradients = true}
                );
                if (replacement->device() != device_) {
                  throw std::runtime_error("Observer relocation selected a different Metal device");
                }
                trace_ = std::move(replacement);
                sampler_->recenter(observer);
              }
              current_observer_ = observer;
            }
            trace_->trace(field);
            current_field_ = std::move(field);
            current_orientation_ = orientation;
            current_vertical_field_of_view_ = vertical_field_of_view;
            current_revision_ = revision;
            current_visibility_points_ = visibility_->project(
                trace_->ray_directions(),
                trace_->distances(),
                current_field_.image
            );
          }

          if (presentation_requested) {
            Timer timer("GPU presentation");
            presentation_->resize(current_field_.image);
            if (presentation.use_surface_normals && presentation.appearance.raytraced_shadows) {
              trace_->trace_shadows(
                  presentation.appearance.sun_azimuth,
                  presentation.appearance.sun_elevation
              );
            }
            const id<MTLBuffer> colour_values =
                presentation.appearance.colour_source == TerrainColourSource::Elevation
                    ? trace_->elevations()
                    : trace_->distances();
            presentation_->render_synthetic(
                trace_->surface_gradients(),
                trace_->distances(),
                trace_->ray_directions(),
                colour_values,
                presentation.use_surface_normals && presentation.appearance.raytraced_shadows
                    ? trace_->shadow_visibility()
                    : nil,
                presentation.appearance,
                presentation.colour_range,
                presentation.use_surface_normals,
                timer
            );
            current_revision_ = revision;
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
          std::optional<TerrainPoint> map_point;
          if (map_point_requested) {
            map_point = sampler_->sample(map_coordinate);
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

          std::lock_guard<std::mutex> lock(mutex_);
          if (presentation_requested) {
            presented_texture_ = presentation_->texture();
            presented_visibility_points_ = current_visibility_points_;
            presented_image_ = current_field_.image;
            presented_orientation_ = orientation;
            presented_vertical_field_of_view_ = vertical_field_of_view;
            presented_revision_ = revision;
            presented_observer_ = current_observer_;
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
          if (map_point_requested) {
            presented_map_point_ = map_point;
            presented_map_point_sequence_++;
            presented_map_point_token_ = map_point_token;
          }
          if (target_requested) {
            presented_target_visibility_ = target_visibility;
            presented_target_visibility_sequence_++;
          }
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
  std::unique_ptr<GpuVisibilityPointProjector> visibility_;
  std::unique_ptr<TerrainPointSampler> sampler_;
  id<MTLDevice> device_;
  id<MTLCommandQueue> display_queue_;
  id<MTLLibrary> library_;
  RayField current_field_;
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
  TerrainPresentationSettings requested_presentation_ = {};
  std::optional<InspectionPixel> requested_inspection_;
  MapCoordinate requested_map_coordinate_ = {};
  MapCoordinate requested_roam_coordinate_ = {};
  RoamAltitudeMode requested_roam_altitude_mode_ = RoamAltitudeMode::FollowTerrain;
  double requested_roam_height_ = 0.0;
  ObserverLocation requested_observer_ = {};
  std::optional<TerrainPoint> requested_target_;
  CameraOrientation presented_orientation_ = {};
  double presented_vertical_field_of_view_ = 0.0;
  ImageSize presented_image_ = {};
  std::optional<PointInspection> presented_inspection_;
  std::optional<TerrainPoint> presented_map_point_;
  ObserverLocation presented_observer_ = {};
  std::optional<TargetVisibility> presented_target_visibility_;
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
  float requested_lod_scale_ = 0.0F;
  std::string error_;
  bool trace_pending_ = false;
  bool presentation_pending_ = false;
  bool inspection_pending_ = false;
  bool observer_pending_ = false;
  bool map_point_pending_ = false;
  bool target_pending_ = false;
  bool roam_pending_ = false;
  bool lod_scale_pending_ = false;
  bool observer_fallback_used_ = false;
  bool stopping_ = false;
};

} // namespace
} // namespace panorama::app

/// Auto-detection may yield several syntactically valid grids. Prefer an
/// interpretation which lands on a prepared tile, while leaving genuine ties
/// for the user to resolve explicitly in the coordinate-system selector.
[[nodiscard]] static bool coordinate_has_terrain_coverage(
    const panorama::TerrainCoverage &coverage,
    panorama::Coord coordinate
) {
  try {
    const panorama::TileKey key = panorama::tile_key_at(coverage.grid, coordinate.x, coordinate.y);
    return std::binary_search(coverage.tiles.begin(), coverage.tiles.end(), key);
  } catch (const std::out_of_range &) {
    return false;
  }
}

/// Keep the compact point footer readable without sacrificing useful precision
/// for nearby terrain samples.
[[nodiscard]] static NSString *format_point_distance(double metres) {
  const double magnitude = std::abs(metres);
  if (magnitude < 100.0) {
    return [NSString stringWithFormat:@"%.1f m", metres];
  }
  if (magnitude < 1'000.0) {
    return [NSString stringWithFormat:@"%.0f m", metres];
  }
  const double kilometres = metres / 1'000.0;
  if (magnitude < 10'000.0) {
    return [NSString stringWithFormat:@"%.2f km", kilometres];
  }
  if (magnitude < 100'000.0) {
    return [NSString stringWithFormat:@"%.1f km", kilometres];
  }
  return [NSString stringWithFormat:@"%.0f km", kilometres];
}

[[nodiscard]] static NSString *format_movement_speed(double metres_per_second) {
  const double kilometres_per_hour = metres_per_second * 3.6;
  return std::abs(kilometres_per_hour) < 100.0
             ? [NSString stringWithFormat:@"%.1f km/h", kilometres_per_hour]
             : [NSString stringWithFormat:@"%.0f km/h", kilometres_per_hour];
}

[[nodiscard]] static double cruise_steering_response(double displacement) {
  const double magnitude = std::abs(displacement);
  if (magnitude <= panorama::app::kCruiseSteeringDeadZone) {
    return 0.0;
  }
  const double normalised = std::clamp(
      (magnitude - panorama::app::kCruiseSteeringDeadZone) /
          (1.0 - panorama::app::kCruiseSteeringDeadZone),
      0.0,
      1.0
  );
  return std::copysign(std::pow(normalised, panorama::app::kCruiseSteeringExponent), displacement);
}

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

/// A pause badge must never take pointer ownership away from the panorama.
@interface ViewerPauseIndicatorView : NSStackView
@end

@implementation ViewerPauseIndicatorView
- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}
@end

static void stroke_hud_path(NSBezierPath *path, CGFloat foregroundWidth) {
  path.lineCapStyle = NSLineCapStyleRound;
  path.lineJoinStyle = NSLineJoinStyleRound;
  [NSColor.blackColor setStroke];
  path.lineWidth = foregroundWidth + 2.0;
  [path stroke];
  [NSColor.whiteColor setStroke];
  path.lineWidth = foregroundWidth;
  [path stroke];
}

[[nodiscard]] static NSString *heading_label(int unwrappedDegrees) {
  const int degrees = (unwrappedDegrees % 360 + 360) % 360;
  switch (degrees) {
  case 0:
    return @"N";
  case 90:
    return @"E";
  case 180:
    return @"S";
  case 270:
    return @"W";
  default:
    return [NSString stringWithFormat:@"%03d", degrees];
  }
}

/// Pointer-transparent attitude HUD centred on Cruise's neutral steering point.
@interface CruiseHUDView : NSView {
@private
  double _heading;
  double _pitch;
  double _bank;
  double _verticalFieldOfView;
  bool _aircraftMode;
}
- (void)setHeading:(double)heading
                  pitch:(double)pitch
                   bank:(double)bank
    verticalFieldOfView:(double)verticalFieldOfView
           aircraftMode:(bool)aircraftMode;
@end

@implementation CruiseHUDView
- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}

- (void)setHeading:(double)heading
                  pitch:(double)pitch
                   bank:(double)bank
    verticalFieldOfView:(double)verticalFieldOfView
           aircraftMode:(bool)aircraftMode {
  _heading = heading;
  _pitch = pitch;
  _bank = bank;
  _verticalFieldOfView = verticalFieldOfView;
  _aircraftMode = aircraftMode;
  self.needsDisplay = YES;
}

- (void)drawRect:(NSRect)dirtyRect {
  (void)dirtyRect;
  if (!(self.bounds.size.width > 0.0) || !(self.bounds.size.height > 0.0) ||
      !(_verticalFieldOfView > 0.0)) {
    return;
  }

  const NSPoint centre = NSMakePoint(NSMidX(self.bounds), NSMidY(self.bounds));
  const double bank = _aircraftMode ? _bank : 0.0;
  const double sinBank = std::sin(bank);
  const double cosBank = std::cos(bank);
  const double focalLength =
      self.bounds.size.height * 0.5 / std::tan(std::clamp(_verticalFieldOfView, 0.01, 3.0) * 0.5);
  const auto attitudePoint = [&](double x, double y) {
    return NSMakePoint(centre.x + x * cosBank - y * sinBank, centre.y + x * sinBank + y * cosBank);
  };

  NSShadow *textShadow = [[NSShadow alloc] init];
  textShadow.shadowColor = NSColor.blackColor;
  textShadow.shadowBlurRadius = 2.0;
  textShadow.shadowOffset = NSMakeSize(0.0, -1.0);
  NSDictionary<NSAttributedStringKey, id> *textAttributes = @{
    NSFontAttributeName : [NSFont monospacedDigitSystemFontOfSize:10.0 weight:NSFontWeightSemibold],
    NSForegroundColorAttributeName : NSColor.whiteColor,
    NSShadowAttributeName : textShadow,
  };
  const auto drawCentredText = [&](NSString *text, NSPoint point) {
    const NSSize size = [text sizeWithAttributes:textAttributes];
    [text drawAtPoint:NSMakePoint(point.x - size.width * 0.5, point.y - size.height * 0.5)
        withAttributes:textAttributes];
  };

  // The pitch ladder is expressed in world elevation angles. It moves behind
  // the fixed boresight and rotates with the apparent horizon as the aircraft
  // banks. Clipping keeps the overlay useful without obscuring much terrain.
  [NSGraphicsContext saveGraphicsState];
  [NSBezierPath clipRect:NSMakeRect(centre.x - 135.0, centre.y - 82.0, 270.0, 164.0)];
  for (int pitchDegrees = -60; pitchDegrees <= 60; pitchDegrees += 10) {
    const double pitchRadians =
        static_cast<double>(pitchDegrees) * panorama::app::kDegreesToRadians;
    const double y = std::tan(pitchRadians - _pitch) * focalLength;
    if (std::abs(y) > 100.0) {
      continue;
    }

    const bool horizon = pitchDegrees == 0;
    const double halfWidth = horizon ? 112.0 : (pitchDegrees % 20 == 0 ? 43.0 : 34.0);
    const double centreGap = horizon ? 20.0 : 9.0;
    NSBezierPath *line = [NSBezierPath bezierPath];
    [line moveToPoint:attitudePoint(-halfWidth, y)];
    [line lineToPoint:attitudePoint(-centreGap, y)];
    [line moveToPoint:attitudePoint(centreGap, y)];
    [line lineToPoint:attitudePoint(halfWidth, y)];
    if (pitchDegrees < 0) {
      const CGFloat dash[] = {4.0, 3.0};
      [line setLineDash:dash count:2 phase:0.0];
    }
    stroke_hud_path(line, horizon ? 1.6 : 1.0);

    if (!horizon) {
      NSString *label = [NSString stringWithFormat:@"%d", std::abs(pitchDegrees)];
      drawCentredText(label, attitudePoint(-halfWidth - 13.0, y));
      drawCentredText(label, attitudePoint(halfWidth + 13.0, y));
    }
  }
  [NSGraphicsContext restoreGraphicsState];

  if (_aircraftMode) {
    constexpr double kBankRadius = 70.0;
    NSBezierPath *bankArc = [NSBezierPath bezierPath];
    [bankArc appendBezierPathWithArcWithCenter:centre
                                        radius:kBankRadius
                                    startAngle:30.0
                                      endAngle:150.0];
    for (const int degrees : {-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60}) {
      const double radians = static_cast<double>(degrees) * panorama::app::kDegreesToRadians;
      const double innerRadius = kBankRadius - (degrees % 30 == 0 ? 7.0 : 4.0);
      [bankArc moveToPoint:NSMakePoint(
                               centre.x + innerRadius * std::sin(radians),
                               centre.y + innerRadius * std::cos(radians)
                           )];
      [bankArc lineToPoint:NSMakePoint(
                               centre.x + kBankRadius * std::sin(radians),
                               centre.y + kBankRadius * std::cos(radians)
                           )];
    }
    stroke_hud_path(bankArc, 1.0);

    const double indicatedBank = std::clamp(
        bank,
        -60.0 * panorama::app::kDegreesToRadians,
        60.0 * panorama::app::kDegreesToRadians
    );
    const double tangentX = std::cos(indicatedBank);
    const double tangentY = -std::sin(indicatedBank);
    const double radialX = std::sin(indicatedBank);
    const double radialY = std::cos(indicatedBank);
    NSBezierPath *bankPointer = [NSBezierPath bezierPath];
    [bankPointer moveToPoint:NSMakePoint(
                                 centre.x + radialX * (kBankRadius - 6.0),
                                 centre.y + radialY * (kBankRadius - 6.0)
                             )];
    [bankPointer lineToPoint:NSMakePoint(
                                 centre.x + radialX * (kBankRadius + 4.0) + tangentX * 4.0,
                                 centre.y + radialY * (kBankRadius + 4.0) + tangentY * 4.0
                             )];
    [bankPointer lineToPoint:NSMakePoint(
                                 centre.x + radialX * (kBankRadius + 4.0) - tangentX * 4.0,
                                 centre.y + radialY * (kBankRadius + 4.0) - tangentY * 4.0
                             )];
    [bankPointer closePath];
    [NSColor.blackColor setStroke];
    bankPointer.lineWidth = 3.0;
    [bankPointer stroke];
    [NSColor.whiteColor setFill];
    [bankPointer fill];
  }

  // A short compass ribbon makes heading readable without moving the user's
  // attention to a corner of the view.
  const double headingDegrees = std::remainder(_heading * panorama::app::kRadiansToDegrees, 360.0);
  constexpr double kHeadingPixelsPerDegree = 2.35;
  const double tapeY = centre.y + 108.0;
  [NSGraphicsContext saveGraphicsState];
  [NSBezierPath clipRect:NSMakeRect(centre.x - 108.0, tapeY - 4.0, 216.0, 35.0)];
  NSBezierPath *headingTape = [NSBezierPath bezierPath];
  [headingTape moveToPoint:NSMakePoint(centre.x - 108.0, tapeY)];
  [headingTape lineToPoint:NSMakePoint(centre.x + 108.0, tapeY)];
  const int firstTick = static_cast<int>(std::floor((headingDegrees - 50.0) / 10.0)) * 10;
  for (int tick = firstTick; tick <= headingDegrees + 50.0; tick += 10) {
    const double x =
        centre.x + (static_cast<double>(tick) - headingDegrees) * kHeadingPixelsPerDegree;
    const int normalisedTick = (tick % 360 + 360) % 360;
    const bool labelled = normalisedTick % 30 == 0;
    [headingTape moveToPoint:NSMakePoint(x, tapeY)];
    [headingTape lineToPoint:NSMakePoint(x, tapeY + (labelled ? 9.0 : 5.0))];
    if (labelled) {
      drawCentredText(heading_label(tick), NSMakePoint(x, tapeY + 18.0));
    }
  }
  stroke_hud_path(headingTape, 1.0);
  [NSGraphicsContext restoreGraphicsState];

  NSBezierPath *headingPointer = [NSBezierPath bezierPath];
  [headingPointer moveToPoint:NSMakePoint(centre.x, tapeY - 1.0)];
  [headingPointer lineToPoint:NSMakePoint(centre.x - 4.0, tapeY - 7.0)];
  [headingPointer lineToPoint:NSMakePoint(centre.x + 4.0, tapeY - 7.0)];
  [headingPointer closePath];
  [NSColor.blackColor setStroke];
  headingPointer.lineWidth = 3.0;
  [headingPointer stroke];
  [NSColor.whiteColor setFill];
  [headingPointer fill];
  const int displayedHeading = (static_cast<int>(std::lround(headingDegrees)) % 360 + 360) % 360;
  drawCentredText(
      [NSString stringWithFormat:@"%03d°", displayedHeading],
      NSMakePoint(centre.x, tapeY - 17.0)
  );

  // Draw the boresight last so it remains the dominant, fixed steering datum.
  NSBezierPath *boresight = [NSBezierPath bezierPath];
  [boresight appendBezierPathWithOvalInRect:NSMakeRect(centre.x - 5.0, centre.y - 5.0, 10.0, 10.0)];
  [boresight moveToPoint:NSMakePoint(centre.x - 23.0, centre.y)];
  [boresight lineToPoint:NSMakePoint(centre.x - 9.0, centre.y)];
  [boresight moveToPoint:NSMakePoint(centre.x + 9.0, centre.y)];
  [boresight lineToPoint:NSMakePoint(centre.x + 23.0, centre.y)];
  [boresight moveToPoint:NSMakePoint(centre.x, centre.y - 16.0)];
  [boresight lineToPoint:NSMakePoint(centre.x, centre.y - 9.0)];
  stroke_hud_path(boresight, 1.3);
}
@end

@interface PanoramaView : MTKView {
@private
  NSPoint _lastMouseLocation;
  NSTrackingArea *_inspectionTrackingArea;
  LockedPointMarkerView *_lockedPointMarker;
  ViewerPauseIndicatorView *_pauseIndicator;
  NSTextField *_pauseIndicatorLabel;
  CruiseHUDView *_cruiseHUD;
  double _lockedPointPixelX;
  double _lockedPointPixelY;
  double _lockedPointDirectionX;
  double _lockedPointDirectionY;
  bool _pointInspectionEnabled;
  bool _mouseTurningEnabled;
  bool _cruiseSteeringEnabled;
  bool _viewerPaused;
  bool _lockedPointIndicatorActive;
  bool _lockedPointOnscreen;
  bool _pointIndicatorLocked;
  bool _lockedPointOccluded;
}
@property(nonatomic, weak) PanoramaController *panoramaController;
- (void)setPointInspectionEnabled:(bool)enabled;
- (void)setMouseTurningEnabled:(bool)enabled;
- (void)setCruiseSteeringEnabled:(bool)enabled;
- (void)setCruiseHUDHeading:(double)heading
                      pitch:(double)pitch
                       bank:(double)bank
        verticalFieldOfView:(double)verticalFieldOfView
               aircraftMode:(bool)aircraftMode;
- (void)setViewerPaused:(bool)paused recoveryMessage:(NSString *)recoveryMessage;
- (void)setTerrainPointIndicator:(std::optional<panorama::app::LockedPointProjection>)projection
                          locked:(bool)locked
                        occluded:(bool)occluded;
@end

@interface PanoramaController
    : NSObject <MTKViewDelegate, NSTextFieldDelegate, MiniMapPanelViewInteractionDelegate> {
@private
  panorama::app::ViewerRenderer *_renderer;
  __weak NSWindow *_window;
  __weak PanoramaView *_panoramaView;
  __weak ViewerOverlayView *_overlayView;
  __weak AspectFitContainerView *_aspectFitView;
  __weak MiniMapPanelView *_miniMapPanel;
  panorama::CameraOrientation _orientation;
  double _verticalFieldOfView;
  panorama::ImageSize _image;
  panorama::TerrainPresentationSettings _presentation;
  panorama::ObserverLocation _observer;
  std::optional<panorama::app::PointInspection> _lockedPoint;
  std::optional<panorama::app::PointInspection> _mapHoverPoint;
  NSPopUpButton *_colourSourceControl;
  NSPopUpButton *_colourmapControl;
  NSPopUpButton *_colourScaleControl;
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
  NSButton *_raytracedShadowsControl;
  NSButton *_featureOutlinesControl;
  NSSlider *_featureOutlineDetailControl;
  NSTextField *_featureOutlineDetailLabel;
  NSSlider *_lodScaleControl;
  NSTextField *_lodScaleLabel;
  NSSlider *_sunAzimuthControl;
  NSTextField *_sunAzimuthLabel;
  NSSlider *_sunAltitudeControl;
  NSTextField *_sunAltitudeLabel;
  NSSegmentedControl *_sunModeControl;
  NSTextField *_astronomicalDateControl;
  NSSlider *_astronomicalTimeControl;
  NSTextField *_astronomicalTimeLabel;
  NSButton *_astronomicalTimeDecreaseControl;
  NSButton *_astronomicalTimeIncreaseControl;
  NSTextField *_daylightTimesLabel;
  NSStackView *_daylightSymbolsRow;
  NSTextField *_sunriseTimeLabel;
  NSTextField *_sunsetTimeLabel;
  MKReverseGeocodingRequest *_timeZoneRequest;
  NSTimeZone *_observerTimeZone;
  uint64_t _timeZoneRequestToken;
  bool _astronomicalControlsUseObserverTime;
  bool _timeZoneLookupInProgress;
  double _manualSunAzimuthDegrees;
  double _manualSunAltitudeDegrees;
  NSArray<NSView *> *_scalarColourRows;
  NSView *_featureOutlineDetailRow;
  NSArray<NSView *> *_normalLightingRows;
  NSArray<NSView *> *_manualSunRows;
  NSArray<NSView *> *_astronomicalSunRows;
  NSSlider *_diffusivityControl;
  NSTextField *_diffusivityLabel;
  NSSlider *_skyStrengthControl;
  NSTextField *_skyStrengthLabel;
  NSSlider *_skyDetailControl;
  NSTextField *_skyDetailLabel;
  NSTextField *_debugInfoLabel;
  NSTextField *_debugPointInfoLabel;
  NSTextField *_observerInfoLabel;
  NSTextField *_movementInfoLabel;
  NSStackView *_pointInfoRow;
  NSTextField *_pointInfoHeading;
  NSTextField *_pointInfoLabel;
  NSImageView *_pointVisibilityIcon;
  NSImageView *_pointLockIcon;
  NSButton *_moveToLockedPointControl;
  NSTextField *_groundClearanceControl;
  NSButton *_groundClearanceDecreaseControl;
  NSButton *_groundClearanceIncreaseControl;
  NSPopUpButton *_coordinateSystemControl;
  NSTextField *_coordinateInputControl;
  NSTextField *_coordinateStatusLabel;
  NSButton *_coordinateMoveControl;
  std::optional<panorama::app::ParsedCoordinateInput> _coordinateDestination;
  NSSegmentedControl *_movementModeControl;
  NSSegmentedControl *_roamTurningModeControl;
  NSView *_roamTurningModeRow;
  NSSlider *_roamMouseSensitivityControl;
  NSTextField *_roamMouseSensitivityLabel;
  NSView *_roamMouseSensitivityRow;
  NSSegmentedControl *_roamAltitudeModeControl;
  NSButton *_aircraftDynamicsControl;
  NSView *_aircraftDynamicsRow;
  NSSlider *_roamSpeedControl;
  NSTextField *_roamSpeedRowLabel;
  NSTextField *_roamSpeedLabel;
  NSSlider *_roamUpdateRateControl;
  NSTextField *_roamUpdateRateLabel;
  NSTextField *_roamStatusLabel;
  NSArray<NSView *> *_roamRows;
  NSTextField *_observerHeightLabel;
  NSTextField *_observerHeightUnit;
  id _pauseKeyMonitor;
  NSTimer *_roamTimer;
  panorama::app::MapCoordinate _roamDesiredPosition;
  std::chrono::steady_clock::time_point _lastRoamTick;
  uint64_t _roamRequestToken;
  uint64_t _displayedRoamResultSequence;
  uint64_t _displayedRevision;
  uint64_t _displayedInspectionSequence;
  uint64_t _pointLockRequestToken;
  uint64_t _displayedMapPointSequence;
  uint64_t _mapPointRequestToken;
  uint64_t _displayedTargetVisibilitySequence;
  uint64_t _targetVisibilityRequestToken;
  uint64_t _targetVisibilityRevision;
  uint64_t _inspectionRequestToken;
  panorama::app::MapPointAction _mapPointAction;
  panorama::app::PointerOwner _pointerOwner;
  double _lockedAspectRatio;
  double _panningSensitivity;
  double _groundClearance;
  double _roamAltitude;
  double _roamSpeed;
  double _cruiseSpeed;
  double _aircraftAirspeed;
  double _aircraftBank;
  double _cruiseSteeringX;
  double _cruiseSteeringY;
  bool _roamForwardPressed;
  bool _roamBackwardPressed;
  bool _roamLeftPressed;
  bool _roamRightPressed;
  bool _pointInspectionEnabled;
  bool _pointInspectionLocked;
  bool _pointLockPending;
  bool _lockedPointOccluded;
  bool _coordinateMovePending;
  bool _invertMousePanning;
  bool _viewerPaused;
  bool _cruiseRecovery;
  bool _cruiseSteeringActive;
  bool _updatingResolutionControls;
}
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)rotateForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)panForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)mouseTurnForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)zoomWithScrollDelta:(double)delta precise:(bool)precise;
- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView
             aspectFitView:(AspectFitContainerView *)aspectFitView
              miniMapPanel:(MiniMapPanelView *)miniMapPanel;
- (void)inspectPixelX:(uint32_t)x y:(uint32_t)y;
- (void)pointerMovedOverPanorama;
- (void)pointerMovedOverOccludingView:(NSView *)view;
- (void)panoramaPointerExited;
- (void)togglePointLockAtPixelX:(uint32_t)x y:(uint32_t)y;
- (void)toggleMapAndPointInspection:(id)sender;
- (BOOL)isMapAndPointInspectionEnabled;
- (BOOL)isRoamingEnabled;
- (BOOL)isCruisingEnabled;
- (BOOL)isAircraftDynamicsEnabled;
- (BOOL)isMouseTurningEnabled;
- (BOOL)isViewerPaused;
- (void)toggleViewerPause;
- (void)pauseCruiseForTerrainCollision:(bool)terrainCollision;
- (void)setRoamKey:(panorama::app::RoamKey)key pressed:(BOOL)pressed;
- (void)adjustCruiseSpeedBy:(double)delta;
- (void)setCruiseSteeringX:(double)x y:(double)y;
- (void)clearCruiseSteering;
- (void)setPointInfoStatus:(NSString *)status;
- (void)updateMiniMapTelemetry;
- (void)setPointInfoSymbolsVisible:(bool)visible locked:(bool)locked occluded:(bool)occluded;
- (void)moveObserverToTerrainPoint:(panorama::app::TerrainPoint)point;
- (void)moveToLockedPoint:(id)sender;
- (void)adjustGroundClearance:(NSButton *)sender;
- (BOOL)commitGroundClearanceControl;
- (void)movementModeChanged:(id)sender;
- (void)roamTurningModeChanged:(id)sender;
- (void)roamMouseSensitivityChanged:(id)sender;
- (void)roamAltitudeModeChanged:(id)sender;
- (void)aircraftDynamicsChanged:(id)sender;
- (void)levelAircraft;
- (void)roamSpeedChanged:(id)sender;
- (void)updateMovementSpeedControl;
- (void)roamUpdateRateChanged:(id)sender;
- (void)updateRoamControls;
- (void)updateCruiseHUD;
- (void)updateMovementStatus;
- (void)scheduleRoamTimer;
- (void)roamTimerFired:(NSTimer *)timer;
- (void)clearRoamKeys;
- (BOOL)hasPressedRoamKey;
- (void)coordinateSystemChanged:(id)sender;
- (void)moveToCoordinate:(id)sender;
- (BOOL)updateCoordinateInputValidation;
- (void)resolveObserverTimeZone;
- (void)setDaylightStatus:(NSString *)status;
- (BOOL)publishAstronomicalLighting;
- (BOOL)publishTerrainControls;
- (BOOL)commitResolutionControls;
- (NSViewController *)makeSettingsViewController;
- (NSViewController *)makePositioningViewController;
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
    NSString *description = _lockedPointOccluded
                                ? @"Locked terrain point is occluded"
                                : (_pointIndicatorLocked ? @"Locked terrain point"
                                                         : @"Terrain point under map pointer");
    _lockedPointMarker.image = [NSImage imageWithSystemSymbolName:@"scope"
                                         accessibilityDescription:description];
    [_lockedPointMarker.layer setAffineTransform:CGAffineTransformIdentity];
    _lockedPointMarker.toolTip = description;
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
    NSString *description = _pointIndicatorLocked ? @"Locked terrain point is outside the view"
                                                  : @"Map pointer is outside the view";
    _lockedPointMarker.image = [NSImage imageWithSystemSymbolName:@"arrow.up.circle.fill"
                                         accessibilityDescription:description];
    const CGFloat rotation = static_cast<CGFloat>(std::atan2(-direction_x, direction_y));
    [_lockedPointMarker.layer setAffineTransform:CGAffineTransformMakeRotation(rotation)];
    _lockedPointMarker.toolTip = description;
  }

  _lockedPointMarker.frame = NSMakeRect(
      centre.x - kMarkerSize * 0.5,
      centre.y - kMarkerSize * 0.5,
      kMarkerSize,
      kMarkerSize
  );
  _lockedPointMarker.hidden = NO;
}

- (void)setTerrainPointIndicator:(std::optional<panorama::app::LockedPointProjection>)projection
                          locked:(bool)locked
                        occluded:(bool)occluded {
  _lockedPointIndicatorActive = projection.has_value();
  _pointIndicatorLocked = locked;
  _lockedPointOccluded = locked && occluded;
  if (!_lockedPointIndicatorActive) {
    _lockedPointMarker.hidden = YES;
    return;
  }
  if (_lockedPointMarker == nil) {
    _lockedPointMarker = [[LockedPointMarkerView alloc] initWithFrame:NSZeroRect];
    _lockedPointMarker.imageScaling = NSImageScaleProportionallyUpOrDown;
    _lockedPointMarker.wantsLayer = YES;
    _lockedPointMarker.layer.shadowColor = NSColor.blackColor.CGColor;
    _lockedPointMarker.layer.shadowOpacity = 0.9F;
    _lockedPointMarker.layer.shadowRadius = 2.0;
    _lockedPointMarker.layer.shadowOffset = CGSizeZero;
    [self addSubview:_lockedPointMarker];
  }
  _lockedPointMarker.contentTintColor =
      locked ? NSColor.systemOrangeColor : NSColor.systemBlueColor;
  _lockedPointMarker.alphaValue = _lockedPointOccluded ? 0.45 : 1.0;
  _lockedPointMarker.layer.shadowOpacity = _lockedPointOccluded ? 0.35F : 0.9F;
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
  if (!_pointInspectionEnabled && !_mouseTurningEnabled && !_cruiseSteeringEnabled) {
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
  if (_viewerPaused) {
    [self addCursorRect:self.bounds cursor:NSCursor.arrowCursor];
  } else if (_cruiseSteeringEnabled) {
    [self addCursorRect:self.bounds cursor:NSCursor.arrowCursor];
  } else if (_pointInspectionEnabled) {
    [self addCursorRect:self.bounds cursor:NSCursor.crosshairCursor];
  }
}

- (void)setPointInspectionEnabled:(bool)enabled {
  _pointInspectionEnabled = enabled;
  self.window.acceptsMouseMovedEvents = enabled || _mouseTurningEnabled || _cruiseSteeringEnabled;
  [self updateTrackingAreas];
  [self.window invalidateCursorRectsForView:self];
}

- (void)setMouseTurningEnabled:(bool)enabled {
  _mouseTurningEnabled = enabled;
  self.window.acceptsMouseMovedEvents =
      enabled || _pointInspectionEnabled || _cruiseSteeringEnabled;
  [self updateTrackingAreas];
}

- (void)setCruiseSteeringEnabled:(bool)enabled {
  _cruiseSteeringEnabled = enabled;
  if (_cruiseHUD == nil) {
    _cruiseHUD = [[CruiseHUDView alloc] initWithFrame:NSZeroRect];
    _cruiseHUD.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:_cruiseHUD];
    [NSLayoutConstraint activateConstraints:@[
      [_cruiseHUD.leadingAnchor constraintEqualToAnchor:self.leadingAnchor],
      [_cruiseHUD.trailingAnchor constraintEqualToAnchor:self.trailingAnchor],
      [_cruiseHUD.topAnchor constraintEqualToAnchor:self.topAnchor],
      [_cruiseHUD.bottomAnchor constraintEqualToAnchor:self.bottomAnchor],
    ]];
  }
  _cruiseHUD.hidden = !enabled;
  self.window.acceptsMouseMovedEvents = enabled || _pointInspectionEnabled || _mouseTurningEnabled;
  [self updateTrackingAreas];
  [self.window invalidateCursorRectsForView:self];
}

- (void)setCruiseHUDHeading:(double)heading
                      pitch:(double)pitch
                       bank:(double)bank
        verticalFieldOfView:(double)verticalFieldOfView
               aircraftMode:(bool)aircraftMode {
  [_cruiseHUD setHeading:heading
                    pitch:pitch
                     bank:bank
      verticalFieldOfView:verticalFieldOfView
             aircraftMode:aircraftMode];
}

- (void)setViewerPaused:(bool)paused recoveryMessage:(NSString *)recoveryMessage {
  _viewerPaused = paused;
  if (_pauseIndicator == nil) {
    NSImageView *icon =
        [NSImageView imageViewWithImage:[NSImage imageWithSystemSymbolName:@"pause.fill"
                                                  accessibilityDescription:@"Viewer paused"]];
    icon.contentTintColor = NSColor.labelColor;
    _pauseIndicatorLabel = [NSTextField labelWithString:@"Paused — Space to resume"];
    _pauseIndicatorLabel.font = [NSFont systemFontOfSize:NSFont.systemFontSize
                                                  weight:NSFontWeightSemibold];
    _pauseIndicator = [[ViewerPauseIndicatorView alloc] initWithFrame:NSZeroRect];
    [_pauseIndicator addArrangedSubview:icon];
    [_pauseIndicator addArrangedSubview:_pauseIndicatorLabel];
    _pauseIndicator.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    _pauseIndicator.alignment = NSLayoutAttributeCenterY;
    _pauseIndicator.spacing = 8.0;
    _pauseIndicator.edgeInsets = NSEdgeInsetsMake(8.0, 12.0, 8.0, 12.0);
    _pauseIndicator.wantsLayer = YES;
    _pauseIndicator.layer.backgroundColor =
        [NSColor.windowBackgroundColor colorWithAlphaComponent:0.88].CGColor;
    _pauseIndicator.layer.cornerRadius = 10.0;
    _pauseIndicator.layer.borderColor =
        [NSColor.separatorColor colorWithAlphaComponent:0.6].CGColor;
    _pauseIndicator.layer.borderWidth = 1.0;
    _pauseIndicator.layer.shadowColor = NSColor.blackColor.CGColor;
    _pauseIndicator.layer.shadowOpacity = 0.35F;
    _pauseIndicator.layer.shadowRadius = 5.0;
    _pauseIndicator.layer.shadowOffset = CGSizeMake(0.0, -1.0);
    _pauseIndicator.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:_pauseIndicator];
    [NSLayoutConstraint activateConstraints:@[
      [_pauseIndicator.centerXAnchor constraintEqualToAnchor:self.centerXAnchor],
      [_pauseIndicator.topAnchor constraintEqualToAnchor:self.topAnchor constant:16.0],
    ]];
  }
  _pauseIndicatorLabel.stringValue =
      recoveryMessage != nil ? recoveryMessage : @"Paused — Space to resume";
  _pauseIndicator.hidden = !paused;
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
  if (!_pointInspectionEnabled && !_mouseTurningEnabled && !_cruiseSteeringEnabled) {
    [super mouseMoved:event];
    return;
  }
  NSView *content = self.window.contentView;
  const NSPoint contentPoint = [content convertPoint:event.locationInWindow fromView:nil];
  NSView *hit = [content hitTest:contentPoint];
  if (hit != self && ![hit isDescendantOf:self]) {
    [self.panoramaController pointerMovedOverOccludingView:hit];
    return;
  }
  [self.panoramaController pointerMovedOverPanorama];
  if (_cruiseSteeringEnabled && !_viewerPaused) {
    const NSPoint location = [self convertPoint:event.locationInWindow fromView:nil];
    const NSRect bounds = self.bounds;
    if (bounds.size.width > 0.0 && bounds.size.height > 0.0) {
      const double x =
          std::clamp((location.x - NSMidX(bounds)) / (bounds.size.width * 0.5), -1.0, 1.0);
      const double y =
          std::clamp((location.y - NSMidY(bounds)) / (bounds.size.height * 0.5), -1.0, 1.0);
      [self.panoramaController setCruiseSteeringX:x y:y];
    }
  } else if (_mouseTurningEnabled && !_viewerPaused) {
    // NSEvent's vertical mouse delta is positive downwards, unlike camera
    // pitch, so invert it to preserve the existing drag direction.
    [self.panoramaController mouseTurnForCurrentZoomHeading:event.deltaX * 0.003
                                                      pitch:-event.deltaY * 0.003];
  }
  if (!_pointInspectionEnabled) {
    return;
  }
  uint32_t x = 0U;
  uint32_t y = 0U;
  if (![self inspectionPixelForEvent:event x:&x y:&y]) {
    return;
  }
  [self.panoramaController inspectPixelX:x y:y];
}

/// Right-click locks the current terrain sample without consuming ordinary
/// left-button interaction.
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
  if (_pointInspectionEnabled || _cruiseSteeringEnabled) {
    [self.panoramaController panoramaPointerExited];
  }
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (void)mouseDown:(NSEvent *)event {
  [self.window makeFirstResponder:self];
  _lastMouseLocation = [self convertPoint:event.locationInWindow fromView:nil];
}

- (void)mouseDragged:(NSEvent *)event {
  if (!_viewerPaused && (_mouseTurningEnabled || _cruiseSteeringEnabled)) {
    // AppKit changes mouse-move events into drag events while a button is
    // held. Mouse turning should remain continuous without requiring a drag.
    [self mouseMoved:event];
    return;
  }
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
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:-kStep pitch:0.0];
    }
    break;
  case 124: // Right arrow.
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:kStep pitch:0.0];
    }
    break;
  case 125: // Down arrow.
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:-kStep];
    }
    break;
  case 126: // Up arrow.
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:kStep];
    }
    break;
  default: {
    const NSString *characters = event.charactersIgnoringModifiers.lowercaseString;
    if ([self.panoramaController isCruisingEnabled] &&
        ([characters isEqualToString:@"w"] || [characters isEqualToString:@"s"])) {
      const double step = (event.modifierFlags & NSEventModifierFlagShift) != 0U ? 4.0 : 1.0;
      [self.panoramaController
          adjustCruiseSpeedBy:[characters isEqualToString:@"w"] ? step : -step];
    } else if ([self.panoramaController isCruisingEnabled] &&
               ([characters isEqualToString:@"a"] || [characters isEqualToString:@"d"])) {
      // Cruise steering is deliberately mouse-only.
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"w"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Forward pressed:YES];
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"s"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Backward pressed:YES];
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"a"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Left pressed:YES];
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"d"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Right pressed:YES];
    } else if ([characters isEqualToString:@"a"]) {
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

- (void)keyUp:(NSEvent *)event {
  const NSString *characters = event.charactersIgnoringModifiers.lowercaseString;
  if ([self.panoramaController isCruisingEnabled] &&
      ([characters isEqualToString:@"w"] || [characters isEqualToString:@"s"] ||
       [characters isEqualToString:@"a"] || [characters isEqualToString:@"d"])) {
    return;
  }
  if (![self.panoramaController isRoamingEnabled]) {
    [super keyUp:event];
    return;
  }
  if ([characters isEqualToString:@"w"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Forward pressed:NO];
  } else if ([characters isEqualToString:@"s"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Backward pressed:NO];
  } else if ([characters isEqualToString:@"a"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Left pressed:NO];
  } else if ([characters isEqualToString:@"d"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Right pressed:NO];
  } else {
    [super keyUp:event];
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

/// A flipped document view keeps the first inspector section at the top when
/// its intrinsic height grows beyond the scroll view's visible area.
@interface InspectorDocumentView : NSView
@end

@implementation InspectorDocumentView

- (BOOL)isFlipped {
  return YES;
}

@end

/// A compact inspector section whose disclosure state survives app launches.
/// Hiding the content stack removes it from its parent stack's fitting height,
/// so collapsed sections also shorten the scrollable document immediately.
@interface InspectorSectionView : NSStackView {
@private
  NSButton *_disclosureButton;
  NSStackView *_contentStack;
  NSString *_defaultsKey;
}
- (instancetype)initWithTitle:(NSString *)title
                     controls:(NSArray<NSView *> *)controls
                  defaultsKey:(NSString *)defaultsKey;
@end

@implementation InspectorSectionView

- (instancetype)initWithTitle:(NSString *)title
                     controls:(NSArray<NSView *> *)controls
                  defaultsKey:(NSString *)defaultsKey {
  self = [super initWithFrame:NSZeroRect];
  if (self != nil) {
    _defaultsKey = [defaultsKey copy];
    _disclosureButton = [NSButton buttonWithTitle:title
                                           target:self
                                           action:@selector(toggleDisclosure:)];
    _disclosureButton.bordered = NO;
    _disclosureButton.imagePosition = NSImageLeading;
    _disclosureButton.alignment = NSTextAlignmentLeft;
    _disclosureButton.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize
                                               weight:NSFontWeightSemibold];

    _contentStack = [NSStackView stackViewWithViews:controls];
    _contentStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    _contentStack.alignment = NSLayoutAttributeLeading;
    _contentStack.spacing = 8.0;

    self.orientation = NSUserInterfaceLayoutOrientationVertical;
    self.alignment = NSLayoutAttributeLeading;
    self.spacing = 8.0;
    [self addArrangedSubview:_disclosureButton];
    [self addArrangedSubview:_contentStack];

    NSNumber *saved = [NSUserDefaults.standardUserDefaults objectForKey:_defaultsKey];
    [self setExpanded:saved == nil || saved.boolValue];
  }
  return self;
}

- (void)setExpanded:(BOOL)expanded {
  _contentStack.hidden = !expanded;
  _disclosureButton.image =
      [NSImage imageWithSystemSymbolName:expanded ? @"chevron.down" : @"chevron.right"
                accessibilityDescription:expanded ? @"Collapse section" : @"Expand section"];
  _disclosureButton.accessibilityValue = expanded ? @"Expanded" : @"Collapsed";
}

- (void)toggleDisclosure:(id)sender {
  (void)sender;
  const BOOL expanded = _contentStack.hidden;
  [self setExpanded:expanded];
  [NSUserDefaults.standardUserDefaults setBool:expanded forKey:_defaultsKey];
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
@interface ViewerOverlayView : NSView <MiniMapPanelViewSizeDelegate> {
@private
  NSView *_contentView;
  NSView *_inspectorView;
  NSView *_settingsView;
  NSView *_debugView;
  NSView *_debugContentView;
  NSView *_mapPanelView;
  MiniMapPanelView *_mapPanelContentView;
  CGFloat _inspectorWidth;
  NSSize _debugSize;
  CGFloat _panelMargin;
  bool _inspectorVisible;
  bool _debugVisible;
  bool _mapAndPointInfoVisible;
}
- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth
                    debugView:(NSView *)debugView
                    debugSize:(NSSize)debugSize
                 mapPanelView:(MiniMapPanelView *)mapPanelView;
- (void)toggleInspector:(id)sender;
- (void)toggleDebugOverlay:(id)sender;
- (void)setMapAndPointInfoVisible:(bool)visible;
@end

@implementation ViewerOverlayView

- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth
                    debugView:(NSView *)debugView
                    debugSize:(NSSize)debugSize
                 mapPanelView:(MiniMapPanelView *)mapPanelView {
  self = [super initWithFrame:frame];
  if (self != nil) {
    _contentView = contentView;
    _settingsView = settingsView;
    _debugContentView = debugView;
    _mapPanelContentView = mapPanelView;
    _mapPanelContentView.sizeDelegate = self;
    _inspectorWidth = inspectorWidth;
    _debugSize = debugSize;
    _panelMargin = 12.0;
    _inspectorVisible = true;
    _debugVisible = false;
    _mapAndPointInfoVisible = false;
    self.wantsLayer = YES;
    self.layer.masksToBounds = YES;
    [self addSubview:_contentView];

    _inspectorView = makeOverlayPanel(_settingsView);
    _debugView = makeOverlayPanel(_debugContentView);
    _mapPanelView = makeOverlayPanel(_mapPanelContentView);
    [self addSubview:_inspectorView];
    [self addSubview:_debugView];
    [self addSubview:_mapPanelView];
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

- (NSRect)mapPanelFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  const NSEdgeInsets safeArea = self.safeAreaInsets;
  NSSize preferredSize = [_mapPanelContentView preferredPanelSize];
  // Preserve the outgoing panel's size while its final visible section slides
  // away; the content reports zero height once both sections are disabled.
  if (!visible && preferredSize.height == 0.0 && _mapPanelView.frame.size.height > 0.0) {
    preferredSize = _mapPanelView.frame.size;
  }
  const CGFloat availableHeight =
      std::max(0.0, bounds.size.height - safeArea.top - safeArea.bottom - 2.0 * _panelMargin);
  const CGFloat availableWidth =
      std::max(0.0, bounds.size.width - safeArea.left - safeArea.right - 2.0 * _panelMargin);
  const CGFloat width = std::min(preferredSize.width, availableWidth);
  const CGFloat height = std::min(preferredSize.height, availableHeight);
  const CGFloat x = visible ? NSMinX(bounds) + safeArea.left + _panelMargin
                            : NSMinX(bounds) - width - _panelMargin;
  const CGFloat y = NSMinY(bounds) + safeArea.bottom + _panelMargin;
  return NSMakeRect(x, y, width, height);
}

- (void)layout {
  [super layout];
  _contentView.frame = self.bounds;
  _inspectorView.frame = [self inspectorFrameForVisible:_inspectorVisible];
  _settingsView.frame = _inspectorView.bounds;
  _debugView.frame = [self debugFrameForVisible:_debugVisible];
  _debugContentView.frame = _debugView.bounds;
  _mapPanelView.frame = [self mapPanelFrameForVisible:_mapAndPointInfoVisible];
  _mapPanelContentView.frame = _mapPanelView.bounds;
}

- (void)miniMapPanelPreferredSizeDidChange:(MiniMapPanelView *)panel {
  if (panel != _mapPanelContentView) {
    return;
  }
  const NSRect targetFrame = [self mapPanelFrameForVisible:_mapAndPointInfoVisible];
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _mapPanelView.animator.frame = targetFrame;
    _mapPanelContentView.animator.frame =
        NSMakeRect(0.0, 0.0, targetFrame.size.width, targetFrame.size.height);
  }];
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

- (void)setMapAndPointInfoVisible:(bool)visible {
  if (_mapAndPointInfoVisible == visible) {
    return;
  }
  _mapAndPointInfoVisible = visible;
  [_mapPanelContentView setMapAndPointInfoVisible:visible];
  const NSRect targetFrame = [self mapPanelFrameForVisible:visible];
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _mapPanelView.animator.frame = targetFrame;
    _mapPanelContentView.animator.frame =
        NSMakeRect(0.0, 0.0, targetFrame.size.width, targetFrame.size.height);
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
    _observer = renderer->observer();
    _orientation = renderer->initial_orientation();
    _verticalFieldOfView = renderer->initial_vertical_field_of_view();
    _image = renderer->initial_image();
    _lockedAspectRatio = static_cast<double>(_image.width) / _image.height;
    _panningSensitivity = 8.0;
    _groundClearance = renderer->ground_clearance();
    _roamAltitude = _observer.elevation;
    _roamSpeed = panorama::app::kDefaultRoamSpeed;
    _cruiseSpeed = panorama::app::kDefaultCruiseSpeed;
    _aircraftAirspeed = _cruiseSpeed;
    _roamDesiredPosition = {_observer.easting, _observer.northing};
    _presentation = renderer->initial_presentation();
    _mapPointAction = panorama::app::MapPointAction::None;
    _pointerOwner = panorama::app::PointerOwner::None;

    __weak PanoramaController *weakSelf = self;
    _pauseKeyMonitor = [NSEvent
        addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                     handler:^NSEvent *(NSEvent *event) {
                                       PanoramaController *controller = weakSelf;
                                       if (controller == nil ||
                                           event.window != controller->_window ||
                                           event.keyCode != 49) {
                                         return event;
                                       }
                                       const NSEventModifierFlags modifiers =
                                           event.modifierFlags &
                                           NSEventModifierFlagDeviceIndependentFlagsMask;
                                       if ((modifiers & (NSEventModifierFlagCommand |
                                                         NSEventModifierFlagControl |
                                                         NSEventModifierFlagOption)) != 0U ||
                                           [controller->_window.firstResponder
                                               isKindOfClass:NSTextView.class]) {
                                         return event;
                                       }
                                       if (![controller isViewerPaused] &&
                                           ![controller isMouseTurningEnabled] &&
                                           ![controller isCruisingEnabled] &&
                                           controller->_window.firstResponder !=
                                               controller->_panoramaView) {
                                         return event;
                                       }
                                       if (!event.isARepeat) {
                                         [controller toggleViewerPause];
                                       }
                                       return nil;
                                     }];
  }
  return self;
}

- (void)dealloc {
  if (_pauseKeyMonitor != nil) {
    [NSEvent removeMonitor:_pauseKeyMonitor];
  }
}

- (BOOL)isRoamingEnabled {
  return _movementModeControl != nil && _movementModeControl.selectedSegment == 1;
}

- (BOOL)isCruisingEnabled {
  return _movementModeControl != nil && _movementModeControl.selectedSegment == 2;
}

- (BOOL)isAircraftDynamicsEnabled {
  return [self isCruisingEnabled] && _aircraftDynamicsControl != nil &&
         _aircraftDynamicsControl.state == NSControlStateValueOn;
}

- (BOOL)isMouseTurningEnabled {
  return [self isRoamingEnabled] && _roamTurningModeControl != nil &&
         _roamTurningModeControl.selectedSegment == 1;
}

- (BOOL)isViewerPaused {
  return _viewerPaused;
}

- (void)toggleViewerPause {
  _viewerPaused = !_viewerPaused;
  _cruiseRecovery = false;
  [self clearRoamKeys];
  if (_viewerPaused) {
    [self clearCruiseSteering];
    [self levelAircraft];
  }
  _lastRoamTick = std::chrono::steady_clock::now();
  [_panoramaView setViewerPaused:_viewerPaused recoveryMessage:nil];
  if (_viewerPaused) {
    if ([self isCruisingEnabled]) {
      [self resolveObserverTimeZone];
    }
  }
  [self updateRoamControls];
}

- (void)pauseCruiseForTerrainCollision:(bool)terrainCollision {
  _viewerPaused = true;
  _cruiseRecovery = true;
  [self clearRoamKeys];
  [self clearCruiseSteering];
  [self levelAircraft];
  _lastRoamTick = std::chrono::steady_clock::now();
  NSString *message = terrainCollision
                          ? @"Terrain ahead — drag to steer or climb, then press Space"
                          : @"No terrain coverage — drag to turn back, then press Space";
  [_panoramaView setViewerPaused:true recoveryMessage:message];
  [self resolveObserverTimeZone];
  [self updateRoamControls];
}

- (BOOL)hasPressedRoamKey {
  return _roamForwardPressed || _roamBackwardPressed || _roamLeftPressed || _roamRightPressed;
}

- (void)clearRoamKeys {
  _roamForwardPressed = false;
  _roamBackwardPressed = false;
  _roamLeftPressed = false;
  _roamRightPressed = false;
}

- (void)setRoamKey:(panorama::app::RoamKey)key pressed:(BOOL)pressed {
  if (![self isRoamingEnabled] || _viewerPaused) {
    return;
  }
  const bool wasMoving = [self hasPressedRoamKey];
  bool *state = nullptr;
  switch (key) {
  case panorama::app::RoamKey::Forward:
    state = &_roamForwardPressed;
    break;
  case panorama::app::RoamKey::Backward:
    state = &_roamBackwardPressed;
    break;
  case panorama::app::RoamKey::Left:
    state = &_roamLeftPressed;
    break;
  case panorama::app::RoamKey::Right:
    state = &_roamRightPressed;
    break;
  }
  *state = pressed;
  if (pressed && !wasMoving) {
    _roamDesiredPosition = {_observer.easting, _observer.northing};
    _pointInspectionLocked = false;
    _pointLockPending = false;
    _lockedPoint.reset();
    _mapHoverPoint.reset();
    [self clearTargetVisibility];
    [_miniMapPanel clearInspectedPoint];
    [_panoramaView setTerrainPointIndicator:std::nullopt locked:true occluded:false];
  }
  if (![self hasPressedRoamKey]) {
    _lastRoamTick = std::chrono::steady_clock::now();
    [self updateMovementStatus];
  }
}

- (void)adjustCruiseSpeedBy:(double)delta {
  if (![self isCruisingEnabled] || (_viewerPaused && !_cruiseRecovery)) {
    return;
  }
  // Quarter-octave steps make key presses useful at walking and kilometre-per-
  // second speeds alike. Shift supplies four steps, i.e. one exact doubling.
  _cruiseSpeed = std::clamp(
      _cruiseSpeed * std::exp2(delta / 4.0),
      panorama::app::kMinimumMovementSpeed,
      panorama::app::kMaximumCruiseSpeed
  );
  [self updateMovementSpeedControl];
  [self updateMovementStatus];
  [self updateMiniMapTelemetry];
}

- (void)setCruiseSteeringX:(double)x y:(double)y {
  if (![self isCruisingEnabled]) {
    return;
  }
  _cruiseSteeringX = std::clamp(x, -1.0, 1.0);
  _cruiseSteeringY = std::clamp(y, -1.0, 1.0);
  const bool becameActive = !_cruiseSteeringActive;
  _cruiseSteeringActive = true;
  if (becameActive) {
    [self updateMovementStatus];
  }
}

- (void)clearCruiseSteering {
  const bool changed = _cruiseSteeringActive;
  _cruiseSteeringX = 0.0;
  _cruiseSteeringY = 0.0;
  _cruiseSteeringActive = false;
  if (changed) {
    [self updateMovementStatus];
  }
}

/// Pausing stops the flight simulation immediately and returns the camera to
/// wings-level attitude. Pitch and heading remain available to paused drag
/// interaction and become the aircraft's new attitude when Cruise resumes.
- (void)levelAircraft {
  _aircraftBank = 0.0;
  if (std::abs(_orientation.roll) <= 1e-12) {
    [self updateCruiseHUD];
    return;
  }
  _orientation.roll = 0.0;
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  [self updateCruiseHUD];
  [self updateMiniMapTelemetry];
}

- (void)scheduleRoamTimer {
  [_roamTimer invalidate];
  _roamTimer = nil;
  if (![self isRoamingEnabled] && ![self isCruisingEnabled]) {
    return;
  }
  const double rate = std::max(1.0, std::round(_roamUpdateRateControl.doubleValue));
  _lastRoamTick = std::chrono::steady_clock::now();
  _roamTimer = [NSTimer timerWithTimeInterval:1.0 / rate
                                       target:self
                                     selector:@selector(roamTimerFired:)
                                     userInfo:nil
                                      repeats:YES];
  [NSRunLoop.mainRunLoop addTimer:_roamTimer forMode:NSRunLoopCommonModes];
}

- (void)roamTimerFired:(NSTimer *)timer {
  (void)timer;
  const auto now = std::chrono::steady_clock::now();
  const double requestedRate = std::max(1.0, std::round(_roamUpdateRateControl.doubleValue));
  const double elapsed = std::chrono::duration<double>(now - _lastRoamTick).count();
  _lastRoamTick = now;
  const bool roaming = [self isRoamingEnabled];
  const bool cruising = [self isCruisingEnabled];
  if ((_viewerPaused && !_cruiseRecovery) || (!roaming && !cruising) || !_window.isKeyWindow ||
      (roaming && _window.firstResponder != _panoramaView)) {
    [self clearRoamKeys];
    return;
  }

  // Do not turn a temporarily stalled main run loop into a large jump.
  const double dt = std::min(elapsed, 2.0 / requestedRate);
  if (_cruiseRecovery) {
    return;
  }

  const bool aircraft = cruising && [self isAircraftDynamicsEnabled];
  if (aircraft) {
    const double sensitivity = _roamMouseSensitivityControl.doubleValue;
    const double direction = _invertMousePanning ? -1.0 : 1.0;
    const double bankCommand =
        _cruiseSteeringActive
            ? std::clamp(
                  cruise_steering_response(_cruiseSteeringX) * sensitivity * direction,
                  -1.0,
                  1.0
              )
            : 0.0;
    const double targetBank = bankCommand * panorama::app::kAircraftMaximumBank;
    const double bankResponse = 1.0 - std::exp(-dt / panorama::app::kAircraftBankResponseSeconds);
    _aircraftBank += (targetBank - _aircraftBank) * bankResponse;

    const double pitchDelta = _cruiseSteeringActive ? cruise_steering_response(_cruiseSteeringY) *
                                                          panorama::app::kCruiseMaximumPitchRate *
                                                          sensitivity * direction * dt
                                                    : 0.0;
    constexpr double kPitchLimit = 60.0 * panorama::app::kDegreesToRadians;
    const double nextPitch = std::clamp(_orientation.pitch + pitchDelta, -kPitchLimit, kPitchLimit);
    const double horizontalSpeed = std::max(
        panorama::app::kMinimumMovementSpeed,
        _aircraftAirspeed * std::max(0.2, std::cos(nextPitch))
    );
    const double turnRate = std::clamp(
        panorama::app::kGravity * std::tan(_aircraftBank) / horizontalSpeed,
        -panorama::app::kCruiseMaximumYawRate,
        panorama::app::kCruiseMaximumYawRate
    );
    const double headingDelta = turnRate * dt;
    const bool attitudeChanged = std::abs(headingDelta) > 1e-12 || std::abs(pitchDelta) > 1e-12 ||
                                 std::abs(_orientation.roll + _aircraftBank) > 1e-12;
    _orientation.heading =
        std::remainder(_orientation.heading + headingDelta, 2.0 * std::numbers::pi);
    _orientation.pitch = nextPitch;
    // CameraOrientation's positive roll tilts the rendered horizon in the
    // opposite direction to the physical positive bank used above.
    _orientation.roll = -_aircraftBank;
    [self updateCruiseHUD];
    if (attitudeChanged) {
      _renderer->request_view(_orientation, _verticalFieldOfView, _image);
    }
  } else if (cruising && _cruiseSteeringActive) {
    const double sensitivity = _roamMouseSensitivityControl.doubleValue;
    const double direction = _invertMousePanning ? -1.0 : 1.0;
    [self rotateHeading:cruise_steering_response(_cruiseSteeringX) *
                        panorama::app::kCruiseMaximumYawRate * sensitivity * direction * dt
                  pitch:cruise_steering_response(_cruiseSteeringY) *
                        panorama::app::kCruiseMaximumPitchRate * sensitivity * direction * dt];
  }

  double forward = cruising ? 1.0
                            : static_cast<double>(_roamForwardPressed) -
                                  static_cast<double>(_roamBackwardPressed);
  double right =
      cruising ? 0.0
               : static_cast<double>(_roamRightPressed) - static_cast<double>(_roamLeftPressed);
  const double magnitude = std::hypot(forward, right);
  if (!(magnitude > 0.0)) {
    return;
  }
  forward /= magnitude;
  right /= magnitude;
  double movement_speed = cruising ? _cruiseSpeed : _roamSpeed;
  if (aircraft) {
    const double previousAirspeed = _aircraftAirspeed;
    const double trimAcceleration =
        (_cruiseSpeed - _aircraftAirspeed) / panorama::app::kAircraftTrimResponseSeconds;
    const double gravityAcceleration = -panorama::app::kGravity * std::sin(_orientation.pitch);
    _aircraftAirspeed = std::clamp(
        _aircraftAirspeed + (trimAcceleration + gravityAcceleration) * dt,
        panorama::app::kMinimumMovementSpeed,
        panorama::app::kMaximumCruiseSpeed
    );
    movement_speed = 0.5 * (previousAirspeed + _aircraftAirspeed);
  }
  const double distance = movement_speed * dt;
  const double heading = _orientation.heading;
  const bool flight = cruising && _roamAltitudeModeControl.selectedSegment == 1;
  const double horizontal_distance = flight ? distance * std::cos(_orientation.pitch) : distance;
  _roamDesiredPosition.easting +=
      horizontal_distance * (forward * std::sin(heading) + right * std::cos(heading));
  _roamDesiredPosition.northing +=
      horizontal_distance * (forward * std::cos(heading) - right * std::sin(heading));
  if (flight) {
    _roamAltitude += distance * std::sin(_orientation.pitch);
    if (_groundClearanceControl.currentEditor == nil) {
      _groundClearanceControl.stringValue = [NSString stringWithFormat:@"%.1f", _roamAltitude];
    }
  }
  const panorama::app::RoamAltitudeMode altitudeMode =
      _roamAltitudeModeControl.selectedSegment == 0 ? panorama::app::RoamAltitudeMode::FollowTerrain
                                                    : panorama::app::RoamAltitudeMode::HoldAltitude;
  const double height = altitudeMode == panorama::app::RoamAltitudeMode::FollowTerrain
                            ? _groundClearance
                            : _roamAltitude;
  // TODO: validate the complete movement segment rather than only its endpoint.
  // At very high Cruise speeds a single step could otherwise pass through a
  // narrow terrain ridge before the worker samples the destination elevation.
  _roamRequestToken = _renderer->request_roam(_roamDesiredPosition, altitudeMode, height);
  if (cruising) {
    [self updateMovementStatus];
    [self updateMiniMapTelemetry];
  } else {
    _roamStatusLabel.stringValue = @"Moving…";
    _roamStatusLabel.textColor = NSColor.secondaryLabelColor;
  }
}

- (void)updateMovementStatus {
  if (_roamStatusLabel == nil) {
    return;
  }
  if (_cruiseRecovery) {
    _roamStatusLabel.stringValue = @"Blocked • drag to steer/climb • Space resumes";
  } else if (_viewerPaused) {
    _roamStatusLabel.stringValue = @"Paused • Space resumes";
  } else if ([self isCruisingEnabled]) {
    _roamStatusLabel.stringValue = [self isAircraftDynamicsEnabled]
                                       ? @"Aircraft • mouse banks/pitches • W/S trim speed"
                                       : @"Cruising • W/S speed • Space pauses";
  } else {
    _roamStatusLabel.stringValue =
        [self isMouseTurningEnabled] ? @"WASD move • mouse looks" : @"WASD move • arrow keys look";
  }
  _roamStatusLabel.textColor = NSColor.secondaryLabelColor;
}

- (void)updateRoamControls {
  const BOOL roaming = [self isRoamingEnabled];
  const BOOL cruising = [self isCruisingEnabled];
  const BOOL moving = roaming || cruising;
  for (NSView *row in _roamRows) {
    row.hidden = !moving;
  }
  _roamTurningModeRow.hidden = !roaming;
  _aircraftDynamicsRow.hidden = !cruising;
  _roamMouseSensitivityRow.hidden = !cruising && ![self isMouseTurningEnabled];
  _roamMouseSensitivityControl.toolTip =
      [self isAircraftDynamicsEnabled]
          ? @"Aircraft bank and pitch sensitivity"
          : (cruising ? @"Maximum Cruise yaw and pitch rate" : @"Mouse turning sensitivity");
  [_roamAltitudeModeControl setLabel:cruising ? @"Flight" : @"Altitude" forSegment:1];
  _roamAltitudeModeControl.toolTip =
      cruising ? @"Maintain height above terrain, or use pitch to climb and descend in Flight mode"
               : @"Maintain height above terrain or hold absolute elevation while moving";
  const BOOL holdAltitude = moving && _roamAltitudeModeControl.selectedSegment == 1;
  _observerHeightLabel.stringValue = holdAltitude ? @"Altitude" : @"Eye height";
  _observerHeightUnit.stringValue = holdAltitude ? @"m AMSL" : @"m AGL";
  _observerHeightLabel.toolTip = holdAltitude
                                     ? @"Observer elevation above mean sea level"
                                     : @"Observer height above the terrain directly beneath it";
  _observerHeightUnit.toolTip =
      holdAltitude ? @"Metres above mean sea level" : @"Metres above ground level";
  _groundClearanceControl.toolTip = holdAltitude
                                        ? @"Observer elevation above mean sea level"
                                        : @"Observer height above the terrain directly beneath it";
  _groundClearanceDecreaseControl.toolTip =
      holdAltitude ? @"Lower altitude by 1 m (Option: 0.1 m; Shift: 10 m)"
                   : @"Lower eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  _groundClearanceIncreaseControl.toolTip =
      holdAltitude ? @"Raise altitude by 1 m (Option: 0.1 m; Shift: 10 m)"
                   : @"Raise eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  _groundClearanceControl.doubleValue = holdAltitude ? _roamAltitude : _groundClearance;
  _groundClearanceControl.stringValue =
      [NSString stringWithFormat:@"%.1f", _groundClearanceControl.doubleValue];
  [self updateMovementSpeedControl];
  [_panoramaView setMouseTurningEnabled:[self isMouseTurningEnabled] && !_viewerPaused];
  [_panoramaView setCruiseSteeringEnabled:cruising && !_viewerPaused];
  [self updateCruiseHUD];
  if (![self hasPressedRoamKey]) {
    [self updateMovementStatus];
  }
  [self updateMiniMapTelemetry];
}

- (void)updateCruiseHUD {
  [_panoramaView setCruiseHUDHeading:_orientation.heading
                               pitch:_orientation.pitch
                                bank:_aircraftBank
                 verticalFieldOfView:_verticalFieldOfView
                        aircraftMode:[self isAircraftDynamicsEnabled]];
}

- (void)movementModeChanged:(id)sender {
  (void)sender;
  [self clearRoamKeys];
  [self clearCruiseSteering];
  [self levelAircraft];
  if (_cruiseRecovery && ![self isCruisingEnabled]) {
    _cruiseRecovery = false;
    [_panoramaView setViewerPaused:_viewerPaused recoveryMessage:nil];
  }
  _roamDesiredPosition = {_observer.easting, _observer.northing};
  _roamAltitude = _observer.elevation;
  if ([self isCruisingEnabled]) {
    // Cruise is potentially fast and begins under continuous input. Enter it
    // in the safer absolute-altitude mode and require an explicit resume.
    _roamAltitudeModeControl.selectedSegment = 1;
    if ([self isAircraftDynamicsEnabled]) {
      _aircraftAirspeed = _cruiseSpeed;
    }
    _viewerPaused = true;
    _cruiseRecovery = false;
    [_panoramaView setViewerPaused:true recoveryMessage:nil];
  }
  [self updateRoamControls];
  [self scheduleRoamTimer];
  if ([self isCruisingEnabled]) {
    [_window makeFirstResponder:_panoramaView];
  }
}

- (void)roamTurningModeChanged:(id)sender {
  (void)sender;
  [self updateRoamControls];
}

- (void)roamMouseSensitivityChanged:(id)sender {
  (void)sender;
  _roamMouseSensitivityLabel.stringValue =
      [NSString stringWithFormat:@"%.2f×", _roamMouseSensitivityControl.doubleValue];
}

- (void)roamAltitudeModeChanged:(id)sender {
  (void)sender;
  [self clearRoamKeys];
  if (_roamAltitudeModeControl.selectedSegment == 0 && [self isAircraftDynamicsEnabled]) {
    _cruiseSpeed = _aircraftAirspeed;
    _aircraftDynamicsControl.state = NSControlStateValueOff;
    [self levelAircraft];
  }
  _roamDesiredPosition = {_observer.easting, _observer.northing};
  if (_roamAltitudeModeControl.selectedSegment == 1) {
    _roamAltitude = _observer.elevation;
  }
  [self updateRoamControls];
}

- (void)aircraftDynamicsChanged:(id)sender {
  (void)sender;
  if (_aircraftDynamicsControl.state == NSControlStateValueOn) {
    // Coordinated flight uses pitch to change absolute altitude and is not
    // compatible with the terrain-hugging Cruise mode.
    _roamAltitudeModeControl.selectedSegment = 1;
    _roamAltitude = _observer.elevation;
    _aircraftAirspeed = _cruiseSpeed;
    _aircraftBank = 0.0;
  } else {
    _cruiseSpeed = _aircraftAirspeed;
    [self levelAircraft];
  }
  _roamDesiredPosition = {_observer.easting, _observer.northing};
  [self updateRoamControls];
}

- (void)updateMovementSpeedControl {
  if (_roamSpeedControl == nil) {
    return;
  }
  if ([self isCruisingEnabled]) {
    const BOOL aircraft = [self isAircraftDynamicsEnabled];
    _roamSpeedControl.minValue = std::log10(panorama::app::kMinimumMovementSpeed);
    _roamSpeedControl.maxValue = std::log10(panorama::app::kMaximumCruiseSpeed);
    _roamSpeedControl.doubleValue = std::log10(_cruiseSpeed);
    _roamSpeedRowLabel.stringValue = aircraft ? @"Trim speed" : @"Speed";
    _roamSpeedControl.toolTip =
        aircraft ? @"Logarithmic target airspeed; climbs and dives change the actual airspeed"
                 : @"Logarithmic cruise speed from 3.6 km/h to 36,000 km/h";
    _roamSpeedLabel.stringValue = format_movement_speed(_cruiseSpeed);
  } else {
    _roamSpeedRowLabel.stringValue = @"Speed";
    _roamSpeedControl.minValue = panorama::app::kMinimumMovementSpeed;
    _roamSpeedControl.maxValue = panorama::app::kMaximumRoamSpeed;
    _roamSpeedControl.doubleValue = _roamSpeed;
    _roamSpeedControl.toolTip = @"Horizontal roaming speed";
    _roamSpeedLabel.stringValue = format_movement_speed(_roamSpeed);
  }
}

- (void)roamSpeedChanged:(id)sender {
  (void)sender;
  if ([self isCruisingEnabled]) {
    _cruiseSpeed = std::pow(10.0, _roamSpeedControl.doubleValue);
    [self updateMovementStatus];
  } else {
    _roamSpeed = _roamSpeedControl.doubleValue;
  }
  _roamSpeedLabel.stringValue =
      format_movement_speed([self isCruisingEnabled] ? _cruiseSpeed : _roamSpeed);
  [self updateMiniMapTelemetry];
}

- (void)roamUpdateRateChanged:(id)sender {
  (void)sender;
  _roamUpdateRateControl.doubleValue = std::round(_roamUpdateRateControl.doubleValue);
  _roamUpdateRateLabel.stringValue =
      [NSString stringWithFormat:@"%.0f Hz", _roamUpdateRateControl.doubleValue];
  [self scheduleRoamTimer];
}

- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta {
  _orientation.heading =
      std::remainder(_orientation.heading + headingDelta, 2.0 * std::numbers::pi);
  constexpr double kPitchLimit = 85.0 * std::numbers::pi / 180.0;
  _orientation.pitch = std::clamp(_orientation.pitch + pitchDelta, -kPitchLimit, kPitchLimit);
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  [self updateCruiseHUD];
  [self updateMiniMapTelemetry];
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

- (void)mouseTurnForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta {
  const double direction = _invertMousePanning ? -1.0 : 1.0;
  const double zoom_scale = _verticalFieldOfView / panorama::app::kDefaultVerticalFieldOfView;
  const double sensitivity =
      _roamMouseSensitivityControl == nil ? 1.0 : _roamMouseSensitivityControl.doubleValue;
  [self rotateHeading:headingDelta * direction * zoom_scale * sensitivity
                pitch:pitchDelta * direction * zoom_scale * sensitivity];
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
  [self updateCruiseHUD];
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

- (void)setDaylightStatus:(NSString *)status {
  _daylightTimesLabel.stringValue = status;
  _daylightSymbolsRow.hidden = YES;
}

- (void)resolveObserverTimeZone {
  const uint64_t requestToken = ++_timeZoneRequestToken;
  _observerTimeZone = nil;
  _timeZoneLookupInProgress = true;
  [self setDaylightStatus:@"Finding observer time zone…"];
  [self updateSettingsControlAvailability];

  [_timeZoneRequest cancel];
  const panorama::LatLon geographic =
      _renderer->terrain_crs().to_lat_lon({_observer.easting, _observer.northing});
  CLLocation *location = [[CLLocation alloc] initWithLatitude:geographic.lat
                                                    longitude:geographic.lon];
  _timeZoneRequest = [[MKReverseGeocodingRequest alloc] initWithLocation:location];
  __weak PanoramaController *weakSelf = self;
  [_timeZoneRequest
      getMapItemsWithCompletionHandler:^(NSArray<MKMapItem *> *mapItems, NSError *error) {
        // Geocoding may finish after another observer move. Marshal UI
        // work to the main queue and discard superseded responses.
        dispatch_async(dispatch_get_main_queue(), ^{
          PanoramaController *strongSelf = weakSelf;
          if (strongSelf == nil || requestToken != strongSelf->_timeZoneRequestToken) {
            return;
          }
          strongSelf->_timeZoneLookupInProgress = false;
          NSTimeZone *timeZone = mapItems.firstObject.timeZone;
          if (error != nil || timeZone == nil) {
            [strongSelf setDaylightStatus:@"Observer time zone unavailable"];
            [strongSelf updateSettingsControlAvailability];
            return;
          }

          strongSelf->_observerTimeZone = timeZone;
          strongSelf->_astronomicalTimeControl.toolTip = [NSString
              stringWithFormat:@"Local time in %@ at one-minute resolution", timeZone.name];
          strongSelf->_daylightTimesLabel.toolTip =
              [NSString stringWithFormat:@"Local geometric-horizon crossings in %@", timeZone.name];

          // Populate the initial controls with the current civil time at
          // the observer. Later observer moves preserve the user's chosen
          // wall-clock date and time, but reinterpret them at the new site.
          if (!strongSelf->_astronomicalControlsUseObserverTime) {
            NSDate *now = [NSDate date];
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            dateFormatter.locale = [[NSLocale alloc] initWithLocaleIdentifier:@"en_US_POSIX"];
            dateFormatter.timeZone = timeZone;
            dateFormatter.dateFormat = @"dd-MM-yyyy";
            strongSelf->_astronomicalDateControl.stringValue = [dateFormatter stringFromDate:now];

            NSCalendar *calendar =
                [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
            calendar.timeZone = timeZone;
            NSDateComponents *components =
                [calendar components:NSCalendarUnitHour | NSCalendarUnitMinute fromDate:now];
            const double minutes = static_cast<double>(components.hour * 60 + components.minute);
            strongSelf->_astronomicalTimeControl.doubleValue = minutes;
            strongSelf->_astronomicalTimeLabel.stringValue =
                panorama::app::format_clock_minutes(minutes);
            strongSelf->_astronomicalControlsUseObserverTime = true;
          }

          [strongSelf updateSettingsControlAvailability];
          if (strongSelf->_sunModeControl.selectedSegment == 1) {
            [strongSelf publishAstronomicalLighting];
          } else {
            [strongSelf setDaylightStatus:[NSString stringWithFormat:@"Observer time · %@",
                                                                     timeZone.abbreviation]];
          }
        });
      }];
}

/// Publish the astronomical direction as grid azimuth and altitude without
/// retracing the terrain.
- (BOOL)publishAstronomicalLighting {
  if (_observerTimeZone == nil) {
    [self setDaylightStatus:_timeZoneLookupInProgress ? @"Finding observer time zone…"
                                                      : @"Observer time zone unavailable"];
    return NO;
  }

  const char *date = _astronomicalDateControl.stringValue.UTF8String;
  NSString *timeValue = panorama::app::format_clock_minutes(_astronomicalTimeControl.doubleValue);
  const char *time = timeValue.UTF8String;
  const std::optional<panorama::app::CalendarDateTime> local = panorama::app::parse_date_time(
      date == nullptr ? std::string_view{} : std::string_view(date),
      time == nullptr ? std::string_view{} : std::string_view(time)
  );
  _astronomicalDateControl.textColor =
      local.has_value() ? NSColor.labelColor : NSColor.systemRedColor;
  if (!local.has_value()) {
    [self setDaylightStatus:@"Enter a valid DD-MM-YYYY date"];
    return NO;
  }
  const std::optional<panorama::app::CalendarDateTime> utc =
      panorama::app::local_date_time_to_utc(*local, _observerTimeZone);
  if (!utc.has_value()) {
    [self setDaylightStatus:@"This local time does not exist"];
    return NO;
  }

  const panorama::app::DaylightTimes daylightInfo = panorama::app::daylight_times(
      _renderer->terrain_crs(),
      {_observer.easting, _observer.northing},
      *local
  );
  NSString *timeZoneSummary =
      panorama::app::format_time_zone_summary(_observerTimeZone, utc.value());
  switch (daylightInfo.state) {
  case panorama::app::DaylightState::Normal:
    _daylightTimesLabel.stringValue = timeZoneSummary;
    _sunriseTimeLabel.stringValue = panorama::app::format_local_daylight_time(
        *local,
        daylightInfo.sunrise_minutes,
        _observerTimeZone
    );
    _sunsetTimeLabel.stringValue = panorama::app::format_local_daylight_time(
        *local,
        daylightInfo.sunset_minutes,
        _observerTimeZone
    );
    _daylightSymbolsRow.hidden = NO;
    break;
  case panorama::app::DaylightState::PolarDay:
    [self setDaylightStatus:[NSString stringWithFormat:@"%@\nSun above horizon all day",
                                                       timeZoneSummary]];
    break;
  case panorama::app::DaylightState::PolarNight:
    [self setDaylightStatus:[NSString stringWithFormat:@"%@\nSun below horizon all day",
                                                       timeZoneSummary]];
    break;
  }

  const panorama::app::SolarPosition sun = panorama::app::solar_position(
      _renderer->terrain_crs(),
      {_observer.easting, _observer.northing},
      utc.value()
  );
  const double azimuthDegrees = sun.azimuth * panorama::app::kRadiansToDegrees;
  const double altitudeDegrees = sun.elevation * panorama::app::kRadiansToDegrees;
  _sunAzimuthControl.doubleValue = azimuthDegrees;
  _sunAltitudeControl.doubleValue = altitudeDegrees;
  _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.1f°", azimuthDegrees];
  _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.1f°", altitudeDegrees];
  _presentation.appearance.sun_azimuth = sun.azimuth;
  _presentation.appearance.sun_elevation = sun.elevation;
  _renderer->request_presentation(_presentation);
  return YES;
}

- (void)publishLightingControls {
  _presentation.appearance.sun_azimuth =
      _sunAzimuthControl.doubleValue * panorama::app::kDegreesToRadians;
  _presentation.appearance.sun_elevation =
      _sunAltitudeControl.doubleValue * panorama::app::kDegreesToRadians;
  _presentation.appearance.diffusivity = static_cast<float>(_diffusivityControl.doubleValue);
  _presentation.appearance.ambient_light = static_cast<float>(_skyStrengthControl.doubleValue);
  _presentation.appearance.ambient_detail = static_cast<float>(_skyDetailControl.doubleValue);
  _renderer->request_presentation(_presentation);
}

- (void)sunModeChanged:(NSSegmentedControl *)sender {
  [self updateSettingsControlAvailability];
  if (sender.selectedSegment == 1) {
    _manualSunAzimuthDegrees = _sunAzimuthControl.doubleValue;
    _manualSunAltitudeDegrees = _sunAltitudeControl.doubleValue;
    if (_observerTimeZone == nil) {
      if (!_timeZoneLookupInProgress) {
        [self resolveObserverTimeZone];
      }
    } else if (![self publishAstronomicalLighting]) {
      NSBeep();
    }
  } else {
    _sunAzimuthControl.doubleValue = _manualSunAzimuthDegrees;
    _sunAltitudeControl.doubleValue = _manualSunAltitudeDegrees;
    _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.0f°", _manualSunAzimuthDegrees];
    _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.0f°", _manualSunAltitudeDegrees];
    [self publishLightingControls];
  }
}

- (void)astronomicalInputChanged:(NSTextField *)sender {
  (void)sender;
  if (_sunModeControl.selectedSegment == 1 && ![self publishAstronomicalLighting]) {
    NSBeep();
  }
}

- (void)astronomicalTimeChanged:(NSSlider *)sender {
  sender.doubleValue = std::round(sender.doubleValue);
  _astronomicalTimeLabel.stringValue = panorama::app::format_clock_minutes(sender.doubleValue);
  [self updateSettingsControlAvailability];
  if (_sunModeControl.selectedSegment == 1) {
    [self publishAstronomicalLighting];
  }
}

- (void)adjustAstronomicalTime:(NSButton *)sender {
  const double minutes = std::clamp(
      std::round(_astronomicalTimeControl.doubleValue) + static_cast<double>(sender.tag),
      _astronomicalTimeControl.minValue,
      _astronomicalTimeControl.maxValue
  );
  _astronomicalTimeControl.doubleValue = minutes;
  _astronomicalTimeLabel.stringValue = panorama::app::format_clock_minutes(minutes);
  [self updateSettingsControlAvailability];
  if (_sunModeControl.selectedSegment == 1) {
    [self publishAstronomicalLighting];
  }
}

- (void)sunAzimuthChanged:(NSSlider *)sender {
  constexpr double kDetentRadiusDegrees = 3.0;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSunAzimuthDegrees) <=
      kDetentRadiusDegrees) {
    sender.doubleValue = panorama::app::kDefaultSunAzimuthDegrees;
  }
  _manualSunAzimuthDegrees = sender.doubleValue;
  _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.0f°", sender.doubleValue];
  [self publishLightingControls];
}

- (void)sunAltitudeChanged:(NSSlider *)sender {
  constexpr double kDetentRadiusDegrees = 2.0;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSunAltitudeDegrees) <=
      kDetentRadiusDegrees) {
    sender.doubleValue = panorama::app::kDefaultSunAltitudeDegrees;
  }
  _manualSunAltitudeDegrees = sender.doubleValue;
  _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.0f°", sender.doubleValue];
  [self publishLightingControls];
}

- (void)diffusivityChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultDiffusivity) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultDiffusivity;
  }
  _diffusivityLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)skyStrengthChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSkyStrength) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultSkyStrength;
  }
  _skyStrengthLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)skyDetailChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSkyDetail) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultSkyDetail;
  }
  _skyDetailLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)normalLightingChanged:(NSButton *)sender {
  _presentation.use_surface_normals = sender.state == NSControlStateValueOn;
  [self updateSettingsControlAvailability];
  _renderer->request_presentation(_presentation);
}

- (void)raytracedShadowsChanged:(NSButton *)sender {
  _presentation.appearance.raytraced_shadows = sender.state == NSControlStateValueOn;
  _renderer->request_presentation(_presentation);
}

/// Feature outlines are presentation-only, like lighting, so both controls
/// update the current trace immediately.
- (void)publishFeatureOutlineControls {
  _presentation.appearance.feature_outlines =
      _featureOutlinesControl.state == NSControlStateValueOn;
  _presentation.appearance.feature_outline_detail =
      static_cast<float>(_featureOutlineDetailControl.doubleValue / 10.0);
  _renderer->request_presentation(_presentation);
}

- (void)featureOutlinesChanged:(NSButton *)sender {
  (void)sender;
  [self updateSettingsControlAvailability];
  [self publishFeatureOutlineControls];
}

- (void)featureOutlineDetailChanged:(NSSlider *)sender {
  sender.doubleValue = std::round(sender.doubleValue);
  _featureOutlineDetailLabel.stringValue = [NSString stringWithFormat:@"%.0f", sender.doubleValue];
  [self publishFeatureOutlineControls];
}

/// LOD is a trace setting: zero retains LOD 1 everywhere, while positive
/// values permit a tile representation no wider than this pixel-footprint
/// multiplier.
- (void)lodScaleChanged:(NSSlider *)sender {
  const double scale = sender.doubleValue;
  _lodScaleLabel.stringValue = scale == 0.0 ? @"Off" : [NSString stringWithFormat:@"%.1f×", scale];
  _renderer->request_lod_scale(static_cast<float>(scale));
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
/// paired field changes immediately; editing completion commits both values.
- (void)controlTextDidChange:(NSNotification *)notification {
  NSTextField *changed = notification.object;
  if (changed == _coordinateInputControl) {
    [self updateCoordinateInputValidation];
    return;
  }
  if (_updatingResolutionControls || _aspectLockControl.state != NSControlStateValueOn) {
    return;
  }
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

- (void)controlTextDidEndEditing:(NSNotification *)notification {
  NSTextField *field = notification.object;
  if (field == _imageWidthControl || field == _imageHeightControl) {
    [self commitResolutionControls];
  } else if (field == _minimumControl || field == _maximumControl) {
    [self publishTerrainControls];
  } else if (field == _astronomicalDateControl) {
    [self astronomicalInputChanged:field];
  } else if (field == _groundClearanceControl) {
    [self commitGroundClearanceControl];
  } else if (field == _coordinateInputControl) {
    [self updateCoordinateInputValidation];
  }
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
  [self commitResolutionControls];
}

- (void)zoomWithScrollDelta:(double)delta precise:(bool)precise {
  if (_viewerPaused) {
    return;
  }
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
             aspectFitView:(AspectFitContainerView *)aspectFitView
              miniMapPanel:(MiniMapPanelView *)miniMapPanel {
  _panoramaView = panoramaView;
  _overlayView = overlayView;
  _aspectFitView = aspectFitView;
  _miniMapPanel = miniMapPanel;
  _miniMapPanel.interactionDelegate = self;

  _pointInspectionEnabled = true;
  [_panoramaView setPointInspectionEnabled:true];
  [_panoramaView setMouseTurningEnabled:[self isMouseTurningEnabled] && !_viewerPaused];
  [_panoramaView setCruiseSteeringEnabled:[self isCruisingEnabled] && !_viewerPaused];
  [self updateCruiseHUD];
  [_panoramaView setViewerPaused:_viewerPaused recoveryMessage:nil];
  [_overlayView setMapAndPointInfoVisible:true];
  [self updateMiniMapTelemetry];
  [_miniMapPanel informationFooterContentDidChange];
}

- (void)inspectPixelX:(uint32_t)x y:(uint32_t)y {
  if (_pointerOwner == panorama::app::PointerOwner::Panorama && _pointInspectionEnabled &&
      !_pointInspectionLocked && !_pointLockPending) {
    _inspectionRequestToken = _renderer->request_inspection(panorama::app::InspectionPixel{x, y});
  }
}

- (void)invalidatePanoramaHover {
  _inspectionRequestToken = _renderer->request_inspection(std::nullopt);
  if (!_pointInspectionLocked && !_pointLockPending) {
    [self updatePointInfo:std::nullopt];
  }
}

- (void)clearTargetVisibility {
  _targetVisibilityRequestToken = _renderer->request_target_visibility(std::nullopt);
  _targetVisibilityRevision = 0U;
  _lockedPointOccluded = false;
}

- (void)requestTargetVisibilityForPoint:(panorama::app::TerrainPoint)point {
  _targetVisibilityRevision = 0U;
  _lockedPointOccluded = false;
  _targetVisibilityRequestToken = _renderer->request_target_visibility(point);
}

- (void)clearMapHover {
  if (_mapPointAction == panorama::app::MapPointAction::Hover) {
    _mapPointAction = panorama::app::MapPointAction::None;
  }
  _mapHoverPoint.reset();
  if (!_pointInspectionLocked && !_pointLockPending) {
    [self updatePointInfo:std::nullopt];
  }
  [self updateLockedPointIndicatorWithOrientation:_orientation
                              verticalFieldOfView:_verticalFieldOfView
                                            image:_image];
}

- (void)pointerMovedOverPanorama {
  if (_pointerOwner == panorama::app::PointerOwner::Panorama) {
    return;
  }
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    [self clearMapHover];
  }
  _pointerOwner = panorama::app::PointerOwner::Panorama;
}

- (void)beginMinimapPointerOwnership {
  [self clearCruiseSteering];
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    return;
  }
  [self invalidatePanoramaHover];
  _pointerOwner = panorama::app::PointerOwner::Minimap;
}

- (void)pointerMovedOverOccludingView:(NSView *)view {
  [self clearCruiseSteering];
  const bool minimap = view == _miniMapPanel || [view isDescendantOf:_miniMapPanel];
  if (minimap) {
    [self beginMinimapPointerOwnership];
    return;
  }
  if (_pointerOwner == panorama::app::PointerOwner::Overlay) {
    return;
  }
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    [self clearMapHover];
  }
  [self invalidatePanoramaHover];
  _pointerOwner = panorama::app::PointerOwner::Overlay;
}

- (void)panoramaPointerExited {
  [self clearCruiseSteering];
  if (_pointerOwner != panorama::app::PointerOwner::Panorama) {
    return;
  }
  _pointerOwner = panorama::app::PointerOwner::None;
  [self invalidatePanoramaHover];
}

- (void)togglePointLockAtPixelX:(uint32_t)x y:(uint32_t)y {
  if (!_pointInspectionEnabled) {
    return;
  }
  if (_pointInspectionLocked || _pointLockPending) {
    _pointInspectionLocked = false;
    _pointLockPending = false;
    _lockedPoint.reset();
    [self clearTargetVisibility];
    [_miniMapPanel clearInspectedPoint];
    [_panoramaView setTerrainPointIndicator:std::nullopt locked:true occluded:false];
    [self setPointInfoStatus:@""];
    _inspectionRequestToken = _renderer->request_inspection(panorama::app::InspectionPixel{x, y});
    return;
  }

  _pointLockPending = true;
  [self setPointInfoStatus:@"Locking point…"];
  _pointLockRequestToken = _renderer->request_inspection(panorama::app::InspectionPixel{x, y});
}

- (void)miniMapPanel:(MiniMapPanelView *)panel
     didHoverEasting:(double)easting
            northing:(double)northing {
  (void)panel;
  [self beginMinimapPointerOwnership];
  if (!_pointInspectionLocked && !_pointLockPending &&
      (_mapPointAction == panorama::app::MapPointAction::None ||
       _mapPointAction == panorama::app::MapPointAction::Hover)) {
    [self requestMapPointEasting:easting
                        northing:northing
                          action:panorama::app::MapPointAction::Hover];
  }
}

- (void)miniMapPanelDidEndHover:(MiniMapPanelView *)panel {
  (void)panel;
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    _pointerOwner = panorama::app::PointerOwner::None;
  }
  [self clearMapHover];
}

- (void)requestMapPointEasting:(double)easting
                      northing:(double)northing
                        action:(panorama::app::MapPointAction)action {
  if (!_pointInspectionEnabled && action != panorama::app::MapPointAction::MoveObserver) {
    return;
  }
  _mapPointAction = action;
  _mapPointRequestToken = _renderer->request_map_point({easting, northing});
  if (action != panorama::app::MapPointAction::Hover) {
    [self setPointInfoStatus:action == panorama::app::MapPointAction::MoveObserver
                                 ? @"Moving observer…"
                                 : @"Locating point…"];
  }
}

- (void)miniMapPanel:(MiniMapPanelView *)panel
    didSelectEasting:(double)easting
            northing:(double)northing {
  (void)panel;
  _coordinateMovePending = false;
  [self beginMinimapPointerOwnership];
  [self requestMapPointEasting:easting
                      northing:northing
                        action:panorama::app::MapPointAction::Look];
}

- (void)miniMapPanel:(MiniMapPanelView *)panel
    didRequestObserverMoveToEasting:(double)easting
                           northing:(double)northing {
  (void)panel;
  _coordinateMovePending = false;
  [self beginMinimapPointerOwnership];
  [self requestMapPointEasting:easting
                      northing:northing
                        action:panorama::app::MapPointAction::MoveObserver];
}

- (void)moveObserverToTerrainPoint:(panorama::app::TerrainPoint)point {
  [self clearRoamKeys];
  _mapHoverPoint.reset();
  _pointInspectionLocked = false;
  _pointLockPending = false;
  _lockedPoint.reset();
  [self clearTargetVisibility];
  [_miniMapPanel clearInspectedPoint];
  [_panoramaView setTerrainPointIndicator:std::nullopt locked:true occluded:false];
  [self setPointInfoStatus:@"Moving observer…"];
  _renderer->request_observer_at(point, _groundClearance);
}

- (void)moveToLockedPoint:(id)sender {
  (void)sender;
  if (!_pointInspectionLocked || !_lockedPoint.has_value()) {
    NSBeep();
    return;
  }
  const panorama::app::PointInspection point = *_lockedPoint;
  const panorama::app::TerrainPoint target = {
      point.easting,
      point.northing,
      point.elevation,
  };
  [self moveObserverToTerrainPoint:target];
}

- (void)lookAtTerrainPoint:(panorama::app::TerrainPoint)point {
  const double east = point.easting - _observer.easting;
  const double north = point.northing - _observer.northing;
  const double horizontal = std::hypot(east, north);
  if (!(horizontal > 0.0)) {
    return;
  }
  const double apparentUp = point.elevation - _observer.elevation -
                            panorama::kCurvatureCoefficient * horizontal * horizontal;
  _orientation.heading = std::atan2(east, north);
  _orientation.pitch = std::atan2(apparentUp, horizontal);
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  [self updateCruiseHUD];
}

- (void)toggleMapAndPointInspection:(id)sender {
  _pointInspectionEnabled = !_pointInspectionEnabled;
  _pointInspectionLocked = false;
  _pointLockPending = false;
  _pointerOwner = panorama::app::PointerOwner::None;
  _mapPointAction = panorama::app::MapPointAction::None;
  _lockedPoint.reset();
  [self clearTargetVisibility];
  [_miniMapPanel clearInspectedPoint];
  _mapHoverPoint.reset();
  [_panoramaView setTerrainPointIndicator:std::nullopt locked:true occluded:false];
  [_panoramaView setPointInspectionEnabled:_pointInspectionEnabled];
  [_overlayView setMapAndPointInfoVisible:_pointInspectionEnabled];
  if (!_pointInspectionEnabled) {
    [self invalidatePanoramaHover];
  }
  [self updatePointInfo:std::nullopt];

  if ([sender isKindOfClass:NSToolbarItem.class]) {
    NSToolbarItem *item = sender;
    if (@available(macOS 26.0, *)) {
      item.style = _pointInspectionEnabled ? NSToolbarItemStyleProminent : NSToolbarItemStylePlain;
    }
  }
}

- (BOOL)isMapAndPointInspectionEnabled {
  return _pointInspectionEnabled;
}

/// Build the controls shown in the trailing render-settings inspector.
- (NSViewController *)makeSettingsViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSScrollView *scrollView =
      [[NSScrollView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 270.0, 400.0)];
  scrollView.borderType = NSNoBorder;
  scrollView.drawsBackground = NO;
  scrollView.hasHorizontalScroller = NO;
  scrollView.hasVerticalScroller = YES;
  scrollView.autohidesScrollers = YES;
  scrollView.scrollerStyle = NSScrollerStyleOverlay;
  viewController.view = scrollView;

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
  _panningSensitivityControl.toolTip = @"Browse-mode drag and keyboard turning sensitivity";
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
  // Leave comfortable edit padding around common four-digit dimensions such
  // as 1920 and 1024 instead of sizing the fields to their initial values.
  [_imageWidthControl.widthAnchor constraintEqualToConstant:56.0].active = YES;
  [_imageHeightControl.widthAnchor constraintEqualToConstant:56.0].active = YES;
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
  _invertMousePanningControl.title = @"Invert drag direction";
  _invertMousePanningControl.state = NSControlStateValueOff;
  _invertMousePanningControl.target = self;
  _invertMousePanningControl.action = @selector(invertMousePanningChanged:);

  _colourSourceControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourSourceControl addItemsWithTitles:@[ @"None (white)", @"Distance", @"Elevation" ]];
  [_colourSourceControl
      selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colour_source)];

  _colourmapControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourmapControl addItemsWithTitles:@[
    @"Viridis",
    @"Plasma",
    @"Inferno",
    @"Magma",
    @"Cividis",
    @"Turbo",
    @"Viewfinder"
  ]];
  [_colourmapControl selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colourmap)];

  _colourScaleControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourScaleControl
      addItemsWithTitles:@[ @"Linear", @"Logarithmic", @"Square root", @"Quadratic" ]];
  [_colourScaleControl
      selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colour_scale)];

  _minimumControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _minimumControl.stringValue =
      panorama::app::format_range_value(_presentation.colour_range.minimum);
  _minimumControl.delegate = self;

  _maximumControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _maximumControl.stringValue =
      panorama::app::format_range_value(_presentation.colour_range.maximum);
  _maximumControl.delegate = self;

  _featureOutlinesControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _featureOutlinesControl.buttonType = NSButtonTypeSwitch;
  _featureOutlinesControl.title = @"Feature outlines";
  _featureOutlinesControl.state =
      _presentation.appearance.feature_outlines ? NSControlStateValueOn : NSControlStateValueOff;
  _featureOutlinesControl.target = self;
  _featureOutlinesControl.action = @selector(featureOutlinesChanged:);
  _featureOutlinesControl.toolTip = @"Draw black lines at multiscale geometric surface separations";

  const double initialOutlineDetail = 10.0 * _presentation.appearance.feature_outline_detail;
  _featureOutlineDetailControl = [NSSlider sliderWithValue:initialOutlineDetail
                                                  minValue:0.0
                                                  maxValue:10.0
                                                    target:self
                                                    action:@selector(featureOutlineDetailChanged:)];
  _featureOutlineDetailControl.continuous = YES;
  _featureOutlineDetailControl.numberOfTickMarks = 11;
  _featureOutlineDetailControl.allowsTickMarkValuesOnly = YES;
  _featureOutlineDetailControl.toolTip = @"Higher values outline smaller surface separations";
  _featureOutlineDetailLabel =
      [NSTextField labelWithString:[NSString stringWithFormat:@"%.0f", initialOutlineDetail]];
  _featureOutlineDetailLabel.alignment = NSTextAlignmentRight;
  [_featureOutlineDetailLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *featureOutlineDetailSetting = [NSStackView
      stackViewWithViews:@[ _featureOutlineDetailControl, _featureOutlineDetailLabel ]];
  featureOutlineDetailSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  featureOutlineDetailSetting.alignment = NSLayoutAttributeCenterY;
  featureOutlineDetailSetting.spacing = 6.0;

  const double initialLodScale = _renderer->initial_lod_scale();
  _lodScaleControl = [NSSlider sliderWithValue:initialLodScale
                                      minValue:0.0
                                      maxValue:8.0
                                        target:self
                                        action:@selector(lodScaleChanged:)];
  _lodScaleControl.numberOfTickMarks = 17; // 0.0, 0.2, 0.4, ... 8.0
  _lodScaleControl.allowsTickMarkValuesOnly = YES;
  // _lodScaleControl.continuous = YES;
  _lodScaleControl.toolTip =
      @"Use coarser independently stored terrain where a cell is smaller than a pixel";
  _lodScaleLabel =
      [NSTextField labelWithString:initialLodScale == 0.0
                                       ? @"Off"
                                       : [NSString stringWithFormat:@"%.1f×", initialLodScale]];
  _lodScaleLabel.alignment = NSTextAlignmentRight;
  [_lodScaleLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *lodScaleSetting =
      [NSStackView stackViewWithViews:@[ _lodScaleControl, _lodScaleLabel ]];
  lodScaleSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  lodScaleSetting.alignment = NSLayoutAttributeCenterY;
  lodScaleSetting.spacing = 6.0;

  _normalLightingControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _normalLightingControl.buttonType = NSButtonTypeSwitch;
  _normalLightingControl.title = @"Surface shading";
  _normalLightingControl.state =
      _presentation.use_surface_normals ? NSControlStateValueOn : NSControlStateValueOff;
  _normalLightingControl.target = self;
  _normalLightingControl.action = @selector(normalLightingChanged:);

  _raytracedShadowsControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _raytracedShadowsControl.buttonType = NSButtonTypeSwitch;
  _raytracedShadowsControl.title = @"Hard shadows";
  _raytracedShadowsControl.state =
      _presentation.appearance.raytraced_shadows ? NSControlStateValueOn : NSControlStateValueOff;
  _raytracedShadowsControl.target = self;
  _raytracedShadowsControl.action = @selector(raytracedShadowsChanged:);
  _raytracedShadowsControl.toolTip = @"Cast one terrain visibility ray towards the sun";

  _sunModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _sunModeControl.segmentCount = 2;
  [_sunModeControl setLabel:@"Manual" forSegment:0];
  [_sunModeControl setLabel:@"Astronomical" forSegment:1];
  _sunModeControl.selectedSegment = 0;
  _sunModeControl.segmentStyle = NSSegmentStyleRounded;
  _sunModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _sunModeControl.target = self;
  _sunModeControl.action = @selector(sunModeChanged:);

  NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
  dateFormatter.locale = [[NSLocale alloc] initWithLocaleIdentifier:@"en_US_POSIX"];
  dateFormatter.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
  NSDate *now = [NSDate date];
  dateFormatter.dateFormat = @"dd-MM-yyyy";
  _astronomicalDateControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _astronomicalDateControl.stringValue = [dateFormatter stringFromDate:now];
  _astronomicalDateControl.placeholderString = @"DD-MM-YYYY";
  _astronomicalDateControl.delegate = self;
  _astronomicalDateControl.toolTip = @"Gregorian date in DD-MM-YYYY format";

  NSCalendar *utcCalendar =
      [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
  utcCalendar.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
  const NSDateComponents *utcComponents =
      [utcCalendar components:NSCalendarUnitHour | NSCalendarUnitMinute fromDate:now];
  const double initialUtcMinutes =
      60.0 * static_cast<double>(utcComponents.hour) + static_cast<double>(utcComponents.minute);
  _astronomicalTimeControl = [NSSlider sliderWithValue:initialUtcMinutes
                                              minValue:0.0
                                              maxValue:1439.0
                                                target:self
                                                action:@selector(astronomicalTimeChanged:)];
  _astronomicalTimeControl.continuous = YES;
  _astronomicalTimeControl.numberOfTickMarks = 7;
  _astronomicalTimeControl.allowsTickMarkValuesOnly = NO;
  _astronomicalTimeControl.toolTip = @"Observer-local time at one-minute resolution";
  _astronomicalTimeLabel =
      [NSTextField labelWithString:panorama::app::format_clock_minutes(initialUtcMinutes)];
  _astronomicalTimeLabel.alignment = NSTextAlignmentRight;
  [_astronomicalTimeLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  _astronomicalTimeDecreaseControl = [NSButton buttonWithTitle:@"−"
                                                        target:self
                                                        action:@selector(adjustAstronomicalTime:)];
  _astronomicalTimeDecreaseControl.tag = -1;
  _astronomicalTimeDecreaseControl.controlSize = NSControlSizeSmall;
  _astronomicalTimeDecreaseControl.continuous = YES;
  [_astronomicalTimeDecreaseControl setPeriodicDelay:0.4F interval:0.08F];
  _astronomicalTimeDecreaseControl.toolTip = @"Move back one minute";
  [_astronomicalTimeDecreaseControl setAccessibilityLabel:@"Decrease time by one minute"];
  [_astronomicalTimeDecreaseControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  _astronomicalTimeIncreaseControl = [NSButton buttonWithTitle:@"+"
                                                        target:self
                                                        action:@selector(adjustAstronomicalTime:)];
  _astronomicalTimeIncreaseControl.tag = 1;
  _astronomicalTimeIncreaseControl.controlSize = NSControlSizeSmall;
  _astronomicalTimeIncreaseControl.continuous = YES;
  [_astronomicalTimeIncreaseControl setPeriodicDelay:0.4F interval:0.08F];
  _astronomicalTimeIncreaseControl.toolTip = @"Move forward one minute";
  [_astronomicalTimeIncreaseControl setAccessibilityLabel:@"Increase time by one minute"];
  [_astronomicalTimeIncreaseControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  NSStackView *astronomicalTimeSlider = [NSStackView stackViewWithViews:@[
    _astronomicalTimeDecreaseControl,
    _astronomicalTimeControl,
    _astronomicalTimeIncreaseControl,
    _astronomicalTimeLabel,
  ]];
  astronomicalTimeSlider.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  astronomicalTimeSlider.alignment = NSLayoutAttributeCenterY;
  astronomicalTimeSlider.spacing = 4.0;
  _daylightTimesLabel = [NSTextField labelWithString:@"Time zone —"];
  _daylightTimesLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _daylightTimesLabel.textColor = NSColor.secondaryLabelColor;
  _daylightTimesLabel.maximumNumberOfLines = 2;
  _daylightTimesLabel.lineBreakMode = NSLineBreakByClipping;
  _daylightTimesLabel.toolTip = @"Local crossings of the geometric horizon";

  NSImageSymbolConfiguration *daylightSymbolConfiguration =
      [NSImageSymbolConfiguration configurationWithPointSize:NSFont.smallSystemFontSize
                                                      weight:NSFontWeightRegular];
  NSImageView *sunriseIcon = [NSImageView
      imageViewWithImage:[[NSImage imageWithSystemSymbolName:@"sunrise"
                                    accessibilityDescription:@"Sunrise"]
                             imageWithSymbolConfiguration:daylightSymbolConfiguration]];
  sunriseIcon.contentTintColor = NSColor.secondaryLabelColor;
  sunriseIcon.toolTip = @"Sunrise";
  [sunriseIcon setAccessibilityLabel:@"Sunrise"];
  [sunriseIcon.widthAnchor constraintEqualToConstant:15.0].active = YES;
  NSImageView *sunsetIcon = [NSImageView
      imageViewWithImage:[[NSImage imageWithSystemSymbolName:@"sunset"
                                    accessibilityDescription:@"Sunset"]
                             imageWithSymbolConfiguration:daylightSymbolConfiguration]];
  sunsetIcon.contentTintColor = NSColor.secondaryLabelColor;
  sunsetIcon.toolTip = @"Sunset";
  [sunsetIcon setAccessibilityLabel:@"Sunset"];
  [sunsetIcon.widthAnchor constraintEqualToConstant:15.0].active = YES;

  _sunriseTimeLabel = [NSTextField labelWithString:@"—"];
  _sunsetTimeLabel = [NSTextField labelWithString:@"—"];
  NSTextField *daylightSeparator = [NSTextField labelWithString:@"•"];
  for (NSTextField *label in @[ _sunriseTimeLabel, daylightSeparator, _sunsetTimeLabel ]) {
    label.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    label.textColor = NSColor.secondaryLabelColor;
  }
  _daylightSymbolsRow = [NSStackView stackViewWithViews:@[
    sunriseIcon,
    _sunriseTimeLabel,
    daylightSeparator,
    sunsetIcon,
    _sunsetTimeLabel,
  ]];
  _daylightSymbolsRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  _daylightSymbolsRow.alignment = NSLayoutAttributeCenterY;
  _daylightSymbolsRow.spacing = 4.0;

  NSStackView *astronomicalTimeSetting = [NSStackView stackViewWithViews:@[
    astronomicalTimeSlider,
    _daylightTimesLabel,
    _daylightSymbolsRow,
  ]];
  astronomicalTimeSetting.orientation = NSUserInterfaceLayoutOrientationVertical;
  astronomicalTimeSetting.alignment = NSLayoutAttributeLeading;
  astronomicalTimeSetting.spacing = 3.0;

  _sunAzimuthControl = [NSSlider
      sliderWithValue:_presentation.appearance.sun_azimuth * panorama::app::kRadiansToDegrees
             minValue:0.0
             maxValue:360.0
               target:self
               action:@selector(sunAzimuthChanged:)];
  _sunAzimuthControl.continuous = YES;
  _sunAzimuthControl.numberOfTickMarks = 9;
  _sunAzimuthControl.allowsTickMarkValuesOnly = NO;
  _sunAzimuthControl.toolTip = @"Clockwise from grid north";
  _sunAzimuthLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.0f°", _sunAzimuthControl.doubleValue]];
  _sunAzimuthLabel.alignment = NSTextAlignmentRight;
  [_sunAzimuthLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *sunAzimuthSetting =
      [NSStackView stackViewWithViews:@[ _sunAzimuthControl, _sunAzimuthLabel ]];
  sunAzimuthSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  sunAzimuthSetting.alignment = NSLayoutAttributeCenterY;
  sunAzimuthSetting.spacing = 6.0;
  _manualSunAzimuthDegrees = _sunAzimuthControl.doubleValue;

  const double initialAltitude =
      _presentation.appearance.sun_elevation * panorama::app::kRadiansToDegrees;
  _sunAltitudeControl = [NSSlider sliderWithValue:initialAltitude
                                         minValue:-90.0
                                         maxValue:90.0
                                           target:self
                                           action:@selector(sunAltitudeChanged:)];
  _sunAltitudeControl.continuous = YES;
  _sunAltitudeControl.numberOfTickMarks = 7;
  _sunAltitudeControl.allowsTickMarkValuesOnly = NO;
  _sunAltitudeControl.toolTip = @"Degrees above or below the horizon";
  _sunAltitudeLabel =
      [NSTextField labelWithString:[NSString stringWithFormat:@"%.0f°", initialAltitude]];
  _sunAltitudeLabel.alignment = NSTextAlignmentRight;
  [_sunAltitudeLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *sunAltitudeSetting =
      [NSStackView stackViewWithViews:@[ _sunAltitudeControl, _sunAltitudeLabel ]];
  sunAltitudeSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  sunAltitudeSetting.alignment = NSLayoutAttributeCenterY;
  sunAltitudeSetting.spacing = 6.0;
  _manualSunAltitudeDegrees = _sunAltitudeControl.doubleValue;

  _diffusivityControl = [NSSlider sliderWithValue:_presentation.appearance.diffusivity
                                         minValue:0.0
                                         maxValue:1.0
                                           target:self
                                           action:@selector(diffusivityChanged:)];
  _diffusivityControl.continuous = YES;
  _diffusivityControl.numberOfTickMarks = 11;
  _diffusivityControl.allowsTickMarkValuesOnly = NO;
  _diffusivityControl.toolTip = @"Strength of directional diffuse lighting";
  _diffusivityLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.2f", _diffusivityControl.doubleValue]];
  _diffusivityLabel.alignment = NSTextAlignmentRight;
  [_diffusivityLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *diffusivitySetting =
      [NSStackView stackViewWithViews:@[ _diffusivityControl, _diffusivityLabel ]];
  diffusivitySetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  diffusivitySetting.alignment = NSLayoutAttributeCenterY;
  diffusivitySetting.spacing = 6.0;

  _skyStrengthControl = [NSSlider sliderWithValue:_presentation.appearance.ambient_light
                                         minValue:0.0
                                         maxValue:1.0
                                           target:self
                                           action:@selector(skyStrengthChanged:)];
  _skyStrengthControl.continuous = YES;
  _skyStrengthControl.numberOfTickMarks = 11;
  _skyStrengthControl.allowsTickMarkValuesOnly = NO;
  _skyStrengthControl.toolTip = @"Overall strength of diffuse atmospheric light";
  _skyStrengthLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.2f", _skyStrengthControl.doubleValue]];
  _skyStrengthLabel.alignment = NSTextAlignmentRight;
  [_skyStrengthLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *skyStrengthSetting =
      [NSStackView stackViewWithViews:@[ _skyStrengthControl, _skyStrengthLabel ]];
  skyStrengthSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  skyStrengthSetting.alignment = NSLayoutAttributeCenterY;
  skyStrengthSetting.spacing = 6.0;

  _skyDetailControl = [NSSlider sliderWithValue:_presentation.appearance.ambient_detail
                                       minValue:0.0
                                       maxValue:1.0
                                         target:self
                                         action:@selector(skyDetailChanged:)];
  _skyDetailControl.continuous = YES;
  _skyDetailControl.numberOfTickMarks = 11;
  _skyDetailControl.allowsTickMarkValuesOnly = NO;
  _skyDetailControl.toolTip = @"Normal-dependent detail from five sampled sky directions";
  _skyDetailLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.2f", _skyDetailControl.doubleValue]];
  _skyDetailLabel.alignment = NSTextAlignmentRight;
  [_skyDetailLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *skyDetailSetting =
      [NSStackView stackViewWithViews:@[ _skyDetailControl, _skyDetailLabel ]];
  skyDetailSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  skyDetailSetting.alignment = NSLayoutAttributeCenterY;
  skyDetailSetting.spacing = 6.0;

  _colourSourceControl.target = self;
  _colourSourceControl.action = @selector(renderModeChanged:);
  _colourmapControl.target = self;
  _colourmapControl.action = @selector(renderModeChanged:);
  _colourScaleControl.target = self;
  _colourScaleControl.action = @selector(renderModeChanged:);

  auto make_row = [](NSString *title, NSView *control) {
    NSTextField *label = [NSTextField labelWithString:title];
    [label.widthAnchor constraintEqualToConstant:82.0].active = YES;
    // The 300-point panel has 268 points inside its horizontal margins.
    // Keep each row within that width instead of allowing controls to crowd
    // the trailing glass edge.
    [control.widthAnchor constraintGreaterThanOrEqualToConstant:178.0].active = YES;
    NSStackView *row = [NSStackView stackViewWithViews:@[ label, control ]];
    row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row.alignment = NSLayoutAttributeCenterY;
    row.spacing = 8.0;
    return row;
  };

  NSStackView *rangeSetting = [NSStackView stackViewWithViews:@[
    _minimumControl,
    [NSTextField labelWithString:@"–"],
    _maximumControl,
  ]];
  rangeSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  rangeSetting.alignment = NSLayoutAttributeCenterY;
  rangeSetting.spacing = 5.0;
  [_minimumControl.widthAnchor constraintEqualToConstant:80.0].active = YES;
  [_maximumControl.widthAnchor constraintEqualToConstant:80.0].active = YES;

  NSView *colourmapRow = make_row(@"Colourmap", _colourmapControl);
  NSView *colourScaleRow = make_row(@"Scale", _colourScaleControl);
  NSView *colourRangeRow = make_row(@"Range (m)", rangeSetting);
  _scalarColourRows = @[ colourmapRow, colourScaleRow, colourRangeRow ];
  _featureOutlineDetailRow = make_row(@"Detail", featureOutlineDetailSetting);
  NSView *lodScaleRow = make_row(@"LOD scale", lodScaleSetting);

  NSView *sunModeRow = make_row(@"Sun", _sunModeControl);
  NSView *dateRow = make_row(@"Date", _astronomicalDateControl);
  NSView *timeRow = make_row(@"Local time", astronomicalTimeSetting);
  NSView *azimuthRow = make_row(@"Azimuth", sunAzimuthSetting);
  NSView *altitudeRow = make_row(@"Altitude", sunAltitudeSetting);
  NSView *skyStrengthRow = make_row(@"Sky strength", skyStrengthSetting);
  NSView *skyDetailRow = make_row(@"Sky detail", skyDetailSetting);
  NSView *diffusivityRow = make_row(@"Sun strength", diffusivitySetting);
  _manualSunRows = @[ azimuthRow, altitudeRow ];
  _astronomicalSunRows = @[ dateRow, timeRow ];
  _normalLightingRows = @[
    _raytracedShadowsControl,
    sunModeRow,
    dateRow,
    timeRow,
    azimuthRow,
    altitudeRow,
    skyStrengthRow,
    skyDetailRow,
    diffusivityRow,
  ];

  InspectorSectionView *cameraSection =
      [[InspectorSectionView alloc] initWithTitle:@"Camera"
                                         controls:@[
                                           make_row(@"FOV", zoomSetting),
                                           make_row(@"Resolution", resolutionSetting),
                                           _matchWindowControl,
                                           make_row(@"Pan speed", panningSensitivitySetting),
                                           _invertMousePanningControl,
                                         ]
                                      defaultsKey:@"panorama.inspector.camera.expanded"];
  InspectorSectionView *terrainSection =
      [[InspectorSectionView alloc] initWithTitle:@"Terrain"
                                         controls:@[
                                           make_row(@"Colour by", _colourSourceControl),
                                           colourmapRow,
                                           colourScaleRow,
                                           colourRangeRow,
                                           lodScaleRow,
                                           _featureOutlinesControl,
                                           _featureOutlineDetailRow,
                                         ]
                                      defaultsKey:@"panorama.inspector.terrain.expanded"];
  InspectorSectionView *lightingSection =
      [[InspectorSectionView alloc] initWithTitle:@"Lighting"
                                         controls:@[
                                           _normalLightingControl,
                                           _raytracedShadowsControl,
                                           sunModeRow,
                                           dateRow,
                                           timeRow,
                                           azimuthRow,
                                           altitudeRow,
                                           skyStrengthRow,
                                           skyDetailRow,
                                           diffusivityRow,
                                         ]
                                      defaultsKey:@"panorama.inspector.lighting.expanded"];

  NSStackView *settings = [[NSStackView alloc] initWithFrame:NSZeroRect];
  for (NSView *view in @[ heading, cameraSection, terrainSection, lightingSection ]) {
    [settings addArrangedSubview:view];
  }
  settings.orientation = NSUserInterfaceLayoutOrientationVertical;
  settings.alignment = NSLayoutAttributeLeading;
  settings.spacing = 18.0;
  settings.edgeInsets = NSEdgeInsetsMake(20.0, 16.0, 20.0, 16.0);
  settings.translatesAutoresizingMaskIntoConstraints = NO;
  InspectorDocumentView *document = [[InspectorDocumentView alloc] initWithFrame:NSZeroRect];
  document.translatesAutoresizingMaskIntoConstraints = NO;
  [document addSubview:settings];
  scrollView.documentView = document;
  NSLayoutConstraint *viewportHeight =
      [document.heightAnchor constraintEqualToAnchor:scrollView.contentView.heightAnchor];
  // Prefer a viewport-height document when the controls are short. Its lower
  // priority lets the settings grow the document and enable scrolling in a
  // shorter window.
  viewportHeight.priority = NSLayoutPriorityDefaultLow;
  [NSLayoutConstraint activateConstraints:@[
    [document.widthAnchor constraintEqualToAnchor:scrollView.contentView.widthAnchor],
    [document.heightAnchor
        constraintGreaterThanOrEqualToAnchor:scrollView.contentView.heightAnchor],
    viewportHeight,
    [settings.topAnchor constraintEqualToAnchor:document.topAnchor],
    [settings.leadingAnchor constraintEqualToAnchor:document.leadingAnchor],
    [settings.trailingAnchor constraintEqualToAnchor:document.trailingAnchor],
    [settings.bottomAnchor constraintLessThanOrEqualToAnchor:document.bottomAnchor],
  ]];

  [self updateSettingsControlAvailability];
  [self resolveObserverTimeZone];
  return viewController;
}

/// Build observer-position controls separately from camera and presentation
/// settings. This pane is intentionally small for now; roaming controls can be
/// added here without crowding the minimap or the viewer tab.
- (NSViewController *)makePositioningViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSScrollView *scrollView =
      [[NSScrollView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 270.0, 400.0)];
  scrollView.borderType = NSNoBorder;
  scrollView.drawsBackground = NO;
  scrollView.hasHorizontalScroller = NO;
  scrollView.hasVerticalScroller = YES;
  scrollView.autohidesScrollers = YES;
  scrollView.scrollerStyle = NSScrollerStyleOverlay;
  viewController.view = scrollView;

  NSTextField *heading = [NSTextField labelWithString:@"Position & Movement"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _coordinateSystemControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_coordinateSystemControl addItemWithTitle:@"Auto"];
  _coordinateSystemControl.lastItem.tag = -1;
  [_coordinateSystemControl.menu addItem:NSMenuItem.separatorItem];
  [_coordinateSystemControl addItemWithTitle:@"Latitude / longitude"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Wgs84);
  [_coordinateSystemControl addItemWithTitle:@"Swiss LV95"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::SwissLv95);
  [_coordinateSystemControl addItemWithTitle:@"OS National Grid"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::BritishNationalGrid);
  [_coordinateSystemControl.menu addItem:NSMenuItem.separatorItem];
  [_coordinateSystemControl
      addItemWithTitle:[NSString
                           stringWithFormat:@"Dataset grid — %s", _renderer->terrain_crs().name()]];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Terrain);
  _coordinateSystemControl.target = self;
  _coordinateSystemControl.action = @selector(coordinateSystemChanged:);
  _coordinateSystemControl.toolTip =
      @"Auto detects the coordinate system; choose one explicitly to resolve ambiguity";
  // Cap the row at the inspector's 268-point content width. Pop-up buttons use
  // their longest menu item as an intrinsic width; without this constraint the
  // dataset-grid title can force the whole inset stack beyond the panel edge.
  [_coordinateSystemControl.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSTextField *coordinateSystemLabel = [NSTextField labelWithString:@"System"];
  [coordinateSystemLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *coordinateSystemRow =
      [NSStackView stackViewWithViews:@[ coordinateSystemLabel, _coordinateSystemControl ]];
  coordinateSystemRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateSystemRow.alignment = NSLayoutAttributeCenterY;
  coordinateSystemRow.spacing = 8.0;

  _coordinateInputControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _coordinateInputControl.delegate = self;
  _coordinateInputControl.placeholderString = @"Enter or paste a coordinate";
  _coordinateInputControl.toolTip = @"The coordinate system will be detected automatically";
  _coordinateInputControl.target = self;
  _coordinateInputControl.action = @selector(moveToCoordinate:);
  [_coordinateInputControl.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSTextField *coordinateLabel = [NSTextField labelWithString:@"Coordinate"];
  [coordinateLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *coordinateRow =
      [NSStackView stackViewWithViews:@[ coordinateLabel, _coordinateInputControl ]];
  coordinateRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateRow.alignment = NSLayoutAttributeCenterY;
  coordinateRow.spacing = 8.0;

  _coordinateStatusLabel = [NSTextField labelWithString:@"Format will be detected automatically"];
  _coordinateStatusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
  _coordinateStatusLabel.maximumNumberOfLines = 2;
  _coordinateStatusLabel.lineBreakMode = NSLineBreakByWordWrapping;
  [_coordinateStatusLabel.widthAnchor constraintEqualToConstant:268.0].active = YES;

  _coordinateMoveControl = [NSButton buttonWithTitle:@"Move"
                                              target:self
                                              action:@selector(moveToCoordinate:)];
  _coordinateMoveControl.image = [NSImage imageWithSystemSymbolName:@"location.fill"
                                           accessibilityDescription:@"Move observer to coordinate"];
  _coordinateMoveControl.imagePosition = NSImageLeading;
  _coordinateMoveControl.enabled = NO;
  NSView *coordinateSpacer = [[NSView alloc] initWithFrame:NSZeroRect];
  [coordinateSpacer setContentHuggingPriority:NSLayoutPriorityDefaultLow
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  [coordinateSpacer
      setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  NSStackView *coordinateActionRow =
      [NSStackView stackViewWithViews:@[ coordinateSpacer, _coordinateMoveControl ]];
  coordinateActionRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateActionRow.alignment = NSLayoutAttributeCenterY;
  [coordinateActionRow.widthAnchor constraintEqualToConstant:268.0].active = YES;

  InspectorSectionView *destinationSection =
      [[InspectorSectionView alloc] initWithTitle:@"Destination"
                                         controls:@[
                                           coordinateSystemRow,
                                           coordinateRow,
                                           _coordinateStatusLabel,
                                           coordinateActionRow,
                                         ]
                                      defaultsKey:@"panorama.inspector.destination.expanded"];

  const auto makeMovementRow = [](NSString *title, NSView *control) {
    NSTextField *label = [NSTextField labelWithString:title];
    [label.widthAnchor constraintEqualToConstant:82.0].active = YES;
    [control.widthAnchor constraintEqualToConstant:178.0].active = YES;
    NSStackView *row = [NSStackView stackViewWithViews:@[ label, control ]];
    row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row.alignment = NSLayoutAttributeCenterY;
    row.spacing = 8.0;
    return row;
  };

  _movementModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _movementModeControl.segmentCount = 3;
  [_movementModeControl setLabel:@"Browse" forSegment:0];
  [_movementModeControl setLabel:@"Roam" forSegment:1];
  [_movementModeControl setLabel:@"Cruise" forSegment:2];
  _movementModeControl.selectedSegment = 0;
  _movementModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _movementModeControl.target = self;
  _movementModeControl.action = @selector(movementModeChanged:);
  _movementModeControl.toolTip = @"Browse looks around; Roam uses WASD; Cruise moves forward "
                                  "continuously under mouse control";
  NSView *movementModeRow = makeMovementRow(@"Mode", _movementModeControl);

  _roamTurningModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _roamTurningModeControl.segmentCount = 2;
  [_roamTurningModeControl setLabel:@"Arrow keys" forSegment:0];
  [_roamTurningModeControl setLabel:@"Mouse" forSegment:1];
  _roamTurningModeControl.selectedSegment = 0;
  _roamTurningModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _roamTurningModeControl.target = self;
  _roamTurningModeControl.action = @selector(roamTurningModeChanged:);
  _roamTurningModeControl.toolTip =
      @"Turn with the arrow keys or by moving the pointer over the panorama";
  _roamTurningModeRow = makeMovementRow(@"Turning", _roamTurningModeControl);

  _roamMouseSensitivityControl = [NSSlider sliderWithValue:1.0
                                                  minValue:0.25
                                                  maxValue:3.0
                                                    target:self
                                                    action:@selector(roamMouseSensitivityChanged:)];
  _roamMouseSensitivityControl.continuous = YES;
  _roamMouseSensitivityControl.toolTip = @"Mouse turning sensitivity";
  _roamMouseSensitivityLabel = [NSTextField labelWithString:@"1.00×"];
  _roamMouseSensitivityLabel.alignment = NSTextAlignmentRight;
  [_roamMouseSensitivityLabel.widthAnchor constraintEqualToConstant:54.0].active = YES;
  NSStackView *roamMouseSensitivitySetting = [NSStackView
      stackViewWithViews:@[ _roamMouseSensitivityControl, _roamMouseSensitivityLabel ]];
  roamMouseSensitivitySetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamMouseSensitivitySetting.alignment = NSLayoutAttributeCenterY;
  roamMouseSensitivitySetting.spacing = 6.0;
  _roamMouseSensitivityRow = makeMovementRow(@"Sensitivity", roamMouseSensitivitySetting);

  _roamAltitudeModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _roamAltitudeModeControl.segmentCount = 2;
  [_roamAltitudeModeControl setLabel:@"Terrain" forSegment:0];
  [_roamAltitudeModeControl setLabel:@"Altitude" forSegment:1];
  _roamAltitudeModeControl.selectedSegment = 0;
  _roamAltitudeModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _roamAltitudeModeControl.target = self;
  _roamAltitudeModeControl.action = @selector(roamAltitudeModeChanged:);
  _roamAltitudeModeControl.toolTip =
      @"Maintain height above terrain or hold absolute elevation while moving";
  NSView *roamAltitudeRow = makeMovementRow(@"Height mode", _roamAltitudeModeControl);

  _aircraftDynamicsControl = [NSButton checkboxWithTitle:@"Enabled"
                                                  target:self
                                                  action:@selector(aircraftDynamicsChanged:)];
  _aircraftDynamicsControl.toolTip =
      @"Use coordinated banked turns and exchange airspeed with climbs and dives";
  _aircraftDynamicsRow = makeMovementRow(@"Aircraft", _aircraftDynamicsControl);

  _roamSpeedControl = [NSSlider sliderWithValue:panorama::app::kDefaultRoamSpeed
                                       minValue:panorama::app::kMinimumMovementSpeed
                                       maxValue:panorama::app::kMaximumRoamSpeed
                                         target:self
                                         action:@selector(roamSpeedChanged:)];
  _roamSpeedControl.continuous = YES;
  _roamSpeedControl.toolTip = @"Horizontal roaming speed";
  _roamSpeedLabel = [NSTextField labelWithString:@"72.0 km/h"];
  _roamSpeedLabel.alignment = NSTextAlignmentRight;
  [_roamSpeedLabel.widthAnchor constraintEqualToConstant:62.0].active = YES;
  NSStackView *roamSpeedSetting =
      [NSStackView stackViewWithViews:@[ _roamSpeedControl, _roamSpeedLabel ]];
  roamSpeedSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamSpeedSetting.alignment = NSLayoutAttributeCenterY;
  roamSpeedSetting.spacing = 6.0;
  _roamSpeedRowLabel = [NSTextField labelWithString:@"Speed"];
  [_roamSpeedRowLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  [roamSpeedSetting.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSStackView *roamSpeedRow =
      [NSStackView stackViewWithViews:@[ _roamSpeedRowLabel, roamSpeedSetting ]];
  roamSpeedRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamSpeedRow.alignment = NSLayoutAttributeCenterY;
  roamSpeedRow.spacing = 8.0;

  _roamUpdateRateControl = [NSSlider sliderWithValue:10.0
                                            minValue:1.0
                                            maxValue:60.0
                                              target:self
                                              action:@selector(roamUpdateRateChanged:)];
  _roamUpdateRateControl.continuous = YES;
  _roamUpdateRateControl.toolTip =
      @"Maximum observer-position requests per second; rendering may complete more slowly";
  _roamUpdateRateLabel = [NSTextField labelWithString:@"10 Hz"];
  _roamUpdateRateLabel.alignment = NSTextAlignmentRight;
  [_roamUpdateRateLabel.widthAnchor constraintEqualToConstant:54.0].active = YES;
  NSStackView *roamUpdateRateSetting =
      [NSStackView stackViewWithViews:@[ _roamUpdateRateControl, _roamUpdateRateLabel ]];
  roamUpdateRateSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamUpdateRateSetting.alignment = NSLayoutAttributeCenterY;
  roamUpdateRateSetting.spacing = 6.0;
  NSView *roamUpdateRateRow = makeMovementRow(@"Updates", roamUpdateRateSetting);

  _roamStatusLabel = [NSTextField labelWithString:@"WASD move • arrow keys look"];
  _roamStatusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _roamStatusLabel.textColor = NSColor.secondaryLabelColor;
  [_roamStatusLabel.widthAnchor constraintEqualToConstant:268.0].active = YES;
  _roamRows = @[
    _roamTurningModeRow,
    _roamMouseSensitivityRow,
    roamAltitudeRow,
    _aircraftDynamicsRow,
    roamSpeedRow,
    roamUpdateRateRow,
    _roamStatusLabel,
  ];

  InspectorSectionView *movementSection =
      [[InspectorSectionView alloc] initWithTitle:@"Movement"
                                         controls:@[
                                           movementModeRow,
                                           _roamTurningModeRow,
                                           _roamMouseSensitivityRow,
                                           roamAltitudeRow,
                                           _aircraftDynamicsRow,
                                           roamSpeedRow,
                                           roamUpdateRateRow,
                                           _roamStatusLabel,
                                         ]
                                      defaultsKey:@"panorama.inspector.movement.expanded"];

  _groundClearanceDecreaseControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"minus"
                                          accessibilityDescription:@"Lower observer"]
                         target:self
                         action:@selector(adjustGroundClearance:)];
  _groundClearanceDecreaseControl.tag = -1;
  _groundClearanceDecreaseControl.controlSize = NSControlSizeSmall;
  _groundClearanceDecreaseControl.bezelStyle = NSBezelStyleTexturedRounded;
  _groundClearanceDecreaseControl.toolTip = @"Lower eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  [_groundClearanceDecreaseControl.widthAnchor constraintEqualToConstant:24.0].active = YES;

  _groundClearanceControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _groundClearanceControl.delegate = self;
  _groundClearanceControl.alignment = NSTextAlignmentRight;
  _groundClearanceControl.font = [NSFont monospacedDigitSystemFontOfSize:12.0
                                                                  weight:NSFontWeightRegular];
  _groundClearanceControl.stringValue = [NSString stringWithFormat:@"%.1f", _groundClearance];
  _groundClearanceControl.toolTip = @"Observer height above the terrain directly beneath it";
  [_groundClearanceControl.widthAnchor constraintEqualToConstant:56.0].active = YES;

  _observerHeightUnit = [NSTextField labelWithString:@"m AGL"];
  _observerHeightUnit.textColor = NSColor.secondaryLabelColor;
  _observerHeightUnit.toolTip = @"Metres above ground level";

  _groundClearanceIncreaseControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"plus"
                                          accessibilityDescription:@"Raise observer"]
                         target:self
                         action:@selector(adjustGroundClearance:)];
  _groundClearanceIncreaseControl.tag = 1;
  _groundClearanceIncreaseControl.controlSize = NSControlSizeSmall;
  _groundClearanceIncreaseControl.bezelStyle = NSBezelStyleTexturedRounded;
  _groundClearanceIncreaseControl.toolTip = @"Raise eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  [_groundClearanceIncreaseControl.widthAnchor constraintEqualToConstant:24.0].active = YES;

  NSStackView *heightSetting = [NSStackView stackViewWithViews:@[
    _groundClearanceDecreaseControl,
    _groundClearanceControl,
    _observerHeightUnit,
    _groundClearanceIncreaseControl,
  ]];
  heightSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  heightSetting.alignment = NSLayoutAttributeCenterY;
  heightSetting.spacing = 6.0;
  [heightSetting.widthAnchor constraintEqualToConstant:178.0].active = YES;

  _observerHeightLabel = [NSTextField labelWithString:@"Eye height"];
  _observerHeightLabel.toolTip = @"Observer height above the terrain directly beneath it";
  [_observerHeightLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *heightRow =
      [NSStackView stackViewWithViews:@[ _observerHeightLabel, heightSetting ]];
  heightRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  heightRow.alignment = NSLayoutAttributeCenterY;
  heightRow.spacing = 8.0;

  InspectorSectionView *observerSection =
      [[InspectorSectionView alloc] initWithTitle:@"Observer"
                                         controls:@[ heightRow ]
                                      defaultsKey:@"panorama.inspector.observer.expanded"];
  NSStackView *settings = [NSStackView
      stackViewWithViews:@[ heading, destinationSection, movementSection, observerSection ]];
  settings.orientation = NSUserInterfaceLayoutOrientationVertical;
  settings.alignment = NSLayoutAttributeLeading;
  settings.spacing = 18.0;
  settings.edgeInsets = NSEdgeInsetsMake(20.0, 16.0, 20.0, 16.0);
  settings.translatesAutoresizingMaskIntoConstraints = NO;

  InspectorDocumentView *document = [[InspectorDocumentView alloc] initWithFrame:NSZeroRect];
  document.translatesAutoresizingMaskIntoConstraints = NO;
  [document addSubview:settings];
  scrollView.documentView = document;
  NSLayoutConstraint *viewportHeight =
      [document.heightAnchor constraintEqualToAnchor:scrollView.contentView.heightAnchor];
  viewportHeight.priority = NSLayoutPriorityDefaultLow;
  [NSLayoutConstraint activateConstraints:@[
    [document.widthAnchor constraintEqualToAnchor:scrollView.contentView.widthAnchor],
    [document.heightAnchor
        constraintGreaterThanOrEqualToAnchor:scrollView.contentView.heightAnchor],
    viewportHeight,
    [settings.topAnchor constraintEqualToAnchor:document.topAnchor],
    [settings.leadingAnchor constraintEqualToAnchor:document.leadingAnchor],
    [settings.trailingAnchor constraintEqualToAnchor:document.trailingAnchor],
    [settings.bottomAnchor constraintLessThanOrEqualToAnchor:document.bottomAnchor],
  ]];
  [self coordinateSystemChanged:_coordinateSystemControl];
  [self updateRoamControls];
  return viewController;
}

/// Build the read-only diagnostics displayed over the leading side of the
/// rendered scene. Camera values change with completed revisions; inspected
/// point details update independently as hover samples arrive.
- (NSViewController *)makeDebugViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 240.0, 430.0)];
  viewController.view = content;

  NSTextField *heading = [NSTextField labelWithString:@"Viewer Debug Info"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _debugInfoLabel = [NSTextField labelWithString:@""];
  _debugInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _debugInfoLabel.maximumNumberOfLines = 0;
  _debugInfoLabel.lineBreakMode = NSLineBreakByClipping;

  NSTextField *pointHeading = [NSTextField labelWithString:@"Inspected Point"];
  pointHeading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _debugPointInfoLabel = [NSTextField labelWithString:@"No point selected."];
  _debugPointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _debugPointInfoLabel.maximumNumberOfLines = 0;
  _debugPointInfoLabel.lineBreakMode = NSLineBreakByClipping;

  NSStackView *debugInfo = [NSStackView
      stackViewWithViews:@[ heading, _debugInfoLabel, pointHeading, _debugPointInfoLabel ]];
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
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 230.0, 36.0)];
  viewController.view = content;

  _observerInfoLabel = [NSTextField labelWithString:@""];
  _observerInfoLabel.font = [NSFont monospacedSystemFontOfSize:10.0 weight:NSFontWeightRegular];
  _observerInfoLabel.textColor = NSColor.secondaryLabelColor;
  _observerInfoLabel.maximumNumberOfLines = 2;
  _observerInfoLabel.lineBreakMode = NSLineBreakByClipping;
  _movementInfoLabel = [NSTextField labelWithString:@""];
  _movementInfoLabel.font = [NSFont monospacedSystemFontOfSize:10.0 weight:NSFontWeightRegular];
  _movementInfoLabel.textColor = NSColor.secondaryLabelColor;
  _movementInfoLabel.maximumNumberOfLines = 2;
  _movementInfoLabel.lineBreakMode = NSLineBreakByClipping;
  _movementInfoLabel.hidden = YES;

  _pointInfoHeading = [NSTextField labelWithString:@" "];
  _pointInfoHeading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _pointInfoHeading.maximumNumberOfLines = 1;
  _pointInfoHeading.lineBreakMode = NSLineBreakByTruncatingTail;
  _pointInfoLabel = [NSTextField labelWithString:@""];
  _pointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _pointInfoLabel.maximumNumberOfLines = 1;
  _pointInfoLabel.lineBreakMode = NSLineBreakByClipping;

  _pointVisibilityIcon = [[NSImageView alloc] initWithFrame:NSZeroRect];
  _pointVisibilityIcon.contentTintColor = NSColor.secondaryLabelColor;
  _pointVisibilityIcon.hidden = YES;
  _pointLockIcon = [[NSImageView alloc] initWithFrame:NSZeroRect];
  _pointLockIcon.contentTintColor = NSColor.secondaryLabelColor;
  _pointLockIcon.hidden = YES;

  _moveToLockedPointControl = [NSButton buttonWithTitle:@"Move here"
                                                 target:self
                                                 action:@selector(moveToLockedPoint:)];
  _moveToLockedPointControl.controlSize = NSControlSizeSmall;
  _moveToLockedPointControl.bezelStyle = NSBezelStyleRounded;
  _moveToLockedPointControl.toolTip = @"Move the observer to the locked terrain point";
  _moveToLockedPointControl.hidden = YES;

  NSView *pointSpacer = [[NSView alloc] initWithFrame:NSZeroRect];
  [pointSpacer setContentHuggingPriority:NSLayoutPriorityDefaultLow
                          forOrientation:NSLayoutConstraintOrientationHorizontal];
  [pointSpacer setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                        forOrientation:NSLayoutConstraintOrientationHorizontal];

  _pointInfoRow = [NSStackView stackViewWithViews:@[
    _pointInfoHeading,
    _pointInfoLabel,
    _pointVisibilityIcon,
    _pointLockIcon,
    pointSpacer,
    _moveToLockedPointControl,
  ]];
  _pointInfoRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  _pointInfoRow.alignment = NSLayoutAttributeCenterY;
  _pointInfoRow.spacing = 7.0;

  NSStackView *footer = [NSStackView stackViewWithViews:@[
    _observerInfoLabel,
    _movementInfoLabel,
    _pointInfoRow,
  ]];
  footer.orientation = NSUserInterfaceLayoutOrientationVertical;
  footer.alignment = NSLayoutAttributeLeading;
  footer.spacing = 2.0;
  footer.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:footer];
  [NSLayoutConstraint activateConstraints:@[
    [footer.topAnchor constraintEqualToAnchor:content.topAnchor constant:6.0],
    [footer.bottomAnchor constraintEqualToAnchor:content.bottomAnchor constant:-6.0],
    [footer.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:14.0],
    [footer.trailingAnchor constraintEqualToAnchor:content.trailingAnchor constant:-14.0],
    [_observerInfoLabel.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_movementInfoLabel.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_pointInfoRow.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_pointInfoRow.heightAnchor constraintEqualToConstant:24.0],
  ]];
  return viewController;
}

- (void)adjustGroundClearance:(NSButton *)sender {
  NSEventModifierFlags modifiers = NSApp.currentEvent.modifierFlags;
  const double step = (modifiers & NSEventModifierFlagShift) != 0U    ? 10.0
                      : (modifiers & NSEventModifierFlagOption) != 0U ? 0.1
                                                                      : 1.0;
  const bool holdAltitude = ([self isRoamingEnabled] || [self isCruisingEnabled]) &&
                            _roamAltitudeModeControl.selectedSegment == 1;
  const double current = holdAltitude ? _roamAltitude : _groundClearance;
  const double minimum = holdAltitude ? -500.0 : 0.0;
  _groundClearanceControl.doubleValue =
      std::clamp(current + static_cast<double>(sender.tag) * step, minimum, 100'000.0);
  _groundClearanceControl.stringValue =
      [NSString stringWithFormat:@"%.1f", _groundClearanceControl.doubleValue];
  [self commitGroundClearanceControl];
}

- (BOOL)commitGroundClearanceControl {
  const std::optional<double> parsed =
      panorama::app::parse_range_value(_groundClearanceControl.stringValue);
  const bool holdAltitude = ([self isRoamingEnabled] || [self isCruisingEnabled]) &&
                            _roamAltitudeModeControl.selectedSegment == 1;
  const double minimum = holdAltitude ? -500.0 : 0.0;
  const BOOL valid = parsed.has_value() && *parsed >= minimum && *parsed <= 100'000.0;
  _groundClearanceControl.textColor = valid ? NSColor.controlTextColor : NSColor.systemRedColor;
  _groundClearanceControl.toolTip =
      valid
          ? (holdAltitude ? @"Observer elevation above mean sea level"
                          : @"Observer height above the terrain directly beneath it")
          : (holdAltitude ? @"Altitude must be between -500 and 100,000 metres AMSL."
                          : @"Eye height must be between 0 and 100,000 metres above ground level.");
  if (!valid) {
    NSBeep();
    return NO;
  }
  _groundClearanceControl.stringValue = [NSString stringWithFormat:@"%.1f", *parsed];
  if (holdAltitude) {
    if (std::abs(*parsed - _roamAltitude) <= 1e-9) {
      return YES;
    }
    _roamAltitude = *parsed;
    _roamRequestToken = _renderer->request_roam(
        _roamDesiredPosition,
        panorama::app::RoamAltitudeMode::HoldAltitude,
        _roamAltitude
    );
    return YES;
  }
  if (std::abs(*parsed - _groundClearance) <= 1e-9) {
    return YES;
  }
  _groundClearance = *parsed;
  if ([self isRoamingEnabled] || [self isCruisingEnabled]) {
    _roamRequestToken = _renderer->request_roam(
        _roamDesiredPosition,
        panorama::app::RoamAltitudeMode::FollowTerrain,
        _groundClearance
    );
    return YES;
  }
  _renderer->request_ground_clearance(_groundClearance);
  return YES;
}

- (void)coordinateSystemChanged:(id)sender {
  (void)sender;
  const NSInteger tag = _coordinateSystemControl.selectedItem.tag;
  switch (tag) {
  case -1:
    _coordinateInputControl.placeholderString = @"Enter or paste a coordinate";
    _coordinateInputControl.toolTip = @"The coordinate system will be detected automatically";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Wgs84):
    _coordinateInputControl.placeholderString = @"46.948, 7.447";
    _coordinateInputControl.toolTip = @"Latitude, longitude in decimal WGS 84 degrees";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::SwissLv95):
    _coordinateInputControl.placeholderString = @"2600000, 1200000";
    _coordinateInputControl.toolTip = @"LV95 easting, northing in metres";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::BritishNationalGrid):
    _coordinateInputControl.placeholderString = @"NG 90716 59877";
    _coordinateInputControl.toolTip =
        @"OS grid reference, or British National Grid easting, northing in metres";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Terrain):
    _coordinateInputControl.placeholderString =
        _renderer->terrain_crs().id() == panorama::CrsId::FrenchLambert93 ? @"700000, 6600000"
        : _renderer->terrain_crs().id() == panorama::CrsId::SwissLv95     ? @"2600000, 1200000"
                                                                          : @"400000, 300000";
    _coordinateInputControl.toolTip = [NSString
        stringWithFormat:@"Easting, northing in %s metres", _renderer->terrain_crs().name()];
    break;
  default:
    throw std::logic_error("Unknown coordinate-system menu item");
  }
  [self updateCoordinateInputValidation];
}

- (BOOL)updateCoordinateInputValidation {
  if (_coordinateSystemControl == nil || _coordinateInputControl == nil ||
      _coordinateStatusLabel == nil || _coordinateMoveControl == nil) {
    return NO;
  }
  _coordinateDestination.reset();
  NSString *input = [_coordinateInputControl.stringValue
      stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
  if (input.length == 0U) {
    _coordinateInputControl.textColor = NSColor.controlTextColor;
    const NSInteger tag = _coordinateSystemControl.selectedItem.tag;
    _coordinateStatusLabel.stringValue =
        tag == -1
            ? @"Format will be detected automatically"
            : [NSString
                  stringWithFormat:@"Enter coordinates as %s",
                                   tag == static_cast<NSInteger>(
                                              panorama::app::CoordinateInputSystem::Terrain
                                          )
                                       ? _renderer->terrain_crs().name()
                                       : _coordinateSystemControl.titleOfSelectedItem.UTF8String];
    _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
    _coordinateMoveControl.enabled = NO;
    return NO;
  }

  try {
    panorama::app::ParsedCoordinateInput parsed;
    const NSInteger tag = _coordinateSystemControl.selectedItem.tag;
    if (tag == -1) {
      std::vector<panorama::app::ParsedCoordinateInput> candidates =
          panorama::app::detect_coordinate_inputs(input.UTF8String, _renderer->terrain_crs());
      if (candidates.size() > 1U) {
        std::vector<panorama::app::ParsedCoordinateInput> covered;
        for (const auto &candidate : candidates) {
          if (coordinate_has_terrain_coverage(_renderer->terrain_coverage(), candidate.projected)) {
            covered.push_back(candidate);
          }
        }
        if (!covered.empty()) {
          candidates = std::move(covered);
        }
      }
      if (candidates.size() > 1U) {
        NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:candidates.size()];
        for (const auto &candidate : candidates) {
          [names addObject:[NSString stringWithUTF8String:candidate.source_name.c_str()]];
        }
        NSString *possibilities = [names componentsJoinedByString:@" or "];
        _coordinateStatusLabel.stringValue =
            [NSString stringWithFormat:@"Could be %@ — choose a system above", possibilities];
        _coordinateStatusLabel.textColor = NSColor.systemOrangeColor;
        _coordinateInputControl.textColor = NSColor.controlTextColor;
        _coordinateInputControl.toolTip = _coordinateStatusLabel.stringValue;
        _coordinateMoveControl.enabled = NO;
        return NO;
      }
      parsed = candidates.front();
    } else {
      parsed = panorama::app::parse_coordinate_input(
          input.UTF8String,
          _renderer->terrain_crs(),
          static_cast<panorama::app::CoordinateInputSystem>(tag)
      );
    }
    _coordinateDestination = parsed;
    NSString *source = [NSString stringWithUTF8String:parsed.source_name.c_str()];
    _coordinateStatusLabel.stringValue = [NSString stringWithFormat:@"%@ • %.5f°, %.5f°",
                                                                    source,
                                                                    parsed.geographic.lat,
                                                                    parsed.geographic.lon];
    _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
    _coordinateInputControl.textColor = NSColor.controlTextColor;
    _coordinateInputControl.toolTip = _coordinateStatusLabel.stringValue;
    _coordinateMoveControl.enabled = YES;
    return YES;
  } catch (const std::exception &exception) {
    NSString *error = [NSString stringWithUTF8String:exception.what()];
    _coordinateStatusLabel.stringValue = error;
    _coordinateStatusLabel.textColor = NSColor.systemRedColor;
    _coordinateInputControl.textColor = NSColor.systemRedColor;
    _coordinateInputControl.toolTip = error;
    _coordinateMoveControl.enabled = NO;
    return NO;
  }
}

- (void)moveToCoordinate:(id)sender {
  (void)sender;
  if (![self updateCoordinateInputValidation] || !_coordinateDestination.has_value()) {
    NSBeep();
    return;
  }
  const panorama::app::ParsedCoordinateInput parsed = *_coordinateDestination;
  _coordinateMovePending = true;
  _coordinateStatusLabel.stringValue =
      [NSString stringWithFormat:@"%s • locating terrain…", parsed.source_name.c_str()];
  _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
  [self requestMapPointEasting:parsed.projected.x
                      northing:parsed.projected.y
                        action:panorama::app::MapPointAction::MoveObserver];
}

- (void)setPointInfoStatus:(NSString *)status {
  if (_pointInfoHeading == nil) {
    return;
  }
  // Keep one blank glyph in the otherwise empty point row so entering and
  // leaving hover does not resize the minimap panel beneath the pointer.
  _pointInfoHeading.stringValue = status.length == 0U ? @" " : status;
  _pointInfoLabel.stringValue = @"";
  _moveToLockedPointControl.hidden = YES;
  [self setPointInfoSymbolsVisible:false locked:false occluded:false];
}

/// Keep observer state visible independently of the transient hover/lock row.
/// Compact projected coordinates are more useful here than place names: they
/// update immediately during movement and match the terrain dataset's grid.
- (void)updateMiniMapTelemetry {
  if (_observerInfoLabel == nil) {
    return;
  }
  _observerInfoLabel.stringValue =
      [NSString stringWithFormat:@"Observer  E %.0f • N %.0f\n%.1f m AGL • %.0f m AMSL",
                                 _observer.easting,
                                 _observer.northing,
                                 _groundClearance,
                                 _observer.elevation];
  _observerInfoLabel.toolTip =
      [NSString stringWithFormat:@"Observer: easting %.1f m, northing %.1f m, %.1f m above ground, "
                                  "%.1f m above mean sea level",
                                 _observer.easting,
                                 _observer.northing,
                                 _groundClearance,
                                 _observer.elevation];

  const BOOL roaming = [self isRoamingEnabled];
  const BOOL cruising = [self isCruisingEnabled];
  const BOOL movementHidden = !roaming && !cruising;
  const BOOL movementVisibilityChanged = _movementInfoLabel.hidden != movementHidden;
  _movementInfoLabel.hidden = movementHidden;
  if (roaming || cruising) {
    const BOOL aircraft = [self isAircraftDynamicsEnabled];
    double heading = std::fmod(_orientation.heading * panorama::app::kRadiansToDegrees, 360.0);
    if (heading < 0.0) {
      heading += 360.0;
    }
    NSString *mode = _roamAltitudeModeControl.selectedSegment == 0
                         ? @"Terrain"
                         : (cruising ? @"Flight" : @"Altitude");
    NSString *state = aircraft ? (_viewerPaused ? @"Aircraft paused" : @"Aircraft")
                               : (cruising ? (_viewerPaused ? @"Cruise paused" : @"Cruise")
                                           : (_viewerPaused ? @"Roam paused" : @"Roam"));
    const double speed = aircraft ? _aircraftAirspeed : (cruising ? _cruiseSpeed : _roamSpeed);
    _movementInfoLabel.stringValue =
        aircraft
            ? [NSString
                  stringWithFormat:@"%@ • %@ • %@\nHeading %03.0f° • Pitch %+.0f° • Bank %+.0f°",
                                   state,
                                   format_movement_speed(speed),
                                   mode,
                                   heading,
                                   _orientation.pitch * panorama::app::kRadiansToDegrees,
                                   _aircraftBank * panorama::app::kRadiansToDegrees]
        : cruising
            ? [NSString stringWithFormat:@"%@ • %@ • %@\nHeading %03.0f° • Pitch %+.0f°",
                                         state,
                                         format_movement_speed(speed),
                                         mode,
                                         heading,
                                         _orientation.pitch * panorama::app::kRadiansToDegrees]
            : [NSString stringWithFormat:@"%@ • %@ • %@\nHeading %03.0f°",
                                         state,
                                         format_movement_speed(speed),
                                         mode,
                                         heading];
    _movementInfoLabel.toolTip = _movementInfoLabel.stringValue;
  }
  if (movementVisibilityChanged) {
    [_miniMapPanel informationFooterContentDidChange];
  }
}

- (void)setPointInfoSymbolsVisible:(bool)visible locked:(bool)locked occluded:(bool)occluded {
  if (_pointVisibilityIcon == nil || _pointLockIcon == nil) {
    return;
  }
  _pointVisibilityIcon.hidden = !visible;
  _pointLockIcon.hidden = !visible;
  _moveToLockedPointControl.hidden =
      !(visible && locked && _pointInspectionLocked && _lockedPoint.has_value());
  if (!visible) {
    return;
  }

  NSString *visibilitySymbol = occluded ? @"eye.slash" : @"eye";
  NSString *visibilityDescription =
      occluded ? @"Terrain point is occluded" : @"Terrain point is visible";
  NSString *lockSymbol = locked ? @"lock.fill" : @"lock.open";
  NSString *lockDescription = locked ? @"Terrain point is locked" : @"Terrain point is unlocked";
  NSImageSymbolConfiguration *configuration =
      [NSImageSymbolConfiguration configurationWithPointSize:12.0 weight:NSFontWeightMedium];
  _pointVisibilityIcon.image = [[NSImage imageWithSystemSymbolName:visibilitySymbol
                                          accessibilityDescription:visibilityDescription]
      imageWithSymbolConfiguration:configuration];
  _pointVisibilityIcon.toolTip = visibilityDescription;
  [_pointVisibilityIcon setAccessibilityLabel:visibilityDescription];
  _pointLockIcon.image = [[NSImage imageWithSystemSymbolName:lockSymbol
                                    accessibilityDescription:lockDescription]
      imageWithSymbolConfiguration:configuration];
  _pointLockIcon.toolTip = lockDescription;
  [_pointLockIcon setAccessibilityLabel:lockDescription];
}

/// Put detailed samples in the opt-in debug overlay, leaving the persistent
/// point card small enough to sit naturally beneath the minimap.
- (void)updateDebugPointInfo:(std::optional<panorama::app::PointInspection>)inspection {
  if (_debugPointInfoLabel == nil) {
    return;
  }
  if (!inspection.has_value()) {
    _debugPointInfoLabel.stringValue = @"No point selected.";
    return;
  }
  const panorama::app::PointInspection &point = *inspection;
  if (!point.hit) {
    _debugPointInfoLabel.stringValue =
        [NSString stringWithFormat:@"Pixel      %4u, %4u\nNo terrain intersection",
                                   point.pixel.x,
                                   point.pixel.y];
    return;
  }
  if (point.map_selected) {
    _debugPointInfoLabel.stringValue =
        [NSString stringWithFormat:@"Map selection\n"
                                    "Distance   %10.1f m\nElevation  %10.1f m\n"
                                    "Easting    %10.1f m\nNorthing   %10.1f m",
                                   point.distance,
                                   point.elevation,
                                   point.easting,
                                   point.northing];
    return;
  }
  _debugPointInfoLabel.stringValue =
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

- (void)updatePointInfo:(std::optional<panorama::app::PointInspection>)inspection {
  if (_pointInfoLabel == nil) {
    return;
  }
  [self updateDebugPointInfo:inspection];
  if (!inspection.has_value()) {
    if (!_pointInspectionLocked) {
      [_miniMapPanel clearInspectedPoint];
    }
    [self setPointInfoStatus:@""];
    return;
  }
  const panorama::app::PointInspection &point = *inspection;
  if (!point.hit) {
    [_miniMapPanel clearInspectedPoint];
    [self setPointInfoStatus:@"No terrain intersection"];
    return;
  }
  [_miniMapPanel setInspectedPointEasting:point.easting
                                 northing:point.northing
                                   locked:_pointInspectionLocked || _pointLockPending];
  _pointInfoHeading.stringValue = @"Distance";
  _pointInfoLabel.stringValue = format_point_distance(point.distance);
  [self setPointInfoSymbolsVisible:true
                            locked:_pointInspectionLocked || _pointLockPending
                          occluded:_pointInspectionLocked && _lockedPointOccluded];
}

/// Keep a locked world point aligned with the latest completed camera view.
/// The off-screen state is represented both by an edge arrow and in text, so
/// the lock remains unambiguous even if another overlay obscures the marker.
- (void)updateLockedPointIndicatorWithOrientation:(panorama::CameraOrientation)orientation
                              verticalFieldOfView:(double)verticalFieldOfView
                                            image:(panorama::ImageSize)image {
  const panorama::app::PointInspection *point = nullptr;
  bool locked = false;
  if (_pointInspectionLocked && _lockedPoint.has_value()) {
    point = &*_lockedPoint;
    locked = true;
  } else if (_mapHoverPoint.has_value()) {
    point = &*_mapHoverPoint;
  }
  if (point == nullptr) {
    [_panoramaView setTerrainPointIndicator:std::nullopt locked:true occluded:false];
    return;
  }
  const panorama::app::LockedPointProjection projection = panorama::app::project_locked_point(
      {point->easting, point->northing, point->elevation},
      _observer,
      image,
      verticalFieldOfView,
      orientation
  );
  const bool occluded =
      locked && _lockedPointOccluded && _targetVisibilityRevision == _displayedRevision;
  [_panoramaView setTerrainPointIndicator:projection locked:locked occluded:occluded];
  if (locked) {
    _pointInfoHeading.stringValue = @"Distance";
    [self setPointInfoSymbolsVisible:true locked:true occluded:occluded];
  }
}

/// Palette and range controls have no effect on the uncoloured white mode.
- (void)updateSettingsControlAvailability {
  const BOOL scalarColour = _colourSourceControl.indexOfSelectedItem != 0;
  for (NSView *row in _scalarColourRows) {
    row.hidden = !scalarColour;
  }
  _featureOutlineDetailRow.hidden = _featureOutlinesControl.state != NSControlStateValueOn;
  const BOOL normalLighting = _normalLightingControl.state == NSControlStateValueOn;
  const BOOL manualSun = _sunModeControl.selectedSegment == 0;
  const BOOL hasObserverTimeZone = _observerTimeZone != nil;
  for (NSView *view in _normalLightingRows) {
    view.hidden = !normalLighting;
  }
  for (NSView *row in _manualSunRows) {
    row.hidden = !normalLighting || !manualSun;
  }
  for (NSView *row in _astronomicalSunRows) {
    row.hidden = !normalLighting || manualSun;
  }

  const BOOL astronomicalTimeEnabled = normalLighting && !manualSun && hasObserverTimeZone;
  _astronomicalDateControl.enabled = astronomicalTimeEnabled;
  _astronomicalTimeControl.enabled = astronomicalTimeEnabled;
  _astronomicalTimeDecreaseControl.enabled =
      astronomicalTimeEnabled && _astronomicalTimeControl.doubleValue > 0.0;
  _astronomicalTimeIncreaseControl.enabled =
      astronomicalTimeEnabled && _astronomicalTimeControl.doubleValue < 1439.0;
}

- (void)renderModeChanged:(id)sender {
  (void)sender;
  [self updateSettingsControlAvailability];
  [self publishTerrainControls];
}

/// Commit the colour controls independently of camera and lighting edits.
- (BOOL)publishTerrainControls {
  const std::optional<double> minimum =
      panorama::app::parse_range_value(_minimumControl.stringValue);
  const std::optional<double> maximum =
      panorama::app::parse_range_value(_maximumControl.stringValue);
  const BOOL validRange = minimum.has_value() && maximum.has_value() && *maximum > *minimum &&
                          *minimum >= -std::numeric_limits<float>::max() &&
                          *maximum <= std::numeric_limits<float>::max();
  _minimumControl.textColor = validRange ? NSColor.controlTextColor : NSColor.systemRedColor;
  _maximumControl.textColor = validRange ? NSColor.controlTextColor : NSColor.systemRedColor;
  NSString *rangeError = validRange
                             ? nil
                             : @"Enter finite metre values with the maximum greater than the "
                                "minimum; commas may only separate thousands.";
  _minimumControl.toolTip = rangeError;
  _maximumControl.toolTip = rangeError;
  const BOOL scalarColour = _colourSourceControl.indexOfSelectedItem != 0;
  if (scalarColour && !validRange) {
    return NO;
  }

  _presentation.appearance.colour_source =
      static_cast<panorama::TerrainColourSource>(_colourSourceControl.indexOfSelectedItem);
  _presentation.appearance.colourmap =
      static_cast<panorama::PresetColourmap>(_colourmapControl.indexOfSelectedItem);
  _presentation.appearance.colour_scale =
      static_cast<panorama::ScalarColourScale>(_colourScaleControl.indexOfSelectedItem);
  if (validRange) {
    _presentation.colour_range = {
        static_cast<float>(*minimum),
        static_cast<float>(*maximum),
    };
  }
  _renderer->request_presentation(_presentation);
  return YES;
}

/// Resize only after both text fields form a valid Metal image size. Invalid
/// edits remain visible in red so the user can correct them without dismissing
/// an alert or losing the partially entered value.
- (BOOL)commitResolutionControls {
  const std::optional<uint32_t> width =
      panorama::app::parse_image_dimension(_imageWidthControl.stringValue);
  const std::optional<uint32_t> height =
      panorama::app::parse_image_dimension(_imageHeightControl.stringValue);
  const uint64_t pixelCount =
      width.has_value() && height.has_value() ? static_cast<uint64_t>(*width) * *height : 0U;
  const BOOL valid =
      width.has_value() && height.has_value() && pixelCount <= std::numeric_limits<uint32_t>::max();
  _imageWidthControl.textColor = valid ? NSColor.controlTextColor : NSColor.systemRedColor;
  _imageHeightControl.textColor = valid ? NSColor.controlTextColor : NSColor.systemRedColor;
  NSString *resolutionError =
      valid ? nil
            : @"Width and height must be positive whole numbers whose product fits in the "
               "32-bit Metal ray-index range.";
  _imageWidthControl.toolTip = resolutionError;
  _imageHeightControl.toolTip = resolutionError;
  if (!valid) {
    return NO;
  }
  const panorama::ImageSize next_image = {*width, *height};
  if (next_image.width != _image.width || next_image.height != _image.height) {
    _image = next_image;
    _inspectionRequestToken = _renderer->request_inspection(std::nullopt);
    _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  }
  return YES;
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

  if (frame.target_visibility_sequence != _displayedTargetVisibilitySequence) {
    _displayedTargetVisibilitySequence = frame.target_visibility_sequence;
    if (frame.target_visibility.has_value() &&
        frame.target_visibility->request_token == _targetVisibilityRequestToken) {
      _targetVisibilityRevision = frame.target_visibility->revision;
      _lockedPointOccluded = frame.target_visibility->occluded;
      [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                  verticalFieldOfView:frame.vertical_field_of_view
                                                image:frame.image];
    }
  }

  bool acceptedRoamMove = false;
  if (frame.roam_result_sequence != _displayedRoamResultSequence) {
    _displayedRoamResultSequence = frame.roam_result_sequence;
    if (frame.roam_result.has_value()) {
      if (frame.roam_result->accepted) {
        acceptedRoamMove = true;
        _groundClearance = std::max(
            0.0,
            frame.observer.elevation - static_cast<double>(frame.roam_result->ground_elevation)
        );
        if (frame.roam_result->request_token == _roamRequestToken) {
          if (_roamAltitudeModeControl.selectedSegment == 0 &&
              _groundClearanceControl.currentEditor == nil) {
            _groundClearanceControl.stringValue =
                [NSString stringWithFormat:@"%.1f", _groundClearance];
          } else if (_roamAltitudeModeControl.selectedSegment == 1) {
            _roamAltitude = frame.observer.elevation;
            if (_groundClearanceControl.currentEditor == nil) {
              _groundClearanceControl.stringValue =
                  [NSString stringWithFormat:@"%.1f", _roamAltitude];
            }
          }
        }
      } else if (frame.roam_result->request_token == _roamRequestToken) {
        [self clearRoamKeys];
        _roamDesiredPosition = {frame.observer.easting, frame.observer.northing};
        const bool terrainCollision = std::isfinite(frame.roam_result->ground_elevation);
        if ([self isCruisingEnabled] && !_viewerPaused) {
          [self pauseCruiseForTerrainCollision:terrainCollision];
        }
        if (_cruiseRecovery) {
          _roamStatusLabel.stringValue =
              terrainCollision ? @"Blocked by terrain • drag to steer/climb • Space resumes"
                               : @"No terrain coverage • drag to turn back • Space resumes";
          _roamStatusLabel.textColor = NSColor.systemRedColor;
        } else if (![self isCruisingEnabled]) {
          _roamStatusLabel.stringValue =
              terrainCollision ? @"Movement stopped: terrain reaches the held altitude"
                               : @"Movement stopped: no terrain coverage";
          _roamStatusLabel.textColor = NSColor.systemRedColor;
        }
        NSBeep();
      }
    }
  }

  if (frame.map_point_sequence != _displayedMapPointSequence) {
    _displayedMapPointSequence = frame.map_point_sequence;
    if (frame.map_point_request_token == _mapPointRequestToken &&
        _mapPointAction != panorama::app::MapPointAction::None) {
      const panorama::app::MapPointAction action = _mapPointAction;
      const bool coordinateMove =
          action == panorama::app::MapPointAction::MoveObserver && _coordinateMovePending;
      _mapPointAction = panorama::app::MapPointAction::None;
      _coordinateMovePending = false;
      if (!frame.map_point.has_value()) {
        _mapHoverPoint.reset();
        if (action == panorama::app::MapPointAction::Hover) {
          [self updatePointInfo:std::nullopt];
          [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                      verticalFieldOfView:frame.vertical_field_of_view
                                                    image:frame.image];
        } else {
          NSBeep();
          [self setPointInfoStatus:@"No terrain coverage"];
          if (coordinateMove) {
            _coordinateStatusLabel.stringValue = @"No terrain coverage at that coordinate";
            _coordinateStatusLabel.textColor = NSColor.systemRedColor;
          }
        }
      } else {
        const double distance = std::hypot(
            frame.map_point->easting - _observer.easting,
            frame.map_point->northing - _observer.northing
        );
        panorama::app::PointInspection point = {
            .pixel = {},
            .revision = frame.revision,
            .hit = true,
            .distance = static_cast<float>(distance),
            .elevation = frame.map_point->elevation,
            .easting = frame.map_point->easting,
            .northing = frame.map_point->northing,
            .slope_degrees = 0.0F,
            .aspect_degrees = 0.0F,
            .map_selected = true,
        };
        if (action == panorama::app::MapPointAction::Hover) {
          _mapHoverPoint = point;
          [self updatePointInfo:point];
          [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                      verticalFieldOfView:frame.vertical_field_of_view
                                                    image:frame.image];
        } else if (action == panorama::app::MapPointAction::MoveObserver) {
          if (coordinateMove) {
            [self updateCoordinateInputValidation];
          }
          [self moveObserverToTerrainPoint:*frame.map_point];
        } else {
          _mapHoverPoint.reset();
          _pointInspectionLocked = true;
          _pointLockPending = false;
          _lockedPoint = point;
          [self requestTargetVisibilityForPoint:*frame.map_point];
          [self updatePointInfo:point];
          [self lookAtTerrainPoint:*frame.map_point];
        }
      }
    }
  }

  if (frame.inspection_sequence != _displayedInspectionSequence) {
    _displayedInspectionSequence = frame.inspection_sequence;
    const bool matches_visible_frame =
        !frame.inspection.has_value() || frame.inspection->revision == frame.revision;
    if (_pointLockPending && frame.inspection_request_token == _pointLockRequestToken &&
        matches_visible_frame && frame.inspection.has_value()) {
      _pointLockPending = false;
      if (frame.inspection->hit) {
        _pointInspectionLocked = true;
        _lockedPoint = frame.inspection;
        [self updatePointInfo:frame.inspection];
        const panorama::app::TerrainPoint target = {
            frame.inspection->easting,
            frame.inspection->northing,
            frame.inspection->elevation,
        };
        [self requestTargetVisibilityForPoint:target];
        [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                    verticalFieldOfView:frame.vertical_field_of_view
                                                  image:frame.image];
        // The label now owns the immutable sampled values. Stop the renderer
        // resampling this screen pixel as subsequent camera views complete.
        _renderer->request_inspection(std::nullopt);
      } else {
        [self updatePointInfo:frame.inspection];
      }
    } else if (_pointInspectionEnabled && !_pointInspectionLocked && !_pointLockPending &&
               _pointerOwner == panorama::app::PointerOwner::Panorama &&
               frame.inspection_request_token == _inspectionRequestToken) {
      [self updatePointInfo:matches_visible_frame ? frame.inspection : std::nullopt];
    }
  }

  if (!frame.error.empty()) {
    _window.title = [NSString stringWithFormat:@"panorama-app — error: %s", frame.error.c_str()];
  } else if (frame.revision != 0U && frame.revision != _displayedRevision) {
    _displayedRevision = frame.revision;
    const bool observerPositionMoved = frame.observer.easting != _observer.easting ||
                                       frame.observer.northing != _observer.northing;
    const bool observerMoved =
        observerPositionMoved || frame.observer.elevation != _observer.elevation;
    _observer = frame.observer;
    if (observerMoved) {
      if (_viewerPaused || (![self isCruisingEnabled] && ![self hasPressedRoamKey])) {
        _roamDesiredPosition = {_observer.easting, _observer.northing};
      }
      [_miniMapPanel setObserverEasting:_observer.easting northing:_observer.northing];
      if (acceptedRoamMove) {
        [_miniMapPanel centerOnObserver];
      }
      if (_pointInspectionLocked && _lockedPoint.has_value()) {
        _lockedPoint->distance = static_cast<float>(std::hypot(
            _lockedPoint->easting - _observer.easting,
            _lockedPoint->northing - _observer.northing
        ));
        [self updatePointInfo:_lockedPoint];
      } else {
        [self updatePointInfo:std::nullopt];
      }
      const bool movingContinuously = ([self isRoamingEnabled] && [self hasPressedRoamKey]) ||
                                      ([self isCruisingEnabled] && !_viewerPaused);
      if (observerPositionMoved && !movingContinuously) {
        [self resolveObserverTimeZone];
      }
    }
    const double fps = frame.milliseconds > 0.0 ? 1'000.0 / frame.milliseconds : 0.0;
    [self updateDebugInfoWithOrientation:frame.orientation
                     verticalFieldOfView:frame.vertical_field_of_view
                                   image:frame.image
                            milliseconds:frame.milliseconds
                                revision:frame.revision];
    [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                verticalFieldOfView:frame.vertical_field_of_view
                                              image:frame.image];
    [_miniMapPanel setCameraOrientation:frame.orientation
                    verticalFieldOfView:frame.vertical_field_of_view
                                  image:frame.image];
    [_miniMapPanel setVisibilityPoints:frame.visibility_points image:frame.image];
    [self updateMiniMapTelemetry];
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
static NSToolbarItemIdentifier const kMapToolbarItemIdentifier = @"panorama.minimap";

@interface PanoramaAppDelegate : NSObject <NSApplicationDelegate, NSToolbarDelegate> {
@private
  std::unique_ptr<panorama::app::ViewerRenderer> _renderer;
  NSWindow *_window;
  PanoramaController *_controller;
  NSTabViewController *_inspectorController;
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
    kMapToolbarItemIdentifier,
    kDebugToolbarItemIdentifier,
    NSToolbarToggleInspectorItemIdentifier,
  ];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarAllowedItemIdentifiers:(NSToolbar *)toolbar {
  (void)toolbar;
  return @[
    NSToolbarFlexibleSpaceItemIdentifier,
    NSToolbarSpaceItemIdentifier,
    kMapToolbarItemIdentifier,
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
  const BOOL isMap = [itemIdentifier isEqualToString:kMapToolbarItemIdentifier];
  if (!isInspector && !isDebug && !isMap) {
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
  } else if (isMap) {
    item.target = _controller;
    item.label = @"Map & Inspect";
    item.paletteLabel = @"Map & Inspect";
    item.toolTip = @"Show the minimap and enable terrain-point inspection";
    item.image = [NSImage imageWithSystemSymbolName:@"map"
                           accessibilityDescription:@"Toggle Map and Point Inspection"];
    item.action = @selector(toggleMapAndPointInspection:);
    if (@available(macOS 26.0, *)) {
      item.style = [_controller isMapAndPointInspectionEnabled] ? NSToolbarItemStyleProminent
                                                                : NSToolbarItemStylePlain;
    }
  }
  return item;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  (void)notification;
  const panorama::ImageSize image = _renderer->initial_image();
  constexpr CGFloat kInspectorWidth = 300.0;
  constexpr NSSize kDebugSize = {240.0, 430.0};
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
  settingsController.title = @"Viewer";
  NSViewController *positioningController = [_controller makePositioningViewController];
  positioningController.title = @"Position";
  _inspectorController = [[NSTabViewController alloc] init];
  _inspectorController.tabStyle = NSTabViewControllerTabStyleSegmentedControlOnTop;
  [_inspectorController addChildViewController:settingsController];
  [_inspectorController addChildViewController:positioningController];
  NSViewController *debugController = [_controller makeDebugViewController];
  NSViewController *pointInfoController = [_controller makePointInfoViewController];
  const panorama::ObserverLocation observer = _renderer->observer();
  MiniMapPanelView *miniMapPanel =
      [[MiniMapPanelView alloc] initWithObserverEasting:observer.easting
                                               northing:observer.northing
                                        terrainEpsgCode:_renderer->terrain_crs().epsg_code()
                                        terrainCoverage:_renderer->terrain_coverage()
                               coverageInitiallyVisible:_renderer->observer_used_fallback()
                                            maxDistance:_renderer->max_distance()
                                          pointInfoView:pointInfoController.view
                                            metalDevice:_renderer->device()
                                           commandQueue:_renderer->command_queue()
                                                library:_renderer->library()];
  _overlayView = [[ViewerOverlayView alloc] initWithFrame:imageFrame
                                              contentView:imageContainer
                                             settingsView:_inspectorController.view
                                           inspectorWidth:kInspectorWidth
                                                debugView:debugController.view
                                                debugSize:kDebugSize
                                             mapPanelView:miniMapPanel];
  [_controller attachPanoramaView:view
                      overlayView:_overlayView
                    aspectFitView:imageContainer
                     miniMapPanel:miniMapPanel];

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
