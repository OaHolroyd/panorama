#pragma once

#include "crs.h"
#include "inspection_coordinates.h"
#include "metalfx_upscaler.h"
#include "peak_catalogue.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "solar_position.h"
#include "terrain_catalogue.h"
#include "terrain_presentation_settings.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <numbers>
#include <optional>
#include <string>

namespace panorama::app {

inline constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
inline constexpr double kDegreesToRadians = std::numbers::pi / 180.0;
inline constexpr double kRadiansToDegrees = 180.0 / std::numbers::pi;
inline constexpr double kDefaultVerticalFieldOfView = 70.0 * kDegreesToRadians;
inline constexpr double kDefaultSunAzimuthDegrees = 225.0;
inline constexpr double kDefaultSunAltitudeDegrees = 35.0;
inline constexpr double kMinimumMovementSpeed = 1.0;
inline constexpr double kMaximumRoamSpeed = 200.0;
inline constexpr double kMaximumCruiseSpeed = 10'000.0;
inline constexpr double kDefaultRoamSpeed = 20.0;
inline constexpr double kDefaultCruiseSpeed = 150.0;
inline constexpr double kCruiseSteeringDeadZone = 0.06;
inline constexpr double kCruiseSteeringExponent = 1.5;
inline constexpr double kCruiseMaximumYawRate = 90.0 * kDegreesToRadians;
inline constexpr double kCruiseMaximumPitchRate = 60.0 * kDegreesToRadians;
inline constexpr double kAircraftMaximumBank = 55.0 * kDegreesToRadians;
inline constexpr double kAircraftBankResponseSeconds = 0.5;
inline constexpr double kAircraftTrimResponseSeconds = 8.0;
inline constexpr double kGravity = 9.80665;
inline constexpr float kDefaultSkyStrength = 0.28F;
inline constexpr float kDefaultSkyDetail = 0.65F;
inline constexpr float kDefaultDiffusivity = 1.0F;

struct ViewerSettings {
  std::filesystem::path tile_dir = "data/swissalti3d-10-level-0-metal-u16-none-lod-point";
  std::filesystem::path peak_gazetteer = "data/gazetteers/peaks.csv";
  uint64_t tile_cache_size_bytes = 2048ULL * kBytesPerMiB;
  uint32_t workers = 8U;
  float max_distance = 600'000.0F;
  float lod_scale = 1.5F;
  Raytracer raytracer = Raytracer::MetalBvh;
  uint32_t bvh_block_cells = 4U;
  uint64_t bvh_cache_size_bytes = 2048ULL * kBytesPerMiB;
  bool discard_quantized = false;
  bool trace_diagnostics = false;
  bool bilinear_collisions = false;
  bool c1_normals = true;
  ObserverLocation observer = {2623452.4, 1100502.2, 3415.0};
  ImageSize image = {1600U, 900U};
  MetalFxActivation metalfx_activation = MetalFxActivation::Disabled;
  MetalFxPreset metalfx_preset = MetalFxPreset::Balanced;
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
  bool map_selected;
};

struct MapCoordinate {
  double easting;
  double northing;
};

struct TerrainPoint {
  double easting;
  double northing;
  float elevation;
};

enum class MapPointAction : uint8_t { None, Hover, Look, MoveObserver };
enum class RoamKey : uint8_t { Forward, Backward, Left, Right };
enum class RoamAltitudeMode : uint8_t { FollowTerrain, HoldAltitude };
enum class PointerOwner : uint8_t { None, Panorama, Minimap, Overlay };

struct RoamResult {
  uint64_t request_token;
  bool accepted;
  float ground_elevation;
};

struct TargetVisibility {
  uint64_t revision;
  uint64_t request_token;
  bool occluded;
};

struct LockedPointProjection {
  bool onscreen;
  double pixel_x;
  double pixel_y;
  double direction_x;
  double direction_y;
};

struct PresentedFrame {
  id<MTLTexture> texture;
  id<MTLBuffer> visibility_points;
  ImageSize image;
  ImageSize output_image;
  bool metalfx_enabled;
  MetalFxPreset metalfx_preset;
  CameraOrientation orientation;
  double vertical_field_of_view;
  uint64_t revision;
  double milliseconds;
  double gpu_milliseconds;
  bool streamed;
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
  std::optional<PeakLabelFrame> peak_labels;
  std::optional<RoamResult> roam_result;
  uint64_t roam_result_sequence;
};

[[nodiscard]] ViewerSettings parse_arguments(int argc, const char *argv[]);
[[nodiscard]] std::optional<double> parse_range_value(NSString *input);
[[nodiscard]] std::optional<uint32_t> parse_image_dimension(NSString *input);
[[nodiscard]] NSString *format_range_value(double value);
[[nodiscard]] NSString *format_clock_minutes(double value);
[[nodiscard]] std::optional<CalendarDateTime>
local_date_time_to_utc(CalendarDateTime local, NSTimeZone *timeZone);
[[nodiscard]] NSString *
format_local_daylight_time(CalendarDateTime date, double utcMinutes, NSTimeZone *timeZone);
[[nodiscard]] NSString *format_time_zone_summary(NSTimeZone *timeZone, CalendarDateTime utc);
[[nodiscard]] LockedPointProjection project_locked_point(
    TerrainPoint point,
    ObserverLocation observer,
    ImageSize image,
    double vertical_field_of_view,
    CameraOrientation orientation
);

class ViewerRenderer {
public:
  virtual ~ViewerRenderer() = default;
  virtual void request_minimap_enabled(bool enabled) = 0;
  virtual void request_peak_labels_enabled(bool enabled) = 0;
  virtual void request_view(CameraOrientation, double, ImageSize) = 0;
  virtual void request_metalfx(MetalFxActivation, MetalFxPreset, bool) = 0;
  virtual void request_raytracer(Raytracer) = 0;
  [[nodiscard]] virtual Raytracer requested_raytracer() const = 0;
  virtual void request_presentation(TerrainPresentationSettings) = 0;
  virtual void request_lod_scale(float) = 0;
  virtual void request_collision_settings(bool, bool) = 0;
  virtual uint64_t request_inspection(std::optional<InspectionLocation>) = 0;
  virtual uint64_t request_map_point(MapCoordinate) = 0;
  virtual uint64_t request_target_visibility(std::optional<TerrainPoint>) = 0;
  virtual void request_observer_at(TerrainPoint, double) = 0;
  virtual void request_ground_clearance(double) = 0;
  virtual uint64_t request_roam(MapCoordinate, RoamAltitudeMode, double) = 0;
  [[nodiscard]] virtual PresentedFrame presented_frame() const = 0;
  [[nodiscard]] virtual bool
  submit_presentation(id<MTLCommandBuffer>, PresentedFrame &, id<CAMetalDrawable>) const = 0;
  [[nodiscard]] virtual id<MTLDevice> device() const = 0;
  [[nodiscard]] virtual id<MTLCommandQueue> command_queue() const = 0;
  [[nodiscard]] virtual id<MTLLibrary> library() const = 0;
  [[nodiscard]] virtual ImageSize initial_image() const = 0;
  [[nodiscard]] virtual ObserverLocation observer() const = 0;
  [[nodiscard]] virtual Crs terrain_crs() const = 0;
  [[nodiscard]] virtual const TerrainCoverage &terrain_coverage() const = 0;
  [[nodiscard]] virtual bool observer_used_fallback() const = 0;
  [[nodiscard]] virtual double ground_clearance() const = 0;
  [[nodiscard]] virtual double initial_vertical_field_of_view() const = 0;
  [[nodiscard]] virtual float max_distance() const = 0;
  [[nodiscard]] virtual float initial_lod_scale() const = 0;
  [[nodiscard]] virtual MetalFxActivation initial_metalfx_activation() const = 0;
  [[nodiscard]] virtual MetalFxPreset initial_metalfx_preset() const = 0;
  [[nodiscard]] virtual bool metalfx_supported() const = 0;
  [[nodiscard]] virtual bool peak_labels_available() const = 0;
  [[nodiscard]] virtual bool initial_bilinear_collisions() const = 0;
  [[nodiscard]] virtual bool initial_c1_normals() const = 0;
  [[nodiscard]] virtual CameraOrientation initial_orientation() const = 0;
  [[nodiscard]] virtual TerrainPresentationSettings initial_presentation() const = 0;
};

[[nodiscard]] std::unique_ptr<ViewerRenderer> make_viewer_renderer(ViewerSettings settings);

} // namespace panorama::app
