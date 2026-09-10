#include "gpu_camera.h"
#include "gpu_camera_types.metalh"
#include "trace_activity.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace panorama {
static_assert(sizeof(camera_gpu::Camera) == 100);
static_assert(sizeof(camera_gpu::Source) == 12);
static_assert(sizeof(camera_gpu::Lod) == 24);
static_assert(sizeof(RayDirection) == 20);
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
float checked_float(double value) {
  const float result = static_cast<float>(value);
  if (!std::isfinite(result))
    throw std::invalid_argument("GPU camera parameters exceed finite Float32 range");
  return result;
}
camera_gpu::Camera uniforms(const RayFieldRequest &request) {
  validate_camera_request(request);
  if (const auto *angular = std::get_if<AngularProjection>(&request.projection)) {
    camera_gpu::Camera result{};
    result.width = request.image.width;
    result.height = request.image.height;
    result.angular = 1;
    result.azimuth_start = checked_float(angular->azimuth_start);
    result.azimuth_step =
        checked_float((angular->azimuth_end - angular->azimuth_start) / request.image.width);
    result.elevation_start = checked_float(angular->elevation_start);
    result.elevation_step =
        checked_float((angular->elevation_end - angular->elevation_start) / request.image.height);
    return result;
  }
  const auto &projection = std::get<CameraProjection>(request.projection);
  const auto &intrinsics = projection.intrinsics;
  const auto &orientation = projection.orientation;
  const double sh = std::sin(orientation.heading), ch = std::cos(orientation.heading);
  const double sp = std::sin(orientation.pitch), cp = std::cos(orientation.pitch);
  const double sr = std::sin(orientation.roll), cr = std::cos(orientation.roll);
  camera_gpu::Camera result{
      request.image.width,
      request.image.height,
      checked_float(intrinsics.focal_x),
      checked_float(intrinsics.focal_y),
      checked_float(intrinsics.principal_x),
      checked_float(intrinsics.principal_y),
      {float(cp * sh), float(cp * ch), float(sp)},
      {float(cr * ch - sr * sp * sh), float(-cr * sh - sr * sp * ch), float(sr * cp)},
      {float(-sr * ch - cr * sp * sh), float(sr * sh - cr * sp * ch), float(cr * cp)},
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0};
  if (const auto *d = std::get_if<BrownConradyDistortion>(&projection.distortion)) {
    result.radial_1 = checked_float(d->radial_1);
    result.radial_2 = checked_float(d->radial_2);
    result.radial_3 = checked_float(d->radial_3);
    result.tangential_1 = checked_float(d->tangential_1);
    result.tangential_2 = checked_float(d->tangential_2);
  }
  return result;
}
bool same_projection(const RayFieldRequest &a, const RayFieldRequest &b) {
  auto x = uniforms(a), y = uniforms(b);
  // The footprint depends on the lens/projection, not the world orientation.
  std::fill_n(x.forward, 3, 0);
  std::fill_n(y.forward, 3, 0);
  std::fill_n(x.right, 3, 0);
  std::fill_n(y.right, 3, 0);
  std::fill_n(x.up, 3, 0);
  std::fill_n(y.up, 3, 0);
  return std::memcmp(&x, &y, sizeof(x)) == 0;
}
} // namespace

uint32_t validate_camera_request(const RayFieldRequest &camera) {
  const uint64_t count = uint64_t(camera.image.width) * camera.image.height;
  if (count == 0 || count > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("GPU camera has invalid image dimensions");
  if (const auto *angular = std::get_if<AngularProjection>(&camera.projection)) {
    for (double value : {angular->azimuth_start,
                         angular->azimuth_end,
                         angular->elevation_start,
                         angular->elevation_end})
      (void)checked_float(value);
    const float dx =
        checked_float((angular->azimuth_end - angular->azimuth_start) / camera.image.width);
    const float dy =
        checked_float((angular->elevation_end - angular->elevation_start) / camera.image.height);
    if (dx == 0 || dy == 0)
      throw std::invalid_argument("Angular projection requires a positive pixel footprint");
    return static_cast<uint32_t>(count);
  }
  const auto &p = std::get<CameraProjection>(camera.projection);
  if (const auto *d = std::get_if<BrownConradyDistortion>(&p.distortion))
    for (double value : {d->radial_1, d->radial_2, d->radial_3, d->tangential_1, d->tangential_2})
      (void)checked_float(value);
  for (double value : {p.orientation.heading,
                       p.orientation.pitch,
                       p.orientation.roll,
                       p.intrinsics.focal_x,
                       p.intrinsics.focal_y,
                       p.intrinsics.principal_x,
                       p.intrinsics.principal_y})
    (void)checked_float(value);
  const float fx = float(p.intrinsics.focal_x), fy = float(p.intrinsics.focal_y);
  if (!(fx > 0) || !(fy > 0) || !std::isfinite(1 / fx) || !std::isfinite(1 / fy))
    throw std::invalid_argument("GPU camera requires finite positive focal lengths");
  // Keep all shader projection arithmetic finite, including adjacent-angle products.
  const double extent = std::max(
      {std::abs(p.intrinsics.principal_x / fx),
       std::abs((double(camera.image.width) - p.intrinsics.principal_x) / fx),
       std::abs(p.intrinsics.principal_y / fy),
       std::abs((double(camera.image.height) - p.intrinsics.principal_y) / fy)}
  );
  if (extent > 1e12)
    throw std::invalid_argument("GPU camera projection is outside the supported numeric range");
  return static_cast<uint32_t>(count);
}

struct GpuCamera::State {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> rays, footprint, reduce, lod;
  id<MTLBuffer> sources, partial, angle, levels, invalid;
  const TileManager *owner;
  double anchor_x, anchor_y;
  camera_gpu::Camera camera{};
  camera_gpu::Lod settings{};
  RayFieldRequest cached{};
  ObserverLocation cached_observer{};
  float cached_scale = -1;
  bool footprint_valid = false, plan_valid = false;
  GpuCameraStatistics stats{};

  id<MTLBuffer> buffer(NSUInteger length, NSString *label) {
    auto result = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    if (result == nil)
      throw std::runtime_error("Could not allocate GPU camera buffer");
    result.label = label;
    return result;
  }
  id<MTLComputeCommandEncoder>
  encoder(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline) {
    auto result = [command computeCommandEncoder];
    if (result == nil)
      throw std::runtime_error("Could not create GPU camera encoder");
    [result setComputePipelineState:pipeline];
    return result;
  }
};

GpuCamera::GpuCamera(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library,
    const TileManager &tiles
)
    : state_(std::make_unique<State>()) {
  auto &s = *state_;
  s.device = device;
  s.queue = queue;
  s.owner = &tiles;
  const auto pipeline = [&](NSString *name) {
    auto function = [library newFunctionWithName:name];
    NSError *error = nil;
    auto result = [device newComputePipelineStateWithFunction:function error:&error];
    if (result == nil || result.maxTotalThreadsPerThreadgroup < 256)
      throw std::runtime_error(
          "Could not create GPU camera pipeline: " + std::string(name.UTF8String)
      );
    return result;
  };
  s.rays = pipeline(@"generate_camera_rays");
  s.footprint = pipeline(@"camera_pixel_footprint");
  s.reduce = pipeline(@"reduce_camera_footprint");
  s.lod = pipeline(@"select_camera_lods");
  const auto &grid = tiles.catalogue().grid();
  const auto origin = tiles.sources().front().key;
  s.anchor_x = grid.origin_x + double(origin.column) * grid.width;
  s.anchor_y = grid.origin_y - (double(origin.row) + 1) * grid.width;
  s.settings = {0,
                0,
                checked_float(grid.width),
                checked_float(tiles.origin_geometry().cell_size),
                0,
                static_cast<uint32_t>(tiles.sources().size())};
  s.sources = s.buffer(tiles.sources().size() * sizeof(camera_gpu::Source), @"GPU LOD sources");
  auto *sources = static_cast<camera_gpu::Source *>(s.sources.contents);
  for (size_t i = 0; i < tiles.sources().size(); ++i) {
    const auto &source = tiles.sources()[i];
    sources[i] = {checked_float((double(source.key.column) - double(origin.column)) * grid.width),
                  checked_float((double(origin.row) - double(source.key.row)) * grid.width),
                  source.lod_count};
  }
  s.angle = s.buffer(sizeof(float), @"GPU pixel footprint");
  s.levels = s.buffer(tiles.sources().size() * sizeof(uint32_t), @"GPU LOD decisions");
  s.invalid = s.buffer(sizeof(uint32_t), @"GPU camera error");
}
GpuCamera::~GpuCamera() = default;

void GpuCamera::prepare(
    const RayFieldRequest &camera,
    ObserverLocation observer,
    float scale,
    TileManager &tiles
) {
  auto &s = *state_;
  trace_activity::Scope activity("GPU camera/LOD preparation");
  const auto started = Clock::now();
  s.stats.preparation_gpu_ms = 0;
  if (&tiles != s.owner || !std::isfinite(scale) || scale < 0)
    throw std::invalid_argument("GPU LOD plan belongs to a different catalogue or invalid scale");
  s.camera = uniforms(camera);
  const bool footprint_changed = !s.footprint_valid || !same_projection(camera, s.cached);
  const bool plan_changed = footprint_changed || !s.plan_valid || scale != s.cached_scale ||
                            observer.easting != s.cached_observer.easting ||
                            observer.northing != s.cached_observer.northing;
  if (plan_changed) {
    // A failed preparation must never leave a stale cache key usable.
    s.plan_valid = false;
    if (footprint_changed)
      s.footprint_valid = false;
    s.settings.observer_x = checked_float(observer.easting - s.anchor_x);
    s.settings.observer_y = checked_float(observer.northing - s.anchor_y);
    s.settings.scale = scale;
    auto command = [s.queue commandBuffer];
    if (command == nil)
      throw std::runtime_error("Could not create camera preparation command");
    command.label = @"GPU camera footprint and LOD";
    if (footprint_changed) {
      const uint32_t groups =
          uint32_t((uint64_t(camera.image.width) * camera.image.height + 255) / 256);
      if (s.partial == nil || s.partial.length < size_t(groups) * sizeof(float))
        s.partial = s.buffer(size_t(groups) * sizeof(float), @"GPU footprint partials");
      auto encoder = s.encoder(command, s.footprint);
      [encoder setBytes:&s.camera length:sizeof(s.camera) atIndex:0];
      [encoder setBuffer:s.partial offset:0 atIndex:1];
      [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      encoder = s.encoder(command, s.reduce);
      [encoder setBuffer:s.partial offset:0 atIndex:0];
      [encoder setBytes:&groups length:sizeof(groups) atIndex:1];
      [encoder setBytes:&s.camera length:sizeof(s.camera) atIndex:2];
      [encoder setBuffer:s.angle offset:0 atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
    }
    auto encoder = s.encoder(command, s.lod);
    [encoder setBuffer:s.sources offset:0 atIndex:0];
    [encoder setBytes:&s.settings length:sizeof(s.settings) atIndex:1];
    [encoder setBuffer:s.angle offset:0 atIndex:2];
    [encoder setBuffer:s.levels offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(s.settings.count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error("GPU camera LOD preparation failed");
    if (trace_activity::current != nullptr) {
      const auto *levels = static_cast<const uint32_t *>(s.levels.contents);
      uint32_t finer = 0, coarser = 0, finest = 0;
      for (uint32_t source = 0; source < s.settings.count; ++source) {
        const uint32_t previous = tiles.lod_for_source(source);
        finer += levels[source] < previous;
        coarser += levels[source] > previous;
        finest += levels[source] == 1U;
      }
      std::printf(
          "LOD plan: sources=%u, changed finer/coarser=%u/%u, selected LOD-1=%u\n",
          s.settings.count,
          finer,
          coarser,
          finest
      );
    }
    tiles.install_lod_plan({static_cast<const uint32_t *>(s.levels.contents), s.settings.count});
    s.stats.preparation_gpu_ms = 1000 * (command.GPUEndTime - command.GPUStartTime);
    ++s.stats.plan_updates;
    s.stats.footprint_updates += footprint_changed ? 1U : 0U;
    s.cached = camera;
    s.cached_observer = observer;
    s.cached_scale = scale;
    s.footprint_valid = s.plan_valid = true;
  }
  s.stats.preparation_wall_ms = elapsed(started);
}

void GpuCamera::encode_rays(id<MTLCommandBuffer> command, id<MTLBuffer> destination) {
  auto &s = *state_;
  if (command == nil ||
      destination.length < uint64_t(s.camera.width) * s.camera.height * sizeof(RayDirection))
    throw std::invalid_argument("GPU camera ray destination has invalid dimensions");
  auto clear = [command blitCommandEncoder];
  if (clear == nil)
    throw std::runtime_error("Could not clear GPU camera status");
  [clear fillBuffer:s.invalid range:NSMakeRange(0, sizeof(uint32_t)) value:0];
  [clear endEncoding];
  auto encoder = s.encoder(command, s.rays);
  [encoder setBytes:&s.camera length:sizeof(s.camera) atIndex:0];
  [encoder setBuffer:destination offset:0 atIndex:1];
  [encoder setBuffer:s.invalid offset:0 atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(uint64_t(s.camera.width) * s.camera.height, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
}
void GpuCamera::validate_completed_rays() const {
  if (*static_cast<const uint32_t *>(state_->invalid.contents) != 0)
    throw std::runtime_error("Camera projection produced a vertical or non-finite terrain ray");
}
GpuCameraStatistics GpuCamera::statistics() const { return state_->stats; }
id<MTLBuffer> GpuCamera::pixel_angle() const { return state_->angle; }
} // namespace panorama
