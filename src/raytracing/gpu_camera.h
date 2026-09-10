#pragma once

#include "ray_projection.h"
#include "tile_manager.h"

namespace panorama {

/// Validate only the small camera descriptor, never an image-sized CPU array.
uint32_t validate_camera_request(const RayFieldRequest &camera);

struct GpuCameraStatistics {
  uint64_t plan_updates = 0;
  uint64_t footprint_updates = 0;
  double preparation_wall_ms = 0;
  double preparation_gpu_ms = 0;
};

/// Serial session-owned camera kernels and GPU-selected source LOD plan.
/// Ray/footprint storage stays on the GPU; only LOD integers and errors are read.
class GpuCamera {
public:
  GpuCamera(
      id<MTLDevice> device,
      id<MTLCommandQueue> queue,
      id<MTLLibrary> library,
      const TileManager &tiles
  );
  ~GpuCamera();
  GpuCamera(const GpuCamera &) = delete;
  GpuCamera &operator=(const GpuCamera &) = delete;

  void prepare(
      const RayFieldRequest &camera,
      ObserverLocation observer,
      float lod_scale,
      TileManager &tiles
  );
  void encode_rays(id<MTLCommandBuffer> command, id<MTLBuffer> destination);
  /// Call only after the ray-producing command completes.
  void validate_completed_rays() const;
  GpuCameraStatistics statistics() const;
  /// Diagnostic/reference tests only; normal preparation never reads the angle.
  id<MTLBuffer> pixel_angle() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
