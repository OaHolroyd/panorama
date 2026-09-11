#pragma once

#include "crs.h"
#include "gpu_camera.h"
#include "metal_bvh_trace.h"
#include "ray_projection.h"
#include "raytrace_config.h"
#include "raytrace_gpu.h"
#include "terrain_catalogue.h"

#import <Metal/Metal.h>

#include <memory>
#include <optional>

namespace panorama {

/// Persistent terrain-tracing state shared by a sequence of camera views.
///
/// Catalogue discovery, Metal pipelines, preparation workers, and the
/// resident terrain/mipmap atlas live for the complete session. Each call to
/// `trace` replaces only the ray field and transient frontier, allowing turns,
/// resolution changes, and movement across retained catalogue tiles to reuse
/// cached terrain.
class TerrainTraceSession {
public:
  /// Reuse shared_queue and its device when rebuilding a viewer's catalogue.
  /// With no supplied queue, the session creates its own default device/queue.
  TerrainTraceSession(
      const RaytraceConfig &config,
      const RayFieldRequest &camera,
      GpuTraceOutputRequirements outputs,
      id<MTLCommandQueue> shared_queue = nil
  );

  TerrainTraceSession(const TerrainTraceSession &) = delete;
  TerrainTraceSession &operator=(const TerrainTraceSession &) = delete;
  ~TerrainTraceSession();

  /// Trace a new view, resizing only ray-dependent GPU buffers when necessary.
  void trace(const RayFieldRequest &camera);
  /// Finish the current prepared input, including streaming repairs, without regenerating rays.
  void trace_prepared();

  /// Append resident primary tracing to an uncommitted producer. Preparation
  /// may build scene structures synchronously. False encodes nothing.
  /// Do not mutate session resources until the producer has completed.
  bool encode_trace(
      id<MTLCommandBuffer> command,
      const RayFieldRequest &camera,
      float lod_footprint_scale = 1.0F
  );
  [[nodiscard]] GpuCameraStatistics camera_statistics() const;
  /// After producer completion, false requires synchronous trace_prepared() and
  /// a new presentation pass before publishing the image.
  bool complete_encoded_trace(id<MTLCommandBuffer> command);
  bool encode_shadows(id<MTLCommandBuffer> command, double azimuth, double elevation);
  bool complete_encoded_shadows(id<MTLCommandBuffer> command);

  /// Move the observer without rebuilding terrain resources when the new
  /// position belongs to any source retained by this session's catalogue.
  /// Returns false when the caller must construct a catalogue and session
  /// centred on terrain outside the current finite source set.
  [[nodiscard]] bool relocate_observer(ObserverLocation observer);

  /// Change the adaptive terrain-LOD scale for subsequent traces. Zero keeps
  /// the original terrain resolution for every source.
  void set_lod_scale(float lod_scale);

  /// Select bilinear or triangular collision patches and C1 or patch-local
  /// surface normals for subsequent traces without rebuilding terrain state.
  void set_collision_options(bool bilinear_collisions, bool c1_normals);

  /// Select primary tracing on the render-owning thread. Outputs and the
  /// software shadow path keep the same device, queue, and buffer ABI.
  void set_raytracer(Raytracer raytracer);
  [[nodiscard]] MetalBvhStatistics bvh_statistics() const;
  [[nodiscard]] TileManagerStatistics tile_statistics() const;

  /// Trace one directional sun ray from each eligible primary collision.
  /// Angles are radians; azimuth is clockwise from grid north and elevation
  /// is above the horizontal plane.
  void trace_shadows(double sun_azimuth, double sun_elevation);

  [[nodiscard]] ImageSize image() const;
  /// Return the projected coordinate system shared by the resident terrain.
  [[nodiscard]] Crs crs() const;
  /// Return the current projected observer position and absolute elevation.
  [[nodiscard]] ObserverLocation observer() const;
  /// Return the complete prepared-data footprint used by the minimap.
  [[nodiscard]] const TerrainCoverage &terrain_coverage() const;
  /// Sample full-resolution terrain through this session's TileManager.
  [[nodiscard]] std::optional<float> sample_terrain(double easting, double northing);
  /// Return the device on which terrain and presentation resources must live.
  [[nodiscard]] id<MTLDevice> device() const;
  /// Return the queue used for ordered primary tracing and presentation.
  [[nodiscard]] id<MTLCommandQueue> command_queue() const;
  /// Return the compiled library containing raytracing and rendering kernels.
  [[nodiscard]] id<MTLLibrary> library() const;
  /// Return the current per-pixel direction buffer.
  [[nodiscard]] id<MTLBuffer> ray_directions() const;
  /// Return horizontal collision distances; zero means no terrain hit.
  [[nodiscard]] id<MTLBuffer> distances() const;
  /// Return absolute collision elevations when requested at construction.
  [[nodiscard]] id<MTLBuffer> elevations() const;
  /// Return packed Float16 east/north gradients when requested at construction.
  [[nodiscard]] id<MTLBuffer> surface_gradients() const;
  /// Return number of steps taken by each ray when requested at construction.
  [[nodiscard]] id<MTLBuffer> num_steps() const;
  /// Return number of collision evaluations done by each ray when requested at construction.
  [[nodiscard]] id<MTLBuffer> num_evaluations() const;
  /// Return hard-shadow visibility for the current trace and sun direction.
  [[nodiscard]] id<MTLBuffer> shadow_visibility() const;

  /// Print cumulative cache, preparation, frontier, and timing statistics.
  void print_statistics() const;

  /// Print the last primary trace's wall time, GPU work, and BVH cache deltas.
  /// GPU traversal is a sum of command durations, not total GPU frame latency.
  void print_trace_statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
