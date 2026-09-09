#include "gpu_terrain_frame.h"
#include "timer.h"
#include <chrono>
#include <stdexcept>

namespace panorama {
GpuTerrainFrameTiming render_terrain_frame(
    TerrainTraceSession &trace,
    const RayField *field,
    GpuImageRenderer &image,
    const TerrainPresentationSettings &settings,
    const std::function<void(id<MTLCommandBuffer>)> &encode_dependent,
    const CameraRayRequest *camera
) {
  const auto started = std::chrono::steady_clock::now();
  GpuTerrainFrameTiming timing;
  if (field != nullptr && camera != nullptr)
    throw std::invalid_argument("Choose either CPU rays or a GPU camera");
  const auto repair = [&] {
    if (camera != nullptr)
      trace.trace_prepared();
    else
      trace.trace(*field);
  };
  Timer timer("Terrain producer");
  const bool shadows = settings.use_surface_normals && settings.appearance.raytraced_shadows;
  const auto make_command = [&] {
    auto command = [trace.command_queue() commandBuffer];
    if (command == nil)
      throw std::runtime_error("Could not allocate terrain producer");
    command.label = @"Terrain frame producer";
    return command;
  };
  auto command = make_command();
  bool encoded_primary = camera != nullptr
                             ? trace.encode_trace(command, *camera)
                             : field != nullptr && trace.encode_trace(command, *field);
  if ((field != nullptr || camera != nullptr) && !encoded_primary) {
    repair();
    timing.streamed = true;
  }
  const auto finish = [&](id<MTLCommandBuffer> producer) {
    [producer commit];
    [producer waitUntilCompleted];
    if (producer.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error(
          "Terrain producer failed: " + std::string(
                                            producer.error == nil
                                                ? "unknown Metal error"
                                                : producer.error.localizedDescription.UTF8String
                                        )
      );
    timing.gpu_milliseconds += 1000.0 * (producer.GPUEndTime - producer.GPUStartTime);
    ++timing.producer_submissions;
  };
  bool encoded_shadows = shadows && trace.encode_shadows(
                                        command,
                                        settings.appearance.sun_azimuth,
                                        settings.appearance.sun_elevation
                                    );
  if (shadows && !encoded_shadows) {
    // Normally only the mipmap backend reaches this branch. If a caller lacks
    // resident shadow outputs, finish its primary work before CPU scheduling.
    if (encoded_primary) {
      finish(command);
      if (!trace.complete_encoded_trace(command))
        repair();
      encoded_primary = false;
      command = make_command();
    }
    trace.trace_shadows(settings.appearance.sun_azimuth, settings.appearance.sun_elevation);
    timing.streamed = true;
  }
  image.resize(trace.image());
  image.begin_frame();
  try {
    const auto encode_image = [&](id<MTLCommandBuffer> producer) {
      id<MTLBuffer> colour = nil;
      switch (settings.appearance.colour_source) {
      case TerrainColourSource::White:
        break;
      case TerrainColourSource::Elevation:
        colour = trace.elevations();
        break;
      case TerrainColourSource::Distance:
        colour = trace.distances();
        break;
      case TerrainColourSource::NumSteps:
        colour = trace.num_steps();
        break;
      case TerrainColourSource::NumEvaluations:
        colour = trace.num_evaluations();
        break;
      }
      // All buffers/textures are tracked resources on one MTLCommandQueue.
      // Encoder boundaries preserve write/read dependencies, including declared
      // indirect BVH resources; no intermediate CPU wait is necessary.
      image.render_synthetic(
          trace.surface_gradients(),
          trace.distances(),
          trace.ray_directions(),
          colour,
          shadows ? trace.shadow_visibility() : nil,
          settings.appearance,
          settings.colour_range,
          settings.use_surface_normals,
          timer,
          producer
      );
      if (encode_dependent)
        encode_dependent(producer);
    };
    encode_image(command);
    finish(command);
    const bool primary_complete = !encoded_primary || trace.complete_encoded_trace(command);
    const bool shadows_complete = !encoded_shadows || trace.complete_encoded_shadows(command);
    if (!primary_complete || !shadows_complete) {
      timing.streamed = true;
      if (!primary_complete)
        repair();
      if (shadows)
        trace.trace_shadows(settings.appearance.sun_azimuth, settings.appearance.sun_elevation);
      command = make_command();
      encode_image(command);
      finish(command);
    }
  } catch (...) {
    image.cancel_frame();
    throw;
  }
  timing.wall_milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return timing;
}
} // namespace panorama
