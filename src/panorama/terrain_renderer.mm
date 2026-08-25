#include "terrain_renderer.h"

#include "gpu_image_renderer.h"
#include "png_writer.h"
#include "terrain_trace_session.h"
#include "timer.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace panorama {
namespace {

/// Validate output choices before starting an otherwise expensive trace.
void validate_output_configuration(const TerrainRenderOutputs &outputs) {
  if (!outputs.write_diagnostics && !outputs.write_synthetic) {
    throw std::invalid_argument("At least one render output must be enabled");
  }
  if (!std::isfinite(outputs.scalar_colour_range.minimum) ||
      !std::isfinite(outputs.scalar_colour_range.maximum) ||
      outputs.scalar_colour_range.maximum <= outputs.scalar_colour_range.minimum) {
    throw std::invalid_argument("Scalar colour range must be finite and increasing");
  }
  if (!outputs.write_synthetic) {
    return;
  }
  const SyntheticRenderOptions &synthetic = outputs.synthetic_options;
  if (!std::isfinite(synthetic.sun_azimuth) || !std::isfinite(synthetic.sun_elevation) ||
      !std::isfinite(synthetic.ambient_light) || synthetic.ambient_light < 0.0F ||
      synthetic.ambient_light > 1.0F || !std::isfinite(synthetic.diffusivity) ||
      synthetic.diffusivity < 0.0F || synthetic.diffusivity > 1.0F ||
      !std::isfinite(synthetic.ambient_detail) || synthetic.ambient_detail < 0.0F ||
      synthetic.ambient_detail > 1.0F || !std::isfinite(synthetic.feature_outline_detail) ||
      synthetic.feature_outline_detail < 0.0F || synthetic.feature_outline_detail > 1.0F ||
      static_cast<uint32_t>(synthetic.colour_source) >
          static_cast<uint32_t>(TerrainColourSource::Elevation) ||
      static_cast<uint32_t>(synthetic.colourmap) >
          static_cast<uint32_t>(PresetColourmap::Viewfinder) ||
      static_cast<uint32_t>(synthetic.colour_scale) >
          static_cast<uint32_t>(ScalarColourScale::Quadratic)) {
    throw std::invalid_argument("Synthetic render options are invalid");
  }
}

/// Render one GPU texture, read it back, and encode it for the CLI output path.
template <typename Render>
void render_and_write_png(
    Timer &timer,
    const std::filesystem::path &path,
    GpuImageRenderer &renderer,
    ImageSize image,
    Render &&render
) {
  render();
  const GpuImageReadback readback = renderer.readback(timer);

  timer.start_wall("PNG encoding");
  write_rgb_png(path, readback.bytes, image.width, image.height, readback.bytes_per_row);
  timer.stop("PNG encoding");
}

/// Present completed GPU trace buffers and write the selected PNG products.
void write_png_outputs(const TerrainRenderOutputs &outputs, TerrainTraceSession &trace) {
  const ImageSize image = trace.image();
  Timer timer("PNG generation");
  timer.start_wall("GPU presentation setup");
  GpuImageRenderer renderer(
      trace.device(),
      trace.command_queue(),
      trace.library(),
      image,
      GpuPresentationRequirements{
          .scalar_diagnostics = outputs.write_diagnostics,
          .normal_diagnostics = outputs.write_diagnostics && outputs.write_normal_diagnostic,
          .white_synthetic = outputs.write_synthetic &&
                             outputs.synthetic_options.colour_source == TerrainColourSource::White,
          .synthetic_scalar_colour =
              outputs.write_synthetic &&
              outputs.synthetic_options.colour_source != TerrainColourSource::White,
          .host_readback = true,
      }
  );
  timer.stop("GPU presentation setup");
  const id<MTLBuffer> distances = trace.distances();

  if (outputs.write_diagnostics) {
    render_and_write_png(timer, "distances.png", renderer, image, [&] {
      renderer.render_scalar(distances, outputs.scalar_colour_range, timer);
    });
    if (outputs.write_elevation_diagnostic) {
      render_and_write_png(timer, "elevations.png", renderer, image, [&] {
        renderer.render_scalar(trace.elevations(), outputs.scalar_colour_range, timer);
      });
    }
    if (outputs.write_normal_diagnostic) {
      render_and_write_png(timer, "normals.png", renderer, image, [&] {
        renderer.render_surface_normals(trace.surface_gradients(), distances, timer);
      });
    }
  }

  if (outputs.write_synthetic) {
    if (outputs.synthetic_options.raytraced_shadows) {
      trace.trace_shadows(
          outputs.synthetic_options.sun_azimuth,
          outputs.synthetic_options.sun_elevation
      );
    }
    const id<MTLBuffer> colour_values =
        outputs.synthetic_options.colour_source == TerrainColourSource::Elevation
            ? trace.elevations()
            : distances;
    render_and_write_png(timer, "synthetic.png", renderer, image, [&] {
      renderer.render_synthetic(
          trace.surface_gradients(),
          distances,
          trace.ray_directions(),
          colour_values,
          outputs.synthetic_options.raytraced_shadows ? trace.shadow_visibility() : nil,
          outputs.synthetic_options,
          outputs.scalar_colour_range,
          true,
          timer
      );
    });
  }
  timer.print();
}

} // namespace

void render_terrain(
    const RaytraceConfig &config,
    const RayField &field,
    const TerrainRenderOutputs &outputs
) {
  validate_output_configuration(outputs);
  @autoreleasepool {
    TerrainTraceSession trace(
        config,
        field,
        GpuTraceOutputRequirements{
            .elevations = outputs.requires_elevations(),
            .surface_gradients = outputs.requires_normals(),
        }
    );
    trace.trace(field);

    // PNG encoding is a replaceable output backend and deliberately remains
    // outside the terrain session's cumulative trace timer.
    write_png_outputs(outputs, trace);
    trace.print_statistics();
  }
}

} // namespace panorama
