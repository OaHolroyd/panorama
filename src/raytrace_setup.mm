#include "raytrace_setup.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "loaded_tile.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
namespace {

#ifndef PANORAMA_METALLIB_PATH
#define PANORAMA_METALLIB_PATH "obj/release/panorama.metallib"
#endif

constexpr const char *kMetallibPath = PANORAMA_METALLIB_PATH;

// This matches the Metal `float2` direction buffer element exactly: its
// direction is horizontal and uses the compass convention x = sin(a),
// y = cos(a). Keeping it as two scalars avoids platform-specific SIMD ABI.
struct HorizontalDirection {
  float x;
  float y;
};

// This scalar-only layout is mirrored exactly by RaytraceParameters in
// panorama.metal. The global tile origin has already been rebased before it
// reaches these float32 fields.
struct RaytraceParameters {
  float tile_x_min;
  float tile_y_min;
  float cell_size;
  float observer_elevation;
  uint32_t num_cell;
  uint32_t num_azimuth;
  uint32_t num_polar;
  float max_distance;
};

static_assert(sizeof(HorizontalDirection) == 2U * sizeof(float));
static_assert(sizeof(RaytraceParameters) == 8U * sizeof(uint32_t));

/// Print a Foundation error in the command-line form used by the host tools.
void print_error(NSString *context, NSError *error) {
  std::fprintf(
      stderr,
      "%s: %s\n",
      context.UTF8String,
      error.localizedDescription.UTF8String
  );
}

/// Reject a count that cannot be represented by the Metal `uint` interface.
void validate_configuration(const RaytraceConfig &config) {
  if (config.num_azimuth == 0U || config.num_polar == 0U) {
    throw std::invalid_argument("Ray counts must both be positive");
  }
  if (!std::isfinite(config.observer.easting) ||
      !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) ||
      !std::isfinite(config.azimuth_start) ||
      !std::isfinite(config.azimuth_end) ||
      !std::isfinite(config.polar_start) || !std::isfinite(config.polar_end) ||
      !std::isfinite(config.max_distance) || config.max_distance <= 0.0F) {
    throw std::invalid_argument("Raytrace configuration must be finite");
  }
}

/// Construct float32 compass directions from evenly spaced azimuth centres.
[[nodiscard]] std::vector<HorizontalDirection>
make_azimuth_directions(const RaytraceConfig &config) {
  std::vector<HorizontalDirection> directions(config.num_azimuth);
  const double azimuth_step = (config.azimuth_end - config.azimuth_start) /
                              static_cast<double>(config.num_azimuth);
  for (uint32_t index = 0; index < config.num_azimuth; ++index) {
    const double bearing = config.azimuth_start +
                           (static_cast<double>(index) + 0.5) * azimuth_step;
    directions[index] = {static_cast<float>(std::sin(bearing)),
                         static_cast<float>(std::cos(bearing))};
  }
  return directions;
}

/// Construct float32 vertical slopes from evenly spaced polar-angle centres.
[[nodiscard]] std::vector<float>
make_polar_slopes(const RaytraceConfig &config) {
  std::vector<float> slopes(config.num_polar);
  const double polar_step = (config.polar_end - config.polar_start) /
                            static_cast<double>(config.num_polar);
  for (uint32_t index = 0; index < config.num_polar; ++index) {
    const double angle =
        config.polar_start + (static_cast<double>(index) + 0.5) * polar_step;
    const double slope = std::tan(angle);
    if (!std::isfinite(slope) ||
        slope < static_cast<double>(std::numeric_limits<float>::lowest()) ||
        slope > static_cast<double>(std::numeric_limits<float>::max())) {
      throw std::invalid_argument(
          "Polar range produces an invalid float32 slope"
      );
    }
    slopes[index] = static_cast<float>(slope);
  }
  return slopes;
}

/// Check that a byte count fits Metal's NSUInteger buffer-length argument.
[[nodiscard]] NSUInteger checked_buffer_length(
    size_t element_count,
    size_t element_size,
    const char *name
) {
  if (element_count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error(std::string(name) + " buffer is too large");
  }
  return static_cast<NSUInteger>(element_count * element_size);
}

/// Allocate a shared Metal buffer or report a precise command-line error.
[[nodiscard]] id<MTLBuffer> make_buffer(
    id<MTLDevice> device,
    const void *data,
    NSUInteger length,
    const char *name
) {
  id<MTLBuffer> buffer =
      data == nullptr
          ? [device newBufferWithLength:length
                                options:MTLResourceStorageModeShared]
          : [device newBufferWithBytes:data
                                length:length
                               options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(
        std::string("Could not allocate ") + name + " Metal buffer"
    );
  }
  return buffer;
}

} // namespace

void prepare_single_tile_raytrace(const RaytraceConfig &config) {
  validate_configuration(config);
  if (config.tile_path.empty()) {
    throw std::invalid_argument("Raytrace tile path must not be empty");
  }

  // This prototype traces the always-present level-1 maximum cells.
  const LoadedTile tile = LoadedTile::load_tif(config.tile_path, false);

  // Do the large-coordinate subtraction in float64 first. Only these local
  // values cross the host/device boundary, preserving float32 cell precision.
  const double local_x_min = tile.lower_left_x - config.observer.easting;
  const double local_y_min = tile.lower_left_y - config.observer.northing;
  if (local_x_min < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_x_min > static_cast<double>(std::numeric_limits<float>::max()) ||
      local_y_min < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_y_min > static_cast<double>(std::numeric_limits<float>::max()) ||
      tile.delta > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation <
          static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation >
          static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::overflow_error("Raytrace geometry does not fit float32");
  }

  // Set up the ray directions
  const std::vector<HorizontalDirection> directions =
      make_azimuth_directions(config);
  const std::vector<float> polar_slopes = make_polar_slopes(config);
  const size_t num_ray =
      static_cast<size_t>(config.num_azimuth) * config.num_polar;

  // Wrap to pass to Metal kernel
  const RaytraceParameters parameters = {
      static_cast<float>(local_x_min),
      static_cast<float>(local_y_min),
      static_cast<float>(tile.delta),
      static_cast<float>(config.observer.elevation),
      tile.size,
      config.num_azimuth,
      config.num_polar,
      config.max_distance,
  };

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("No Metal device is available");
    }

    NSError *error = nil;
    NSURL *library_url =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:library_url
                                                 error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      throw std::runtime_error("Could not load the Metal library");
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@"trace_single_tile"];
    if (function == nil) {
      throw std::runtime_error("Kernel trace_single_tile is missing");
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      print_error(@"Could not create the raytrace pipeline", error);
      throw std::runtime_error("Could not create the raytrace pipeline");
    }

    id<MTLBuffer> heights = make_buffer(
        device,
        tile.level_1_cells.data(),
        checked_buffer_length(
            tile.level_1_cells.size(),
            sizeof(float),
            "level-1 heights"
        ),
        "level-1 heights"
    );
    id<MTLBuffer> azimuths = make_buffer(
        device,
        directions.data(),
        checked_buffer_length(
            directions.size(),
            sizeof(HorizontalDirection),
            "azimuth directions"
        ),
        "azimuth directions"
    );
    id<MTLBuffer> slopes = make_buffer(
        device,
        polar_slopes.data(),
        checked_buffer_length(
            polar_slopes.size(),
            sizeof(float),
            "polar slopes"
        ),
        "polar slopes"
    );
    id<MTLBuffer> distances = make_buffer(
        device,
        nullptr,
        checked_buffer_length(num_ray, sizeof(float), "distance output"),
        "distance output"
    );
    id<MTLBuffer> elevations = make_buffer(
        device,
        nullptr,
        checked_buffer_length(num_ray, sizeof(float), "elevation output"),
        "elevation output"
    );

    // This is the complete planned independent-ray ABI. Buffer indices must
    // remain in sync with panorama.metal before the TODO dispatch is enabled.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (queue == nil || command == nil || encoder == nil) {
      throw std::runtime_error("Could not create a Metal raytrace command");
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:heights offset:0 atIndex:0];
    [encoder setBuffer:azimuths offset:0 atIndex:1];
    [encoder setBuffer:slopes offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:distances offset:0 atIndex:4];
    [encoder setBuffer:elevations offset:0 atIndex:5];

    // TODO: dispatch one thread for each (azimuth, polar) ray, then commit
    // and read the distance/elevation fields. Traversal deliberately begins
    // in the next implementation stage.
    [encoder endEncoding];
  }

  std::printf(
      "Prepared %u level-1 cells and %u x %u rays for GPU tracing; "
      "kernel dispatch is not implemented yet.\n",
      tile.size,
      config.num_azimuth,
      config.num_polar
  );
}

} // namespace panorama
