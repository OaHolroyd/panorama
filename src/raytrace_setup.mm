#include "raytrace_setup.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "loaded_tile.h"
#include "png_writer.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
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
  uint32_t num_levels;
  uint32_t num_cell;
  uint32_t num_azimuth;
  uint32_t num_polar;
  float max_distance;
};

static_assert(sizeof(HorizontalDirection) == 2U * sizeof(float));
static_assert(sizeof(RaytraceParameters) == 9U * sizeof(uint32_t));

/// Print a Foundation error in the command-line form used by the host tools.
void print_error(NSString *context, NSError *error) {
  std::fprintf(stderr, "%s: %s\n", context.UTF8String, error.localizedDescription.UTF8String);
}

/// Return whether Metal's capture layer was enabled before this process began.
[[nodiscard]] bool capture_requested() {
  const char *value = std::getenv("MTL_CAPTURE_ENABLED");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

/// Start a queue-scoped GPU trace when Metal's capture layer is enabled.
[[nodiscard]] bool start_capture_if_requested(id<MTLCommandQueue> queue) {
  if (!capture_requested()) {
    return false;
  }

  NSString *path = [[NSFileManager.defaultManager currentDirectoryPath]
      stringByAppendingPathComponent:@"panorama.gputrace"];
  if ([NSFileManager.defaultManager fileExistsAtPath:path]) {
    throw std::runtime_error(
        "Refusing to overwrite panorama.gputrace; move or remove the existing capture first"
    );
  }

  MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
  descriptor.captureObject = queue;
  descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
  descriptor.outputURL = [NSURL fileURLWithPath:path];
  NSError *error = nil;
  if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor
                                                                      error:&error]) {
    print_error(@"Could not start the Metal GPU capture", error);
    throw std::runtime_error("Could not start the Metal GPU capture");
  }
  std::printf("Capturing GPU work to %s\n", path.UTF8String);
  return true;
}

/// Reject a count that cannot be represented by the Metal `uint` interface.
void validate_configuration(const RaytraceConfig &config) {
  if (config.num_azimuth == 0U || config.num_polar == 0U) {
    throw std::invalid_argument("Ray counts must both be positive");
  }
  if (!std::isfinite(config.observer.easting) || !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) || !std::isfinite(config.azimuth_start) ||
      !std::isfinite(config.azimuth_end) || !std::isfinite(config.polar_start) ||
      !std::isfinite(config.polar_end) || !std::isfinite(config.max_distance) ||
      config.max_distance <= 0.0F) {
    throw std::invalid_argument("Raytrace configuration must be finite");
  }
}

/// Construct float32 compass directions from evenly spaced azimuth centres.
[[nodiscard]] std::vector<HorizontalDirection>
make_azimuth_directions(const RaytraceConfig &config) {
  std::vector<HorizontalDirection> directions(config.num_azimuth);
  const double azimuth_step =
      (config.azimuth_end - config.azimuth_start) / static_cast<double>(config.num_azimuth);
  for (uint32_t index = 0; index < config.num_azimuth; ++index) {
    const double bearing = config.azimuth_start + (static_cast<double>(index) + 0.5) * azimuth_step;
    directions[index] = {static_cast<float>(std::sin(bearing)),
                         static_cast<float>(std::cos(bearing))};
  }
  return directions;
}

/// Construct float32 vertical slopes from evenly spaced polar-angle centres.
[[nodiscard]] std::vector<float> make_polar_slopes(const RaytraceConfig &config) {
  std::vector<float> slopes(config.num_polar);
  const double polar_step =
      (config.polar_end - config.polar_start) / static_cast<double>(config.num_polar);
  for (uint32_t index = 0; index < config.num_polar; ++index) {
    const double angle = config.polar_start + (static_cast<double>(index) + 0.5) * polar_step;
    const double slope = std::tan(angle);
    if (!std::isfinite(slope) ||
        slope < static_cast<double>(std::numeric_limits<float>::lowest()) ||
        slope > static_cast<double>(std::numeric_limits<float>::max())) {
      throw std::invalid_argument("Polar range produces an invalid float32 slope");
    }
    slopes[index] = static_cast<float>(slope);
  }
  return slopes;
}

/// Check that a byte count fits Metal's NSUInteger buffer-length argument.
[[nodiscard]] NSUInteger
checked_buffer_length(size_t element_count, size_t element_size, const char *name) {
  if (element_count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error(std::string(name) + " buffer is too large");
  }
  return static_cast<NSUInteger>(element_count * element_size);
}

/// Allocate a shared Metal buffer or report a precise command-line error.
/// If `data == nullptr` then allocate a new buffer otherwise link it to the
/// supplied data.
[[nodiscard]] id<MTLBuffer>
make_buffer(id<MTLDevice> device, const void *data, NSUInteger length, const char *name) {
  id<MTLBuffer> buffer = data == nullptr ? [device newBufferWithLength:length
                                                               options:MTLResourceStorageModeShared]
                                         : [device newBufferWithBytes:data
                                                               length:length
                                                              options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(std::string("Could not allocate ") + name + " Metal buffer");
  }
  return buffer;
}

/// Fill a shared Metal buffer with zero bytes before its first GPU dispatch.
void clear_buffer(id<MTLBuffer> buffer, const char *name) {
  void *contents = buffer.contents;
  if (contents == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name + " Metal buffer");
  }
  std::memset(contents, 0, buffer.length);
}

/// Return a non-owning float32 view of a completed shared Metal buffer.
[[nodiscard]] std::span<const float>
view_float_buffer(id<MTLBuffer> buffer, size_t num_value, const char *name) {
  const auto *contents = static_cast<const float *>(buffer.contents);
  if (contents == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name + " Metal buffer");
  }
  return {contents, num_value};
}

} // namespace

void perform_single_tile_raytrace(const RaytraceConfig &config) {
  validate_configuration(config);
  if (config.tile_path.empty()) {
    throw std::invalid_argument("Raytrace tile path must not be empty");
  }

  // Level-0 vertices provide exact bilinear intersections; the loader derives
  // the accompanying level-1 maximum field used for cheap cell rejection.
  LoadedTile tile = LoadedTile::load_tif(config.tile_path, true);
  if (tile.vertices == nullptr) {
    throw std::logic_error("Level-0 raytrace tile has no vertex elevations");
  }
  // Keep the complete flat maximum hierarchy resident alongside the exact
  // vertices. The current kernel still reads level 1 only; later adaptive DDA
  // traversal will index the appended coarser levels in this same buffer.
  tile.compute_mipmap();

  // Do the large-coordinate subtraction in float64 first. Only these local
  // values cross the host/device boundary, preserving float32 cell precision.
  const double local_x_min = tile.lower_left_x - config.observer.easting;
  const double local_y_min = tile.lower_left_y - config.observer.northing;
  if (local_x_min < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_x_min > static_cast<double>(std::numeric_limits<float>::max()) ||
      local_y_min < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_y_min > static_cast<double>(std::numeric_limits<float>::max()) ||
      tile.delta > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::overflow_error("Raytrace geometry does not fit float32");
  }

  // Set up the ray directions
  const std::vector<HorizontalDirection> directions = make_azimuth_directions(config);
  const std::vector<float> polar_slopes = make_polar_slopes(config);
  const size_t num_ray = static_cast<size_t>(config.num_azimuth) * config.num_polar;

  // Wrap to pass to Metal kernel
  const RaytraceParameters parameters = {
      static_cast<float>(local_x_min),
      static_cast<float>(local_y_min),
      static_cast<float>(tile.delta),
      static_cast<float>(config.observer.elevation),
      tile.num_levels,
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
    NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      throw std::runtime_error("Could not load the Metal library");
    }
    id<MTLFunction> trace_single_tile_kernel = [library newFunctionWithName:@"trace_single_tile"];
    if (trace_single_tile_kernel == nil) {
      throw std::runtime_error("Kernel trace_single_tile is missing");
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:trace_single_tile_kernel error:&error];
    if (pipeline == nil) {
      print_error(@"Could not create the raytrace pipeline", error);
      throw std::runtime_error("Could not create the raytrace pipeline");
    }

    id<MTLBuffer> heights = make_buffer(
        device,
        tile.mipmap.data(),
        checked_buffer_length(tile.mipmap.size(), sizeof(float), "maximum mipmap"),
        "maximum mipmap"
    );
    id<MTLBuffer> vertices = make_buffer(
        device,
        tile.vertices->data(),
        checked_buffer_length(tile.vertices->size(), sizeof(float), "level-0 vertex heights"),
        "level-0 vertex heights"
    );
    id<MTLBuffer> azimuths = make_buffer(
        device,
        directions.data(),
        checked_buffer_length(directions.size(), sizeof(HorizontalDirection), "azimuth directions"),
        "azimuth directions"
    );
    id<MTLBuffer> slopes = make_buffer(
        device,
        polar_slopes.data(),
        checked_buffer_length(polar_slopes.size(), sizeof(float), "polar slopes"),
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
    // Overwrite with zeros
    clear_buffer(distances, "distance output");
    clear_buffer(elevations, "elevation output");

    // Setup the queue/command/encoder
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      throw std::runtime_error("Could not create a Metal raytrace command queue");
    }
    // A GPU trace begins at command-buffer creation, not merely encoding.
    const bool capture_active = start_capture_if_requested(queue);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      if (capture_active) {
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
      }
      throw std::runtime_error("Could not create a Metal raytrace command");
    }
    // Instruments displays these labels in its GPU command and encoder lists.
    command.label = @"single-tile raytrace";
    encoder.label = @"trace_single_tile";
    // Physical X is polar and physical Y is azimuth. A 32-wide X row places
    // adjacent SIMD lanes on one horizontal DDA path with different slopes.
    // The output buffer still uses its ordinary (polar, azimuth) layout.
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:heights offset:0 atIndex:0];
    [encoder setBuffer:vertices offset:0 atIndex:1];
    [encoder setBuffer:azimuths offset:0 atIndex:2];
    [encoder setBuffer:slopes offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBuffer:distances offset:0 atIndex:5];
    [encoder setBuffer:elevations offset:0 atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(config.num_polar, config.num_azimuth, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

    [encoder endEncoding];

    // Start the GPU work
    [command commit];

    // Finish the GPU work
    [command waitUntilCompleted];
    if (capture_active) {
      [[MTLCaptureManager sharedCaptureManager] stopCapture];
    }
    if (command.status == MTLCommandBufferStatusError) {
      print_error(@"The Metal raytrace command failed", command.error);
      throw std::runtime_error("The Metal raytrace command failed");
    }
    const double gpu_milliseconds = 1'000.0 * (command.GPUEndTime - command.GPUStartTime);
    std::printf("GPU raytrace: %.3f ms\n", gpu_milliseconds);

    // Shared storage is now coherent with the CPU, so ImageIO can read these
    // views directly. The buffers remain alive until the autorelease pool ends.
    const auto image_start = std::chrono::steady_clock::now();
    const std::span<const float> distance_values =
        view_float_buffer(distances, num_ray, "distance output");
    write_colormapped_png(
        "distances.png",
        distance_values,
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    write_colormapped_png(
        "elevations.png",
        view_float_buffer(elevations, num_ray, "elevation output"),
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    const auto image_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> image_duration = image_end - image_start;
    std::printf("PNG generation: %.3f ms\n", image_duration.count());
  }
}

} // namespace panorama
