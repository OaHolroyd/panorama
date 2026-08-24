#include "gpu_image_renderer.h"

#include "timer.h"

#import <Foundation/Foundation.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace panorama {
namespace {

constexpr NSUInteger kReductionThreads = 256U;
constexpr NSUInteger kPresentationSide = 16U;

/// Print a Foundation error in the host application's diagnostic style.
void print_error(NSString *context, NSError *error) {
  const char *detail = error == nil ? "unknown error" : error.localizedDescription.UTF8String;
  std::fprintf(stderr, "%s: %s\n", context.UTF8String, detail);
}

/// Build one required presentation pipeline from the shared application library.
[[nodiscard]] id<MTLComputePipelineState>
make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil) {
    throw std::runtime_error("Missing Metal presentation kernel " + std::string(name.UTF8String));
  }
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (pipeline == nil) {
    print_error(@"Could not create GPU presentation pipeline", error);
    throw std::runtime_error("Could not create GPU presentation pipeline");
  }
  return pipeline;
}

/// Throw after a synchronously awaited presentation or readback command fails.
void check_command(id<MTLCommandBuffer> command, NSString *context) {
  if (command.status != MTLCommandBufferStatusCompleted) {
    print_error(context, command.error);
    throw std::runtime_error(std::string(context.UTF8String));
  }
}

/// Complete one timed GPU operation and record both wall and device work.
void complete_timed_command(
    id<MTLCommandBuffer> command,
    Timer &timer,
    std::string_view measurement,
    NSString *failure_context
) {
  [command commit];
  [command waitUntilCompleted];
  check_command(command, failure_context);
  timer.add_work(measurement, 1'000.0 * (command.GPUEndTime - command.GPUStartTime));
  timer.stop(measurement);
}

/// Dispatch one thread per output pixel in square presentation groups.
void dispatch_image(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    ImageSize image
) {
  [encoder setComputePipelineState:pipeline];
  [encoder dispatchThreads:MTLSizeMake(image.width, image.height, 1U)
      threadsPerThreadgroup:MTLSizeMake(kPresentationSide, kPresentationSide, 1U)];
}

/// Encode the shared scalar min/max reduction, optionally excluding ray misses.
void encode_range_reduction(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> values,
    id<MTLBuffer> distances,
    id<MTLBuffer> range,
    uint32_t pixel_count,
    bool collisions_only
) {
  encoder.label = @"reduce_finite_range";
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:values offset:0 atIndex:0];
  [encoder setBuffer:distances offset:0 atIndex:1];
  [encoder setBuffer:range offset:0 atIndex:2];
  [encoder setBytes:&pixel_count length:sizeof(pixel_count) atIndex:3];
  const uint32_t collision_flag = collisions_only ? 1U : 0U;
  [encoder setBytes:&collision_flag length:sizeof(collision_flag) atIndex:4];
  const NSUInteger group_count =
      (static_cast<NSUInteger>(pixel_count) + kReductionThreads - 1U) / kReductionThreads;
  [encoder dispatchThreadgroups:MTLSizeMake(group_count, 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(kReductionThreads, 1U, 1U)];
}

} // namespace

struct GpuImageRenderer::State {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> reduce_range;
  id<MTLComputePipelineState> scalar;
  id<MTLComputePipelineState> normals;
  id<MTLComputePipelineState> synthetic;
  id<MTLComputePipelineState> pack_rgb;
  id<MTLTexture> output;
  id<MTLBuffer> finite_range;
  id<MTLBuffer> readback;
  ImageSize image;
  uint32_t pixel_count;
  NSUInteger bytes_per_row;
  bool synthetic_scalar_colour;
};

GpuImageRenderer::GpuImageRenderer(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library,
    ImageSize image,
    GpuPresentationRequirements requirements
) {
  const uint64_t pixel_count = static_cast<uint64_t>(image.width) * image.height;
  if (device == nil || queue == nil || library == nil || pixel_count == 0U ||
      pixel_count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("GPU image renderer requires valid Metal state and dimensions");
  }

  auto state = std::make_unique<State>();
  state->queue = queue;
  state->image = image;
  state->pixel_count = static_cast<uint32_t>(pixel_count);
  state->synthetic_scalar_colour = requirements.synthetic_scalar_colour;
  if (requirements.scalar_diagnostics || requirements.synthetic_scalar_colour) {
    state->reduce_range = make_pipeline(device, library, @"reduce_finite_range");
  }
  if (requirements.scalar_diagnostics) {
    state->scalar = make_pipeline(device, library, @"present_scalar_viridis");
  }
  if (requirements.normal_diagnostics) {
    state->normals = make_pipeline(device, library, @"present_surface_normals");
  }
  if (requirements.synthetic) {
    NSString *name = requirements.synthetic_scalar_colour
                         ? @"present_colourmapped_synthetic_terrain"
                         : @"present_synthetic_terrain";
    state->synthetic = make_pipeline(device, library, name);
  }
  if (requirements.host_readback) {
    state->pack_rgb = make_pipeline(device, library, @"pack_presented_rgb");
  }

  MTLTextureDescriptor *texture_descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:image.width
                                                        height:image.height
                                                     mipmapped:NO];
  texture_descriptor.storageMode = MTLStorageModePrivate;
  texture_descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  state->output = [device newTextureWithDescriptor:texture_descriptor];
  if (state->output == nil) {
    throw std::runtime_error("Could not allocate GPU presentation texture");
  }
  state->output.label = @"Presented image";

  if (requirements.scalar_diagnostics || requirements.synthetic_scalar_colour) {
    state->finite_range = [device newBufferWithLength:2U * sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
    if (state->finite_range == nil) {
      throw std::runtime_error("Could not allocate presentation finite-range buffer");
    }
    state->finite_range.label = @"Presentation finite range";
  }
  if (requirements.host_readback) {
    state->bytes_per_row = static_cast<NSUInteger>(image.width) * 3U;
    if (image.height > std::numeric_limits<NSUInteger>::max() / state->bytes_per_row) {
      throw std::overflow_error("GPU image readback is too large");
    }
    state->readback = [device newBufferWithLength:state->bytes_per_row * image.height
                                          options:MTLResourceStorageModeShared];
    if (state->readback == nil) {
      throw std::runtime_error("Could not allocate presented-image readback buffer");
    }
    state->readback.label = @"Presented image readback";
  }
  state_ = std::move(state);
}

GpuImageRenderer::~GpuImageRenderer() = default;

void GpuImageRenderer::render_scalar(id<MTLBuffer> values, Timer &timer) {
  State &state = *state_;
  if (state.scalar == nil || state.reduce_range == nil) {
    throw std::logic_error("Scalar diagnostic presentation was not enabled");
  }
  auto *range = static_cast<uint32_t *>(state.finite_range.contents);
  if (values == nil || range == nullptr) {
    throw std::invalid_argument("Scalar presentation requires a valid value buffer");
  }
  range[0] = std::numeric_limits<uint32_t>::max();
  range[1] = 0U;

  timer.start_wall("Pixel conversion");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create scalar presentation command");
  }
  command.label = @"Present scalar diagnostic";
  encode_range_reduction(
      encoder,
      state.reduce_range,
      values,
      values,
      state.finite_range,
      state.pixel_count,
      false
  );
  [encoder endEncoding];

  encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Could not create scalar colourmap encoder");
  }
  encoder.label = @"present_scalar_viridis";
  [encoder setBuffer:values offset:0 atIndex:0];
  [encoder setBuffer:state.finite_range offset:0 atIndex:1];
  [encoder setTexture:state.output atIndex:0];
  dispatch_image(encoder, state.scalar, state.image);
  [encoder endEncoding];
  complete_timed_command(command, timer, "Pixel conversion", @"GPU scalar presentation failed");
}

void GpuImageRenderer::render_surface_normals(
    id<MTLBuffer> packed_gradients,
    id<MTLBuffer> distances,
    Timer &timer
) {
  State &state = *state_;
  if (state.normals == nil) {
    throw std::logic_error("Normal diagnostic presentation was not enabled");
  }
  if (packed_gradients == nil || distances == nil) {
    throw std::invalid_argument("Normal presentation requires valid trace buffers");
  }
  timer.start_wall("Pixel conversion");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create normal presentation command");
  }
  command.label = @"Present surface normals";
  encoder.label = @"present_surface_normals";
  [encoder setBuffer:packed_gradients offset:0 atIndex:0];
  [encoder setBuffer:distances offset:0 atIndex:1];
  [encoder setTexture:state.output atIndex:0];
  dispatch_image(encoder, state.normals, state.image);
  [encoder endEncoding];
  complete_timed_command(command, timer, "Pixel conversion", @"GPU normal presentation failed");
}

void GpuImageRenderer::render_synthetic(
    id<MTLBuffer> packed_gradients,
    id<MTLBuffer> distances,
    id<MTLBuffer> colour_values,
    const SyntheticRenderOptions &options,
    Timer &timer
) {
  State &state = *state_;
  if (state.synthetic == nil) {
    throw std::logic_error("Synthetic presentation was not enabled");
  }
  const uint32_t colour_source = static_cast<uint32_t>(options.colour_source);
  const uint32_t colourmap = static_cast<uint32_t>(options.colourmap);
  if (packed_gradients == nil || distances == nil || colour_values == nil ||
      !std::isfinite(options.sun_azimuth) || !std::isfinite(options.sun_elevation) ||
      !std::isfinite(options.ambient_light) || options.ambient_light < 0.0F ||
      options.ambient_light > 1.0F ||
      colour_source > static_cast<uint32_t>(TerrainColourSource::Elevation) ||
      colourmap > static_cast<uint32_t>(PresetColourmap::Turbo)) {
    throw std::invalid_argument("Synthetic presentation inputs are invalid");
  }
  if (options.colour_source != TerrainColourSource::White &&
      (state.reduce_range == nil || state.finite_range == nil)) {
    throw std::logic_error("Synthetic scalar colouring was not enabled");
  }
  const bool scalar_colour = options.colour_source != TerrainColourSource::White;
  if (scalar_colour != state.synthetic_scalar_colour) {
    throw std::logic_error("Synthetic presentation pipeline does not match its colour source");
  }
  const float azimuth = static_cast<float>(options.sun_azimuth);
  const float elevation = static_cast<float>(options.sun_elevation);
  const float horizontal = std::cos(elevation);
  const float sun_and_ambient[4] = {
      std::sin(azimuth) * horizontal,
      std::cos(azimuth) * horizontal,
      std::sin(elevation),
      options.ambient_light,
  };

  if (scalar_colour) {
    auto *range = static_cast<uint32_t *>(state.finite_range.contents);
    if (range == nullptr) {
      throw std::runtime_error("Could not map synthetic finite-range buffer");
    }
    range[0] = std::numeric_limits<uint32_t>::max();
    range[1] = 0U;
  }

  timer.start_wall("Pixel conversion");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create synthetic presentation command");
  }
  command.label = @"Present synthetic terrain";
  if (scalar_colour) {
    encode_range_reduction(
        encoder,
        state.reduce_range,
        colour_values,
        distances,
        state.finite_range,
        state.pixel_count,
        true
    );
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    if (encoder == nil) {
      throw std::runtime_error("Could not create synthetic colourmap encoder");
    }
  }
  encoder.label =
      scalar_colour ? @"present_colourmapped_synthetic_terrain" : @"present_synthetic_terrain";
  [encoder setBuffer:packed_gradients offset:0 atIndex:0];
  [encoder setBuffer:distances offset:0 atIndex:1];
  [encoder setBytes:sun_and_ambient length:sizeof(sun_and_ambient) atIndex:2];
  if (scalar_colour) {
    [encoder setBuffer:colour_values offset:0 atIndex:3];
    [encoder setBuffer:state.finite_range offset:0 atIndex:4];
    [encoder setBytes:&colourmap length:sizeof(colourmap) atIndex:5];
  }
  [encoder setTexture:state.output atIndex:0];
  dispatch_image(encoder, state.synthetic, state.image);
  [encoder endEncoding];
  complete_timed_command(command, timer, "Pixel conversion", @"GPU synthetic presentation failed");
}

id<MTLTexture> GpuImageRenderer::texture() const { return state_->output; }

GpuImageReadback GpuImageRenderer::readback(Timer &timer) {
  State &state = *state_;
  if (state.pack_rgb == nil || state.readback == nil) {
    throw std::logic_error("Host image readback was not enabled");
  }
  timer.start_wall("Pixel readback");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create image readback command");
  }
  command.label = @"Read back presented image";
  encoder.label = @"pack_presented_rgb";
  [encoder setTexture:state.output atIndex:0];
  [encoder setBuffer:state.readback offset:0 atIndex:0];
  dispatch_image(encoder, state.pack_rgb, state.image);
  [encoder endEncoding];
  complete_timed_command(command, timer, "Pixel readback", @"GPU image readback failed");

  const auto *bytes = static_cast<const uint8_t *>(state.readback.contents);
  if (bytes == nullptr) {
    throw std::runtime_error("Could not map presented image readback");
  }
  return {{bytes, state.readback.length}, static_cast<size_t>(state.bytes_per_row)};
}

} // namespace panorama
