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

} // namespace

struct GpuImageRenderer::State {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> scalar;
  id<MTLComputePipelineState> normals;
  id<MTLComputePipelineState> white_synthetic;
  id<MTLComputePipelineState> colourmapped_synthetic;
  id<MTLComputePipelineState> pack_rgb;
  id<MTLTexture> output;
  id<MTLBuffer> readback;
  ImageSize image;
  NSUInteger bytes_per_row;
  MTLPixelFormat output_pixel_format;
  bool host_readback;

  void resize(ImageSize next_image) {
    const uint64_t pixel_count = static_cast<uint64_t>(next_image.width) * next_image.height;
    if (pixel_count == 0U || pixel_count > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("GPU image dimensions are invalid");
    }
    if (output != nil && image.width == next_image.width && image.height == next_image.height) {
      return;
    }

    MTLTextureDescriptor *texture_descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:output_pixel_format
                                                           width:next_image.width
                                                          height:next_image.height
                                                       mipmapped:NO];
    texture_descriptor.storageMode = MTLStorageModePrivate;
    texture_descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> next_output = [device newTextureWithDescriptor:texture_descriptor];
    if (next_output == nil) {
      throw std::runtime_error("Could not allocate GPU presentation texture");
    }
    next_output.label = @"Presented image";

    id<MTLBuffer> next_readback = nil;
    NSUInteger next_bytes_per_row = 0U;
    if (host_readback) {
      next_bytes_per_row = static_cast<NSUInteger>(next_image.width) * 3U;
      if (next_image.height > std::numeric_limits<NSUInteger>::max() / next_bytes_per_row) {
        throw std::overflow_error("GPU image readback is too large");
      }
      next_readback = [device newBufferWithLength:next_bytes_per_row * next_image.height
                                          options:MTLResourceStorageModeShared];
      if (next_readback == nil) {
        throw std::runtime_error("Could not allocate presented-image readback buffer");
      }
      next_readback.label = @"Presented image readback";
    }

    image = next_image;
    output = next_output;
    readback = next_readback;
    bytes_per_row = next_bytes_per_row;
  }
};

GpuImageRenderer::GpuImageRenderer(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library,
    ImageSize image,
    GpuPresentationRequirements requirements,
    MTLPixelFormat output_pixel_format
) {
  const uint64_t pixel_count = static_cast<uint64_t>(image.width) * image.height;
  if (device == nil || queue == nil || library == nil || pixel_count == 0U ||
      pixel_count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("GPU image renderer requires valid Metal state and dimensions");
  }

  auto state = std::make_unique<State>();
  state->device = device;
  state->queue = queue;
  state->output_pixel_format = output_pixel_format;
  state->host_readback = requirements.host_readback;
  if (requirements.scalar_diagnostics) {
    state->scalar = make_pipeline(device, library, @"present_scalar_viridis");
  }
  if (requirements.normal_diagnostics) {
    state->normals = make_pipeline(device, library, @"present_surface_normals");
  }
  if (requirements.white_synthetic) {
    state->white_synthetic = make_pipeline(device, library, @"present_synthetic_terrain");
  }
  if (requirements.synthetic_scalar_colour) {
    state->colourmapped_synthetic =
        make_pipeline(device, library, @"present_colourmapped_synthetic_terrain");
  }
  if (requirements.host_readback) {
    state->pack_rgb = make_pipeline(device, library, @"pack_presented_rgb");
  }

  if (output_pixel_format != MTLPixelFormatRGBA8Unorm &&
      output_pixel_format != MTLPixelFormatBGRA8Unorm) {
    throw std::invalid_argument("GPU image renderer requires an RGBA8 or BGRA8 target");
  }
  state->resize(image);
  state_ = std::move(state);
}

GpuImageRenderer::~GpuImageRenderer() = default;

void GpuImageRenderer::resize(ImageSize image) { state_->resize(image); }

void GpuImageRenderer::render_scalar(id<MTLBuffer> values, ScalarColourRange range, Timer &timer) {
  State &state = *state_;
  if (state.scalar == nil) {
    throw std::logic_error("Scalar diagnostic presentation was not enabled");
  }
  if (values == nil || !std::isfinite(range.minimum) || !std::isfinite(range.maximum) ||
      range.maximum <= range.minimum) {
    throw std::invalid_argument("Scalar presentation requires a valid value buffer");
  }

  timer.start_wall("Pixel conversion");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create scalar presentation command");
  }
  command.label = @"Present scalar diagnostic";
  encoder.label = @"present_scalar_viridis";
  [encoder setBuffer:values offset:0 atIndex:0];
  [encoder setBytes:&range length:sizeof(range) atIndex:1];
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
    ScalarColourRange range,
    bool use_surface_normals,
    Timer &timer
) {
  State &state = *state_;
  const uint32_t colour_source = static_cast<uint32_t>(options.colour_source);
  const uint32_t colourmap = static_cast<uint32_t>(options.colourmap);
  const bool scalar_colour = options.colour_source != TerrainColourSource::White;
  if ((use_surface_normals && packed_gradients == nil) || distances == nil ||
      (scalar_colour && colour_values == nil) || !std::isfinite(options.sun_azimuth) ||
      !std::isfinite(options.sun_elevation) || !std::isfinite(options.ambient_light) ||
      options.ambient_light < 0.0F || options.ambient_light > 1.0F ||
      !std::isfinite(options.diffusivity) || options.diffusivity < 0.0F ||
      options.diffusivity > 1.0F || !std::isfinite(range.minimum) ||
      !std::isfinite(range.maximum) || range.maximum <= range.minimum ||
      colour_source > static_cast<uint32_t>(TerrainColourSource::Elevation) ||
      colourmap > static_cast<uint32_t>(PresetColourmap::Turbo)) {
    throw std::invalid_argument("Synthetic presentation inputs are invalid");
  }
  id<MTLComputePipelineState> pipeline =
      scalar_colour ? state.colourmapped_synthetic : state.white_synthetic;
  if (pipeline == nil) {
    throw std::logic_error("Selected synthetic presentation pipeline was not enabled");
  }
  const uint32_t normal_lighting = use_surface_normals ? 1U : 0U;
  const float azimuth = static_cast<float>(options.sun_azimuth);
  const float elevation = static_cast<float>(options.sun_elevation);
  const float horizontal = std::cos(elevation);
  const float sun_and_ambient[4] = {
      std::sin(azimuth) * horizontal,
      std::cos(azimuth) * horizontal,
      std::sin(elevation),
      options.ambient_light,
  };

  timer.start_wall("Pixel conversion");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create synthetic presentation command");
  }
  command.label = @"Present synthetic terrain";
  encoder.label =
      scalar_colour ? @"present_colourmapped_synthetic_terrain" : @"present_synthetic_terrain";
  [encoder setBuffer:packed_gradients offset:0 atIndex:0];
  [encoder setBuffer:distances offset:0 atIndex:1];
  [encoder setBytes:sun_and_ambient length:sizeof(sun_and_ambient) atIndex:2];
  if (scalar_colour) {
    [encoder setBuffer:colour_values offset:0 atIndex:3];
    [encoder setBytes:&range length:sizeof(range) atIndex:4];
    [encoder setBytes:&colourmap length:sizeof(colourmap) atIndex:5];
    [encoder setBytes:&normal_lighting length:sizeof(normal_lighting) atIndex:6];
    [encoder setBytes:&options.diffusivity length:sizeof(options.diffusivity) atIndex:7];
  } else {
    [encoder setBytes:&normal_lighting length:sizeof(normal_lighting) atIndex:3];
    [encoder setBytes:&options.diffusivity length:sizeof(options.diffusivity) atIndex:4];
  }
  [encoder setTexture:state.output atIndex:0];
  dispatch_image(encoder, pipeline, state.image);
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
