#include "metalfx_upscaler.h"

#import <MetalFX/MetalFX.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace panorama::app {
namespace {
float preset_scale(MetalFxPreset preset) {
  switch (preset) {
  case MetalFxPreset::Off:
    return 1.0F;
  case MetalFxPreset::Quality:
    return 0.75F;
  case MetalFxPreset::Balanced:
    return 0.5F;
  case MetalFxPreset::Performance:
    return 0.33F;
  }
  return 1.0F;
}
} // namespace

MetalFxResolution metalfx_resolution(ImageSize output, MetalFxSelection selection, bool supported) {
  if (output.width == 0 || output.height == 0)
    throw std::invalid_argument("MetalFX output dimensions must be positive");
  const bool enabled =
      supported && selection.preset != MetalFxPreset::Off &&
      (selection.activation == MetalFxActivation::Always ||
       (selection.activation == MetalFxActivation::PanMoveOnly && selection.interacting));
  if (!enabled)
    return {output, false};
  const float scale = preset_scale(selection.preset);
  const auto dimension = [scale](uint32_t value) {
    return std::clamp<uint32_t>(uint32_t(std::lround(double(value) * scale)), 1U, value);
  };
  const ImageSize trace = {dimension(output.width), dimension(output.height)};
  return {trace, trace != output};
}

RayFieldRequest metalfx_ray_request(const RayFieldRequest &output, ImageSize trace) {
  if (output.image.width == 0 || output.image.height == 0 || trace.width == 0 || trace.height == 0)
    throw std::invalid_argument("MetalFX ray dimensions must be positive");
  RayFieldRequest request = output;
  request.image = trace;
  if (auto *camera = std::get_if<CameraProjection>(&request.projection)) {
    const double x = double(trace.width) / output.image.width;
    const double y = double(trace.height) / output.image.height;
    camera->intrinsics.focal_x *= x;
    camera->intrinsics.principal_x *= x;
    camera->intrinsics.focal_y *= y;
    camera->intrinsics.principal_y *= y;
  }
  return request;
}

const char *metalfx_preset_name(MetalFxPreset preset) {
  switch (preset) {
  case MetalFxPreset::Off:
    return "Off";
  case MetalFxPreset::Quality:
    return "Quality";
  case MetalFxPreset::Balanced:
    return "Balanced";
  case MetalFxPreset::Performance:
    return "Performance";
  }
  return "Off";
}

struct MetalFxUpscaler::State {
  id<MTLDevice> device;
  MTLPixelFormat format;
  bool supported = false;
  id<MTLFXSpatialScaler> scaler;
  id<MTLTexture> output, spare;
  ImageSize input = {}, output_dimensions = {};
  bool selected_spare = false;
  ImageSize failed_input = {}, failed_output = {};
};

MetalFxUpscaler::MetalFxUpscaler(id<MTLDevice> device, MTLPixelFormat format)
    : state_(std::make_unique<State>()) {
  state_->device = device;
  state_->format = format;
  if (@available(macOS 13.0, *)) {
    state_->supported = device != nil && [MTLFXSpatialScalerDescriptor supportsDevice:device];
  }
}
MetalFxUpscaler::~MetalFxUpscaler() = default;

bool MetalFxUpscaler::supported() const { return state_->supported; }

bool MetalFxUpscaler::configure(ImageSize input, ImageSize output) {
  if (input.width == 0 || input.height == 0 || output.width == 0 || output.height == 0 ||
      input.width > output.width || input.height > output.height)
    throw std::invalid_argument("MetalFX dimensions must describe a positive spatial upscale");
  if (!supported() || input == output)
    return false;
  if (state_->scaler != nil && state_->input == input && state_->output_dimensions == output)
    return true;
  if (state_->failed_input == input && state_->failed_output == output)
    return false;
  // Remember a failed configuration until another configuration succeeds.
  state_->failed_input = input;
  state_->failed_output = output;
  @autoreleasepool {
    MTLFXSpatialScalerDescriptor *descriptor = [[MTLFXSpatialScalerDescriptor alloc] init];
    descriptor.colorTextureFormat = state_->format;
    descriptor.outputTextureFormat = state_->format;
    descriptor.inputWidth = input.width;
    descriptor.inputHeight = input.height;
    descriptor.outputWidth = output.width;
    descriptor.outputHeight = output.height;
    // The terrain kernels encode sRGB values into BGRA8Unorm; Linear would be
    // incorrect without changing that existing colour pipeline.
    descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
    id<MTLFXSpatialScaler> scaler = [descriptor newSpatialScalerWithDevice:state_->device];
    if (scaler == nil)
      return false;
    MTLTextureDescriptor *textures =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:state_->format
                                                           width:output.width
                                                          height:output.height
                                                       mipmapped:NO];
    textures.storageMode = MTLStorageModePrivate;
    textures.usage = scaler.outputTextureUsage | MTLTextureUsageShaderRead;
    id<MTLTexture> first = [state_->device newTextureWithDescriptor:textures];
    id<MTLTexture> second = [state_->device newTextureWithDescriptor:textures];
    if (first == nil || second == nil)
      return false;
    first.label = @"MetalFX upscaled terrain";
    second.label = @"MetalFX upscaled terrain spare";
    state_->scaler = scaler;
    state_->output = first;
    state_->spare = second;
    state_->input = input;
    state_->output_dimensions = output;
    state_->selected_spare = false;
    state_->failed_input = {};
    state_->failed_output = {};
    return true;
  }
}

MTLTextureUsage MetalFxUpscaler::input_texture_usage() const {
  return state_->scaler == nil ? MTLTextureUsageUnknown : state_->scaler.colorTextureUsage;
}

void MetalFxUpscaler::begin_frame() { state_->selected_spare = !state_->selected_spare; }
void MetalFxUpscaler::cancel_frame() noexcept { state_->selected_spare = !state_->selected_spare; }

void MetalFxUpscaler::encode(id<MTLCommandBuffer> command, id<MTLTexture> input) {
  if (state_->scaler == nil || command == nil || input == nil)
    throw std::invalid_argument("MetalFX scaler is not configured");
  if ((input.usage & state_->scaler.colorTextureUsage) != state_->scaler.colorTextureUsage)
    throw std::runtime_error("Terrain texture lacks required MetalFX input usage");
  state_->scaler.colorTexture = input;
  state_->scaler.inputContentWidth = state_->input.width;
  state_->scaler.inputContentHeight = state_->input.height;
  state_->scaler.outputTexture = state_->selected_spare ? state_->spare : state_->output;
  [state_->scaler encodeToCommandBuffer:command];
}

id<MTLTexture> MetalFxUpscaler::texture() const {
  return state_->selected_spare ? state_->spare : state_->output;
}

} // namespace panorama::app
