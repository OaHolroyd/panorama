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
  const bool enabled =
      supported && selection.preset != MetalFxPreset::Off &&
      (selection.activation == MetalFxActivation::Always ||
       (selection.activation == MetalFxActivation::PanMoveOnly && selection.interacting));
  if (!enabled)
    return {output, output, false};
  const float scale = preset_scale(selection.preset);
  const auto dimension = [scale](uint32_t value) {
    return std::clamp<uint32_t>(uint32_t(std::lround(double(value) * scale)), 1U, value);
  };
  const ImageSize trace = {dimension(output.width), dimension(output.height)};
  return {trace, output, trace.width != output.width || trace.height != output.height};
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
  id<MTLFXSpatialScaler> scaler;
  id<MTLTexture> output, spare;
  ImageSize input = {}, dimensions = {};
  bool selected_spare = false;
};

MetalFxUpscaler::MetalFxUpscaler(id<MTLDevice> device, MTLPixelFormat format) {
  auto state = std::make_unique<State>();
  state->device = device;
  state->format = format;
  state_ = state.release();
}
MetalFxUpscaler::~MetalFxUpscaler() { delete state_; }

bool MetalFxUpscaler::supported() const {
  if (@available(macOS 13.0, *)) {
    return state_->device != nil && [MTLFXSpatialScalerDescriptor supportsDevice:state_->device];
  }
  return false;
}

bool MetalFxUpscaler::configure(ImageSize input, ImageSize output) {
  if (!supported() || (input.width == output.width && input.height == output.height))
    return false;
  if (state_->scaler != nil && state_->input.width == input.width &&
      state_->input.height == input.height && state_->dimensions.width == output.width &&
      state_->dimensions.height == output.height)
    return true;
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
    state_->dimensions = output;
    state_->selected_spare = false;
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
