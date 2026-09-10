#pragma once

#include "ray_projection.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>

namespace panorama::app {

enum class MetalFxActivation : uint8_t { Disabled, PanMoveOnly, Always };
enum class MetalFxPreset : uint8_t { Off, Quality, Balanced, Performance };

struct MetalFxSelection {
  MetalFxActivation activation = MetalFxActivation::Disabled;
  MetalFxPreset preset = MetalFxPreset::Balanced;
  bool interacting = false;
  bool operator==(const MetalFxSelection &) const = default;
};

struct MetalFxResolution {
  ImageSize trace;
  bool enabled;
};

[[nodiscard]] MetalFxResolution
metalfx_resolution(ImageSize output, MetalFxSelection selection, bool supported);
[[nodiscard]] /// Scale output calibration independently on each axis after dimension rounding.
/// The GPU still owns per-pixel ray generation and the resulting LOD footprint.
RayFieldRequest metalfx_ray_request(const RayFieldRequest &output, ImageSize trace);
const char *metalfx_preset_name(MetalFxPreset preset);

/// Worker-owned spatial scaler and its two publish-safe private outputs.
class MetalFxUpscaler {
public:
  MetalFxUpscaler(id<MTLDevice> device, MTLPixelFormat format);
  ~MetalFxUpscaler();
  [[nodiscard]] bool supported() const;
  [[nodiscard]] bool configure(ImageSize input, ImageSize output);
  [[nodiscard]] MTLTextureUsage input_texture_usage() const;
  void begin_frame();
  void cancel_frame() noexcept;
  void encode(id<MTLCommandBuffer> command, id<MTLTexture> input);
  [[nodiscard]] id<MTLTexture> texture() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama::app
