#pragma once

#include "ray_projection.h"
#include "synthetic_render_options.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace panorama {

class Timer;

/// A CPU view of one completed, tightly packed RGB8 texture readback.
///
/// The bytes remain owned by `GpuImageRenderer` and are valid until its next
/// readback or destruction. The GPU performs RGBA-to-RGB packing so ImageIO
/// receives its faster 24-bit input without host-side conversion.
struct GpuImageReadback {
  std::span<const uint8_t> bytes;
  size_t bytes_per_row;
};

/// Presentation pipelines required by one caller's selected output products.
///
/// Pipeline creation has a noticeable fixed cost, so unused image styles are
/// omitted while each enabled pipeline remains reusable across frames.
struct GpuPresentationRequirements {
  bool scalar_diagnostics;
  bool normal_diagnostics;
  bool synthetic;
  /// Allocate scalar-range resources for distance/elevation terrain colour.
  bool synthetic_scalar_colour;
  /// Allocate the packing pipeline and shared buffer used by CLI image files.
  bool host_readback;
};

/// Reusable post-trace GPU presentation resources for one output image size.
///
/// Each render method converts scientific trace buffers into the same RGBA8
/// Metal texture. The CLI reads that texture back for PNG encoding; a future
/// interactive renderer can instead sample or blit `texture()` directly and
/// omit all host readback and ImageIO work.
class GpuImageRenderer {
public:
  GpuImageRenderer(
      id<MTLDevice> device,
      id<MTLCommandQueue> queue,
      id<MTLLibrary> library,
      ImageSize image,
      GpuPresentationRequirements requirements
  );

  GpuImageRenderer(const GpuImageRenderer &) = delete;
  GpuImageRenderer &operator=(const GpuImageRenderer &) = delete;
  ~GpuImageRenderer();

  /// Render a self-normalised scalar diagnostic using the viridis palette.
  void render_scalar(id<MTLBuffer> values, Timer &timer);

  /// Render packed east/north gradients as a conventional RGB normal map.
  void render_surface_normals(
      id<MTLBuffer> packed_gradients,
      id<MTLBuffer> distances,
      Timer &timer
  );

  /// Render lit terrain on black using white or colourmapped scalar values.
  void render_synthetic(
      id<MTLBuffer> packed_gradients,
      id<MTLBuffer> distances,
      id<MTLBuffer> colour_values,
      const SyntheticRenderOptions &options,
      Timer &timer
  );

  /// Return the reusable GPU render target populated by the last render call.
  [[nodiscard]] id<MTLTexture> texture() const;

  /// Pack the current texture into reusable shared RGB8 host storage.
  [[nodiscard]] GpuImageReadback readback(Timer &timer);

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
