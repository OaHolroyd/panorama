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
  /// Enable uncoloured (white base) synthetic terrain presentation.
  bool white_synthetic;
  /// Enable distance/elevation colourmapped synthetic presentation.
  bool synthetic_scalar_colour;
  /// Allocate the packing pipeline and shared buffer used by CLI image files.
  bool host_readback;
};

/// Reusable post-trace GPU presentation resources for one output image size.
///
/// Each render method converts scientific trace buffers into the same 8-bit
/// four-channel Metal texture. The CLI reads that texture back for PNG
/// encoding, while the interactive viewer blits `texture()` directly and
/// omits host readback and ImageIO work.
class GpuImageRenderer {
public:
  GpuImageRenderer(
      id<MTLDevice> device,
      id<MTLCommandQueue> queue,
      id<MTLLibrary> library,
      ImageSize image,
      GpuPresentationRequirements requirements,
      MTLPixelFormat output_pixel_format = MTLPixelFormatRGBA8Unorm
  );

  GpuImageRenderer(const GpuImageRenderer &) = delete;
  GpuImageRenderer &operator=(const GpuImageRenderer &) = delete;
  ~GpuImageRenderer();

  /// Reallocate only image-sized targets while retaining compiled pipelines.
  void resize(ImageSize image);

  /// Render a scalar diagnostic using viridis over a fixed value range.
  void render_scalar(id<MTLBuffer> values, ScalarColourRange range, Timer &timer);

  /// Render packed east/north gradients as a conventional RGB normal map.
  void
  render_surface_normals(id<MTLBuffer> packed_gradients, id<MTLBuffer> distances, Timer &timer);

  /// Render terrain on black using white or colourmapped scalar values.
  ///
  /// `packed_gradients` may be nil when normal lighting is disabled. Callers
  /// which switch lighting at runtime should retain it so enabling lighting
  /// remains a presentation-only operation. `ray_directions` is required only
  /// when feature outlines are enabled.
  void render_synthetic(
      id<MTLBuffer> packed_gradients,
      id<MTLBuffer> distances,
      id<MTLBuffer> ray_directions,
      id<MTLBuffer> colour_values,
      id<MTLBuffer> shadow_visibility,
      const SyntheticRenderOptions &options,
      ScalarColourRange range,
      bool use_surface_normals,
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
