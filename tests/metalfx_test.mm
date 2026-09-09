#include "inspection_coordinates.h"
#include "metalfx_upscaler.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>

using panorama::ImageSize;
using namespace panorama::app;

static void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

static void check_inspection_coordinates() {
  // The same bottom-right cursor location must survive queued requests while
  // switching presets, settling to native resolution, or resizing the window.
  const InspectionLocation cursor = {0.875, 0.875};
  struct Sample {
    ImageSize image;
    InspectionPixel expected;
  };
  for (const auto &sample : std::array<Sample, 7>{{
           {{1600, 900}, {1400, 787}},
           {{1200, 675}, {1050, 590}},
           {{800, 450}, {700, 393}},
           {{528, 297}, {462, 259}},
           {{1600, 900}, {1400, 787}},
           {{3200, 1800}, {2800, 1575}},
           {{1, 1}, {0, 0}},
       }}) {
    const auto pixel = inspection_pixel(cursor, sample.image);
    require(
        pixel && pixel->x == sample.expected.x && pixel->y == sample.expected.y,
        "Inspection must track ray resolution, independently of drawable pixels"
    );
    const auto edge = inspection_pixel({1.0, 1.0}, sample.image);
    require(
        edge && edge->x == sample.image.width - 1 && edge->y == sample.image.height - 1,
        "Bottom-right boundary must remain inside the ray buffer"
    );
    const auto origin = inspection_pixel({0.0, 0.0}, sample.image);
    require(origin && origin->x == 0 && origin->y == 0, "Inspection origin must remain top-left");
  }
  require(
      !inspection_pixel(cursor, {0, 900}) && !inspection_pixel(cursor, {1600, 0}),
      "Empty ray images must clear inspection"
  );
  for (const auto location : std::array<InspectionLocation, 6>{{
           {-0.1, 0.5},
           {1.1, 0.5},
           {0.5, -0.1},
           {0.5, 1.1},
           {std::numeric_limits<double>::quiet_NaN(), 0.5},
           {0.5, std::numeric_limits<double>::infinity()},
       }})
    require(
        !inspection_pixel(location, {800, 450}),
        "Invalid cursor locations must clear inspection without throwing"
    );
  std::puts("MetalFX inspection coordinate regression tests passed.");
}

int main() {
  try {
    check_inspection_coordinates();
    const ImageSize output = {1600, 900};
    const MetalFxSelection disabled = {MetalFxActivation::Disabled, MetalFxPreset::Balanced, true};
    require(!metalfx_resolution(output, disabled, true).enabled, "Disabled must remain native");
    const MetalFxSelection off = {MetalFxActivation::Always, MetalFxPreset::Off, true};
    require(!metalfx_resolution(output, off, true).enabled, "Off must remain native");
    const MetalFxSelection quality = {MetalFxActivation::Always, MetalFxPreset::Quality, false};
    const MetalFxSelection balanced = {MetalFxActivation::Always, MetalFxPreset::Balanced, false};
    const MetalFxSelection performance = {MetalFxActivation::Always,
                                          MetalFxPreset::Performance,
                                          false};
    require(
        metalfx_resolution(output, quality, true).trace.width == 1200 &&
            metalfx_resolution(output, quality, true).trace.height == 675,
        "Quality dimensions incorrect"
    );
    require(
        metalfx_resolution(output, balanced, true).trace.width == 800 &&
            metalfx_resolution(output, balanced, true).trace.height == 450,
        "Balanced dimensions incorrect"
    );
    require(
        metalfx_resolution(output, performance, true).trace.width == 528 &&
            metalfx_resolution(output, performance, true).trace.height == 297,
        "Performance dimensions incorrect"
    );
    const MetalFxSelection moving = {MetalFxActivation::PanMoveOnly, MetalFxPreset::Balanced, true};
    const MetalFxSelection settled = {MetalFxActivation::PanMoveOnly,
                                      MetalFxPreset::Balanced,
                                      false};
    require(
        metalfx_resolution(output, moving, true).enabled,
        "Movement should enable Pan/move only"
    );
    require(
        !metalfx_resolution(output, settled, true).enabled,
        "Settled Pan/move only must return native"
    );
    require(
        !metalfx_resolution(output, moving, false).enabled,
        "Unsupported GPU must remain native"
    );
    const auto odd = metalfx_resolution({1, 2}, performance, true);
    require(
        odd.trace.width == 1 && odd.trace.height == 1,
        "Tiny dimensions must clamp to one pixel"
    );
    @autoreleasepool {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      require(device != nil, "No Metal device");
      MetalFxUpscaler scaler(device, MTLPixelFormatBGRA8Unorm);
      if (scaler.supported()) {
        require(scaler.configure({4, 3}, {8, 6}), "Could not configure supported MetalFX scaler");
        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:4
                                                              height:3
                                                           mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage =
            MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | scaler.input_texture_usage();
        id<MTLTexture> input = [device newTextureWithDescriptor:descriptor];
        require(input != nil, "Could not create MetalFX input texture");
        const std::array<uint8_t, 4 * 3 * 4> pixels = [] {
          std::array<uint8_t, 4 * 3 * 4> result{};
          for (size_t i = 0; i < result.size(); i += 4) {
            result[i] = 40;
            result[i + 1] = 90;
            result[i + 2] = 180;
            result[i + 3] = 255;
          }
          return result;
        }();
        [input replaceRegion:MTLRegionMake2D(0, 0, 4, 3)
                 mipmapLevel:0
                   withBytes:pixels.data()
                 bytesPerRow:4 * 4];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        scaler.begin_frame();
        scaler.encode(command, input);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "MetalFX command failed");
        require(
            scaler.texture().width == 8 && scaler.texture().height == 6,
            "MetalFX output dimensions incorrect"
        );
        id<MTLTexture> published = scaler.texture();
        const auto read_output = [&](id<MTLTexture> texture) {
          constexpr NSUInteger row_bytes = 256;
          id<MTLBuffer> readback = [device newBufferWithLength:row_bytes * 6
                                                       options:MTLResourceStorageModeShared];
          id<MTLCommandBuffer> copy = [queue commandBuffer];
          id<MTLBlitCommandEncoder> blit = [copy blitCommandEncoder];
          [blit copyFromTexture:texture
                           sourceSlice:0
                           sourceLevel:0
                          sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(8, 6, 1)
                              toBuffer:readback
                     destinationOffset:0
                destinationBytesPerRow:row_bytes
              destinationBytesPerImage:row_bytes * 6];
          [blit endEncoding];
          [copy commit];
          [copy waitUntilCompleted];
          require(copy.status == MTLCommandBufferStatusCompleted, "Output readback failed");
          std::array<uint8_t, 8 * 6 * 4> result;
          const auto *bytes = static_cast<const uint8_t *>(readback.contents);
          for (size_t row = 0; row < 6; ++row)
            std::copy_n(bytes + row * row_bytes, 8 * 4, result.data() + row * 8 * 4);
          return result;
        };
        const auto published_pixels = read_output(published);
        std::array<uint8_t, 4 * 3 * 4> changed_pixels = pixels;
        for (size_t i = 0; i < changed_pixels.size(); i += 4)
          changed_pixels[i] = 220;
        [input replaceRegion:MTLRegionMake2D(0, 0, 4, 3)
                 mipmapLevel:0
                   withBytes:changed_pixels.data()
                 bytesPerRow:4 * 4];
        // A producer can run both its initial pass and a streaming repair.
        // Both must target the same unpublished output, preserving the visible one.
        scaler.begin_frame();
        id<MTLTexture> pending = scaler.texture();
        require(pending != published, "New frame must use an unpublished texture");
        for (int pass = 0; pass < 2; ++pass) {
          command = [queue commandBuffer];
          scaler.encode(command, input);
          [command commit];
          [command waitUntilCompleted];
          require(command.status == MTLCommandBufferStatusCompleted, "Repair upscale failed");
          require(scaler.texture() == pending, "Repair must keep the selected frame texture");
        }
        require(
            read_output(published) == published_pixels,
            "Streaming repair must preserve the published image"
        );
        require(read_output(pending) != published_pixels, "Pending image must contain new pixels");
        scaler.cancel_frame();
        require(scaler.texture() == published, "Cancelled producer must restore published output");
        std::puts("MetalFX GPU scaler test passed.");
      } else {
        std::puts("MetalFX GPU scaler test skipped: unsupported device.");
      }
    }
    std::puts("MetalFX resolution policy tests passed.");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "MetalFX test failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
