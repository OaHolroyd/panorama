#include "crs.h"
#include "ray_projection.h"
#include "trace_diagnostics.h"
#include "visibility_mask.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

using panorama::app::VisibilityMask;
using panorama::app::VisibilityMaskParameters;
using MaskPoint = std::array<float, 2>;

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Independent CPU coverage oracle: test the distance of pixel centres to each
// projected hit, rather than reproducing the GPU's integer scatter algorithm.
static std::vector<uint8_t>
reference(const std::vector<MaskPoint> &points, VisibilityMaskParameters p) {
  std::vector<uint8_t> pixels(size_t(p.width) * p.height * 4, 0);
  for (uint32_t y = 0; y < p.height; ++y) {
    for (uint32_t x = 0; x < p.width; ++x) {
      for (const auto &point : points) {
        const double px = p.centre_x + point[0] * p.east_x + point[1] * p.north_x;
        const double py = p.centre_y + point[0] * p.east_y + point[1] * p.north_y;
        const double dx = x + 0.5 - px, dy = y + 0.5 - py;
        if (std::isfinite(px) && std::isfinite(py) && dx >= -1 && dx < 1 && dy >= -1 && dy < 1) {
          const size_t index = (size_t(y) * p.width + x) * 4;
          pixels[index + 1] = 42;
          pixels[index + 2] = pixels[index + 3] = 87;
          break;
        }
      }
    }
  }
  return pixels;
}

static void compare(CGImageRef image, const std::vector<uint8_t> &expected) {
  require(image != nullptr, "Missing mask image");
  CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(image));
  const bool equal = CFDataGetLength(data) == CFIndex(expected.size()) &&
                     std::memcmp(CFDataGetBytePtr(data), expected.data(), expected.size()) == 0;
  CFRelease(data);
  require(equal, "GPU coverage or premultiplied RGBA differs from CPU reference");
}

static void check_projection(id<MTLDevice> device, VisibilityMask &mask) {
  using panorama::app::VisibilityMapRegion;
  const auto mapPoint = [](panorama::LatLon point) {
    const double lat = point.lat * std::numbers::pi / 180;
    return std::array<double, 2>{
        (point.lon / 360 + 0.5) * 268435456,
        (0.5 - std::log(std::tan(std::numbers::pi / 4 + lat / 2)) / (2 * std::numbers::pi)) *
            268435456};
  };
  const std::array<std::pair<uint32_t, panorama::Coord>, 3> observers = {{
      {2056, {2623452.4, 1100502.2}},
      {2154, {700000, 6600000}},
      {27700, {530000, 180000}},
  }};
  for (const auto &[epsg, observer] : observers) {
    const auto crs = panorama::Crs::from_epsg(epsg);
    // Include distant map centres: the former observer-local affine basis
    // drifts most noticeably when inspecting terrain far from the observer.
    for (const double offset : {0.0, 200000.0}) {
      const panorama::Coord centre = {observer.x + offset, observer.y + offset};
      const auto geographic = crs.to_lat_lon(centre);
      const auto origin = mapPoint(geographic);
      for (const double span : {1000.0, 50000.0, 600000.0}) {
        const double mapSpan = span / std::cos(geographic.lat * std::numbers::pi / 180) *
                               268435456 / (2 * std::numbers::pi * 6378137);
        const VisibilityMapRegion region = {origin[0] - mapSpan / 2,
                                            origin[1] - mapSpan / 2,
                                            mapSpan,
                                            mapSpan,
                                            observer.x,
                                            observer.y,
                                            600000,
                                            epsg};
        const auto grid = make_visibility_projection_grid(region, 1040, 800);
        for (uint32_t row = 0; row + 1 < grid.size; ++row) {
          for (uint32_t col = 0; col + 1 < grid.size; ++col) {
            // Asymmetric interior samples differ from the midpoint samples
            // used by the adaptive-grid builder.
            constexpr double tx = .31, ty = .73;
            const panorama::Coord terrain = {
                observer.x + grid.easting + (col + tx) * grid.step_easting,
                observer.y + grid.northing + (row + ty) * grid.step_northing};
            const auto exact = mapPoint(crs.to_lat_lon(terrain));
            const uint32_t i = row * grid.size + col;
            for (size_t axis = 0; axis < 2; ++axis) {
              const double approximate = std::lerp(
                  std::lerp(grid.pixels[i][axis], grid.pixels[i + 1][axis], tx),
                  std::lerp(
                      grid.pixels[i + grid.size][axis],
                      grid.pixels[i + grid.size + 1][axis],
                      tx
                  ),
                  ty
              );
              const double pixel = axis == 0 ? (exact[0] - region.x) * 1040 / region.width
                                             : (exact[1] - region.y) * 800 / region.height;
              require(std::abs(approximate - pixel) < .5, "CRS grid exceeds half-pixel error");
            }
          }
        }
        // A centre hit checks GPU grid coordinates, north-up placement and
        // observer-relative addressing on all supported terrain CRSs.
        const MaskPoint point = {float(offset), float(offset)};
        id<MTLBuffer> buffer = [device newBufferWithBytes:point.data()
                                                   length:sizeof(point)
                                                  options:MTLResourceStorageModeShared];
        CGImageRef image = mask.render(buffer, {0, 0, 0, 0, 0, 0, 1040, 800, 1}, &region);
        compare(image, reference({{520, 400}}, {0, 0, 1, 0, 0, 1, 1040, 800, 1}));
        CGImageRelease(image);
      }
    }
  }
}

// Keep the former point rasterization in the test only, to catch changes in
// footprint, Y orientation or blending as the compute path evolves.
static void compare_legacy(id<MTLDevice> device, id<MTLCommandQueue> queue, VisibilityMask &mask) {
  NSString *source = @R"metal(
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex { float4 position [[position]]; float size [[point_size]]; };
    vertex Vertex point_vertex(device const float2 *points [[buffer(0)]], uint i [[vertex_id]]) {
      return {float4(points[i].x / 32.0f - 1.0f, 1.0f - points[i].y / 16.0f, 0, 1), 2};
    }
    fragment float4 point_fragment() { return float4(0, .48f * .34f, .34f, .34f); }
  )metal";
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
  require(library != nil, "Could not compile legacy reference shaders");
  MTLRenderPipelineDescriptor *descriptor = [[MTLRenderPipelineDescriptor alloc] init];
  descriptor.vertexFunction = [library newFunctionWithName:@"point_vertex"];
  descriptor.fragmentFunction = [library newFunctionWithName:@"point_fragment"];
  auto colour = descriptor.colorAttachments[0];
  colour.pixelFormat = MTLPixelFormatRGBA8Unorm;
  colour.blendingEnabled = YES;
  colour.rgbBlendOperation = colour.alphaBlendOperation = MTLBlendOperationMax;
  colour.sourceRGBBlendFactor = colour.destinationRGBBlendFactor = MTLBlendFactorOne;
  colour.sourceAlphaBlendFactor = colour.destinationAlphaBlendFactor = MTLBlendFactorOne;
  id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor
                                                                               error:&error];
  require(pipeline != nil, "Could not create legacy reference pipeline");
  std::vector<MaskPoint> points = {{4.25f, 7.75f}, {4.25f, 7.75f}, {55.75f, 20.25f}, {31, 12}};
  id<MTLBuffer> buffer = [device newBufferWithBytes:points.data()
                                             length:points.size() * sizeof(MaskPoint)
                                            options:MTLResourceStorageModeShared];
  MTLTextureDescriptor *textureDescriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:64
                                                        height:32
                                                     mipmapped:NO];
  textureDescriptor.usage = MTLTextureUsageRenderTarget;
  textureDescriptor.storageMode = MTLStorageModeShared;
  id<MTLTexture> target = [device newTextureWithDescriptor:textureDescriptor];
  MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
  [encoder setRenderPipelineState:pipeline];
  [encoder setVertexBuffer:buffer offset:0 atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:points.size()];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  require(command.status == MTLCommandBufferStatusCompleted, "Legacy reference rendering failed");
  std::vector<uint8_t> pixels(64 * 32 * 4);
  [target getBytes:pixels.data()
       bytesPerRow:64 * 4
        fromRegion:MTLRegionMake2D(0, 0, 64, 32)
       mipmapLevel:0];
  CGImageRef image = mask.render(buffer, {0, 0, 1, 0, 0, 1, 64, 32, uint32_t(points.size())});
  compare(image, pixels);
  CGImageRelease(image);
}

int main() {
  @autoreleasepool {
    try {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      require(device != nil, "No Metal device available");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      NSError *error = nil;
      id<MTLLibrary> library =
          [device newLibraryWithURL:[NSURL fileURLWithPath:@PANORAMA_METALLIB_PATH] error:&error];
      require(library != nil, "Could not load Metal library");
      VisibilityMask mask(device, queue, library);
      compare_legacy(device, queue, mask);
      check_projection(device, mask);
      const float infinity = std::numeric_limits<float>::infinity();
      const float nan = std::numeric_limits<float>::quiet_NaN();
      std::vector<MaskPoint> points = {
          {0, 0},
          {0, 0},
          {3.25f, 1.75f},
          {-1000, 1},
          {infinity, infinity},
          {nan, 0},
          {0, nan},
          {1e30f, 1e30f},
          {-0.5f, -0.5f},
          {31.5f, 18.5f},
      };
      std::mt19937 random(123);
      std::uniform_real_distribution<float> coordinate(-40, 40);
      for (size_t i = 0; i < 300; ++i)
        points.push_back({coordinate(random), coordinate(random)});
      id<MTLBuffer> buffer = [device newBufferWithBytes:points.data()
                                                 length:points.size() * sizeof(MaskPoint)
                                                options:MTLResourceStorageModeShared];
      VisibilityMaskParameters p = {0, 0, 1, 0, 0, 1, 32, 19, uint32_t(points.size())};
      CGImageRef original = mask.render(buffer, p);
      const auto originalPixels = reference(points, p);
      compare(original, originalPixels);
      // A rotated terrain basis, map-only pan/zoom, Retina-sized output and a
      // resize all consume the same immutable point snapshot without a trace.
      for (uint32_t scale : {1u, 2u}) {
        p = {13.25f,
             8.75f,
             0.8f,
             0.6f,
             -0.6f,
             0.8f,
             32 * scale,
             19 * scale,
             uint32_t(points.size())};
        CGImageRef next = mask.render(buffer, p);
        compare(next, reference(points, p));
        CGImageRelease(next);
        compare(original, originalPixels);
      }
      mask.clear();
      compare(original, originalPixels);
      CGImageRelease(original);

      // Exercise the actual trace-to-snapshot kernel, including horizontal
      // distance semantics: slope must not alter east/north collision position.
      std::array<panorama::RayDirection, 5> rays = {};
      for (auto &ray : rays) {
        ray.x = 0.6f;
        ray.y = 0.8f;
        ray.slope = 100;
      }
      const std::array<float, 5> distances = {10, 0, -1, infinity, nan};
      id<MTLBuffer> rayBuffer = [device newBufferWithBytes:rays.data()
                                                    length:sizeof(rays)
                                                   options:MTLResourceStorageModeShared];
      id<MTLBuffer> distanceBuffer = [device newBufferWithBytes:distances.data()
                                                         length:sizeof(distances)
                                                        options:MTLResourceStorageModeShared];
      id<MTLBuffer> snapshot = [device newBufferWithLength:5 * sizeof(MaskPoint)
                                                   options:MTLResourceStorageModeShared];
      panorama::app::GpuVisibilityPointProjector projector(device, library);
      id<MTLCommandBuffer> command = [queue commandBuffer];
      const uint32_t count = 5;
      id<MTLBuffer> privateSnapshot = projector.project(rayBuffer, distanceBuffer, {5, 1}, command);
      id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
      [copy copyFromBuffer:privateSnapshot
               sourceOffset:0
                   toBuffer:snapshot
          destinationOffset:0
                       size:snapshot.length];
      [copy endEncoding];
      [command commit];
      [command waitUntilCompleted];
      require(command.status == MTLCommandBufferStatusCompleted, "Collision snapshot failed");
      const auto *hits = static_cast<const MaskPoint *>(snapshot.contents);
      require(
          std::abs(hits[0][0] - 6) < 1e-5 && std::abs(hits[0][1] - 8) < 1e-5,
          "Collision snapshot used a 3D distance or slope"
      );
      for (size_t i = 1; i < count; ++i)
        require(
            !std::isfinite(hits[i][0]) && !std::isfinite(hits[i][1]),
            "Invalid distance produced a visible point"
        );
      // Full panorama density in one pixel is the worst contention/overdraw
      // case. Check constant coverage and expose its cost in the test output.
      std::vector<MaskPoint> dense(1600 * 900, {16, 9});
      id<MTLBuffer> denseBuffer = [device newBufferWithBytes:dense.data()
                                                      length:dense.size() * sizeof(MaskPoint)
                                                     options:MTLResourceStorageModeShared];
      panorama::app::diagnostics::enabled = true;
      CGImageRef denseImage =
          mask.render(denseBuffer, {0, 0, 1, 0, 0, 1, 1040, 800, uint32_t(dense.size())});
      panorama::app::diagnostics::enabled = false;
      compare(denseImage, reference({{16, 9}}, {0, 0, 1, 0, 0, 1, 1040, 800, 1}));
      CGImageRelease(denseImage);
      std::puts(
          "Minimap tests passed: legacy raster parity, coverage, duplicate opacity, invalid hits, "
          "CRS grids, resize, immutable images and horizontal-distance reconstruction."
      );
      return 0;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "Minimap test failed: %s\n", error.what());
      return 1;
    }
  }
}
