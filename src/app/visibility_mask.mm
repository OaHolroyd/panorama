#include "visibility_mask.h"
#include "threadgroup_sizes.h"
#include "trace_diagnostics.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace panorama::app {

GpuVisibilityPointProjector::GpuVisibilityPointProjector(
    id<MTLDevice> device,
    id<MTLLibrary> library
)
    : device_(device) {
  if (device == nil || library == nil)
    throw std::invalid_argument("Visibility projection requires valid Metal resources");
  id<MTLFunction> function = [library newFunctionWithName:@"visibility_collision_points"];
  NSError *error = nil;
  pipeline_ =
      function == nil ? nil : [device newComputePipelineStateWithFunction:function error:&error];
  if (pipeline_ == nil)
    throw std::runtime_error("Could not create visibility collision-point pipeline");
}

id<MTLBuffer> GpuVisibilityPointProjector::project(
    id<MTLBuffer> rays,
    id<MTLBuffer> distances,
    ImageSize image,
    id<MTLCommandBuffer> command
) const {
  const uint64_t count64 = uint64_t(image.width) * image.height;
  if (count64 == 0 || count64 > UINT32_MAX || command == nil || rays == nil || distances == nil ||
      rays.length < count64 * sizeof(RayDirection) || distances.length < count64 * sizeof(float))
    throw std::invalid_argument("Visibility projection requires valid trace buffers");
  id<MTLBuffer> points = [device_ newBufferWithLength:count64 * 2 * sizeof(float)
                                              options:MTLResourceStorageModePrivate];
  if (points == nil)
    throw std::runtime_error("Could not allocate visibility collision-point buffer");
  points.label = @"Minimap visibility collision points";
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (encoder == nil)
    throw std::runtime_error("Could not create visibility collision-point encoder");
  encoder.label = @"visibility_collision_points";
  [encoder setComputePipelineState:pipeline_];
  [encoder setBuffer:rays offset:0 atIndex:0];
  [encoder setBuffer:distances offset:0 atIndex:1];
  [encoder setBuffer:points offset:0 atIndex:2];
  [encoder setBytes:&image length:sizeof(image) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(image.width, image.height, 1)
      threadsPerThreadgroup:threadgroups::spatial];
  [encoder endEncoding];
  if (diagnostics::enabled)
    ++diagnostics::minimap.refreshed;
  return points;
}

VisibilityMask::VisibilityMask(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library
)
    : device_(device), queue_(queue) {
  const auto pipeline = [&](NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    NSError *error = nil;
    id<MTLComputePipelineState> result =
        function == nil ? nil : [device newComputePipelineStateWithFunction:function error:&error];
    if (result == nil)
      throw std::runtime_error("Could not create minimap mask compute pipeline");
    return result;
  };
  scatter_ = pipeline(@"visibility_mask_scatter");
  resolve_ = pipeline(@"visibility_mask_resolve");
}

void VisibilityMask::clear() {
  occupancy_ = nil;
  pixels_ = nil;
  grid_ = nil;
  grid_region_ = {};
  projection_ = {};
  grid_width_ = 0;
  grid_height_ = 0;
}

CGImageRef VisibilityMask::render(
    id<MTLBuffer> points,
    VisibilityMaskParameters p,
    const VisibilityMapRegion *region
) {
  const auto started = std::chrono::steady_clock::now();
  diagnostics::Scope scope(diagnostics::minimap);
  if (!p.width || !p.height || p.width > 4096 || p.height > 4096 || !p.count || points == nil ||
      points.length < uint64_t(p.count) * 8)
    throw std::invalid_argument("Invalid visibility mask dimensions or points");
  const NSUInteger bytes = NSUInteger(p.width) * p.height * 4;
  if (pixels_ == nil || pixels_.length != bytes) {
    occupancy_ = [device_ newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    pixels_ = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (occupancy_ == nil || pixels_ == nil)
      throw std::runtime_error("Could not allocate minimap mask workspace");
  }
  if (region != nullptr) {
    diagnostics::minimap.mark("mask-projection");
    if (grid_ == nil || !(grid_region_ == *region) || grid_width_ != p.width ||
        grid_height_ != p.height) {
      projection_ = make_visibility_projection_grid(*region, p.width, p.height);
      grid_ = [device_ newBufferWithBytes:projection_.pixels.data()
                                   length:projection_.pixels.size() * sizeof(projection_.pixels[0])
                                  options:MTLResourceStorageModeShared];
      if (grid_ == nil)
        throw std::runtime_error("Could not allocate minimap projection grid");
      grid_region_ = *region;
      grid_width_ = p.width;
      grid_height_ = p.height;
    }
    p.grid_easting = float(projection_.easting);
    p.grid_northing = float(projection_.northing);
    p.grid_step_easting = float(projection_.step_easting);
    p.grid_step_northing = float(projection_.step_northing);
    p.grid_size = projection_.size;
  }
  id<MTLCommandBuffer> command = [queue_ commandBuffer];
  if (command == nil)
    throw std::runtime_error("Could not create minimap mask command");
  command.label = @"Minimap visibility mask";
  id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
  [clear fillBuffer:occupancy_ range:NSMakeRange(0, bytes) value:0];
  [clear endEncoding];
  id<MTLComputeCommandEncoder> scatter = [command computeCommandEncoder];
  [scatter setComputePipelineState:scatter_];
  [scatter setBuffer:points offset:0 atIndex:0];
  [scatter setBuffer:occupancy_ offset:0 atIndex:1];
  [scatter setBytes:&p length:sizeof(p) atIndex:2];
  [scatter setBuffer:grid_ == nil ? occupancy_ : grid_ offset:0 atIndex:3];
  [scatter dispatchThreads:MTLSizeMake(p.count, 1, 1)
      threadsPerThreadgroup:threadgroups::bounded_linear(scatter_.maxTotalThreadsPerThreadgroup)];
  [scatter endEncoding];
  id<MTLComputeCommandEncoder> resolve = [command computeCommandEncoder];
  [resolve setComputePipelineState:resolve_];
  [resolve setBuffer:occupancy_ offset:0 atIndex:0];
  [resolve setBuffer:pixels_ offset:0 atIndex:1];
  [resolve setBytes:&p length:sizeof(p) atIndex:2];
  [resolve dispatchThreads:MTLSizeMake(p.width, p.height, 1)
      threadsPerThreadgroup:threadgroups::spatial];
  [resolve endEncoding];
  diagnostics::minimap.mark("mask-gpu");
  diagnostics::track_submission(diagnostics::minimap, command, nil);
  [command commit];
  [command waitUntilCompleted]; // Only the mask worker waits; never AppKit.
  if (command.status == MTLCommandBufferStatusError)
    throw std::runtime_error("Minimap mask GPU command failed");
  diagnostics::minimap.mark("mask-image");
  CFDataRef data = CFDataCreate(
      kCFAllocatorDefault,
      static_cast<const UInt8 *>(pixels_.contents),
      CFIndex(bytes)
  );
  CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
  CGColorSpaceRef colour = CGColorSpaceCreateDeviceRGB();
  CGImageRef image = CGImageCreate(
      p.width,
      p.height,
      8,
      32,
      NSUInteger(p.width) * 4,
      colour,
      CGBitmapInfo(kCGImageAlphaPremultipliedLast) | kCGBitmapByteOrder32Big,
      provider,
      nullptr,
      false,
      kCGRenderingIntentDefault
  );
  CGColorSpaceRelease(colour);
  CGDataProviderRelease(provider);
  CFRelease(data);
  if (image == nullptr)
    throw std::runtime_error("Could not create minimap mask image");
  if (diagnostics::enabled) {
    const double wall =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    std::printf(
        "Minimap mask: %ux%u, %u rays, worker %.3f ms, GPU %.3f ms\n",
        p.width,
        p.height,
        p.count,
        wall,
        (command.GPUEndTime - command.GPUStartTime) * 1000.0
    );
  }
  return image;
}

} // namespace panorama::app
