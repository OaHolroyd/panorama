#include "metal_bvh_resources.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace panorama::bvh_resources {
std::string error_text(NSError *error) {
  return error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String;
}
uint32_t checked_count(uint64_t count) {
  if (count > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("Metal BVH count exceeds uint32");
  return static_cast<uint32_t>(count);
}
id<MTLBuffer> buffer(
    id<MTLDevice> device,
    uint64_t count,
    size_t stride,
    NSString *label,
    MTLResourceOptions options
) {
  if (stride == 0 || count > device.maxBufferLength / stride)
    throw std::overflow_error(
        "Metal BVH " + std::string(label.UTF8String) + " exceeds device buffer limit of " +
        std::to_string(device.maxBufferLength) + " bytes"
    );
  id<MTLBuffer> result = [device newBufferWithLength:std::max<uint64_t>(count * stride, 4U)
                                             options:options];
  if (result == nil)
    throw std::runtime_error("Could not allocate Metal BVH " + std::string(label.UTF8String));
  result.label = label;
  return result;
}
double complete(id<MTLCommandBuffer> command) {
  if (command == nil)
    throw std::runtime_error("Could not create BVH command buffer");
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error("Metal BVH command failed: " + error_text(command.error));
  return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}
MTLPrimitiveAccelerationStructureDescriptor *
primitive_descriptor(uint32_t count, id<MTLBuffer> bounds) {
  auto *geometry = [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
  geometry.boundingBoxBuffer = bounds;
  geometry.boundingBoxCount = count;
  geometry.boundingBoxStride = sizeof(BvhBounds);
  geometry.opaque = NO;
  geometry.allowDuplicateIntersectionFunctionInvocation = NO;
  auto *descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
  descriptor.geometryDescriptors = @[ geometry ];
  return descriptor;
}

BvhPipeline make_bvh_pipeline(
    GpuRaytraceResources &gpu,
    NSString *kernel_name,
    NSString *intersection_name,
    MTLFunctionConstantValues *constants,
    bool scene_mode,
    NSString *missing_intersection_name
) {
  NSError *error = nil;
  id<MTLFunction> kernel = constants == nil ? [gpu.library() newFunctionWithName:kernel_name]
                                            : [gpu.library() newFunctionWithName:kernel_name
                                                                  constantValues:constants
                                                                           error:&error];
  id<MTLFunction> intersection = constants == nil
                                     ? [gpu.library() newFunctionWithName:intersection_name]
                                     : [gpu.library() newFunctionWithName:intersection_name
                                                           constantValues:constants
                                                                    error:&error];
  if (kernel == nil || intersection == nil)
    throw std::runtime_error("Could not specialize BVH shader: " + error_text(error));
  auto *descriptor = [[MTLComputePipelineDescriptor alloc] init];
  descriptor.computeFunction = kernel;
  descriptor.linkedFunctions = [[MTLLinkedFunctions alloc] init];
  descriptor.linkedFunctions.functions = @[ intersection ];
  id<MTLFunction> missing = nil;
  if (scene_mode) {
    missing = [gpu.library() newFunctionWithName:missing_intersection_name == nil
                                                     ? @"terrain_tile_intersection"
                                                     : missing_intersection_name];
    if (missing == nil)
      throw std::runtime_error("Could not load scene missing-tile intersection function");
    descriptor.linkedFunctions.functions = @[ intersection, missing ];
  }
  BvhPipeline result;
  result.state = [gpu.device() newComputePipelineStateWithDescriptor:descriptor
                                                             options:MTLPipelineOptionNone
                                                          reflection:nil
                                                               error:&error];
  if (result.state == nil)
    throw std::runtime_error("Could not link BVH pipeline: " + error_text(error));
  auto *table = [MTLIntersectionFunctionTableDescriptor intersectionFunctionTableDescriptor];
  table.functionCount = 1;
  result.table = [result.state newIntersectionFunctionTableWithDescriptor:table];
  auto handle = [result.state functionHandleWithFunction:intersection];
  if (result.table == nil || handle == nil)
    throw std::runtime_error("Could not create BVH intersection table");
  [result.table setFunction:handle atIndex:0];
  if (scene_mode) {
    result.missing_table = [result.state newIntersectionFunctionTableWithDescriptor:table];
    result.coverage_table = [result.state newIntersectionFunctionTableWithDescriptor:table];
    auto missing_handle = [result.state functionHandleWithFunction:missing];
    if (result.missing_table == nil || result.coverage_table == nil || missing_handle == nil)
      throw std::runtime_error("Could not create scene missing-tile intersection table");
    [result.missing_table setFunction:missing_handle atIndex:0];
    [result.coverage_table setFunction:missing_handle atIndex:0];
  }
  return result;
}

} // namespace panorama::bvh_resources
