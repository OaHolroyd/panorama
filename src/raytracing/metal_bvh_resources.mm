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

void index_coverage_polygons(std::vector<BvhCoveragePolygon> &polygons, uint32_t source_count) {
  std::vector<BvhCoveragePolygon> indexed(polygons.begin(), polygons.begin() + source_count);
  indexed.reserve(polygons.size() + polygons.size() / 4U);
  const auto append = [&](auto &&self, uint32_t offset, uint32_t count) -> void {
    if (count <= 8U) {
      indexed.insert(indexed.end(), polygons.begin() + offset, polygons.begin() + offset + count);
      return;
    }
    const size_t parent = indexed.size();
    indexed.push_back({});
    const uint32_t half = count / 2U;
    self(self, offset, half);
    self(self, offset + half, count - half);
    BvhCoveragePolygon bounds = {};
    bounds.skip_count = checked_count(indexed.size() - parent - 1U);
    bounds.minimum_x = bounds.minimum_y = std::numeric_limits<float>::infinity();
    bounds.maximum_x = bounds.maximum_y = -std::numeric_limits<float>::infinity();
    // Visit immediate children only: each child's box already includes its descendants.
    for (size_t child = parent + 1U; child < indexed.size();) {
      const auto &box = indexed[child];
      bounds.minimum_x = std::min(bounds.minimum_x, box.minimum_x);
      bounds.minimum_y = std::min(bounds.minimum_y, box.minimum_y);
      bounds.maximum_x = std::max(bounds.maximum_x, box.maximum_x);
      bounds.maximum_y = std::max(bounds.maximum_y, box.maximum_y);
      child += 1U + box.skip_count;
    }
    indexed[parent] = bounds;
  };
  const auto append_range = [&](uint32_t &offset, uint32_t &count) {
    const uint32_t start = checked_count(indexed.size());
    append(append, offset, count);
    offset = start;
    count = checked_count(indexed.size()) - start;
  };
  for (uint32_t source = 0U; source < source_count; ++source) {
    auto footprint = polygons[source];
    const bool shared = footprint.ownership_offset == footprint.coverage_offset &&
                        footprint.ownership_count == footprint.coverage_count;
    append_range(footprint.coverage_offset, footprint.coverage_count);
    if (shared) {
      footprint.ownership_offset = footprint.coverage_offset;
      footprint.ownership_count = footprint.coverage_count;
    } else {
      append_range(footprint.ownership_offset, footprint.ownership_count);
    }
    append_range(footprint.blocker_offset, footprint.blocker_count);
    indexed[source] = footprint;
  }
  polygons = std::move(indexed);
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
    NSString *missing_intersection_name,
    NSString *coverage_intersection_name
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
  id<MTLFunction> coverage = nil;
  if (scene_mode) {
    missing = [gpu.library() newFunctionWithName:missing_intersection_name == nil
                                                     ? @"terrain_tile_intersection"
                                                     : missing_intersection_name];
    if (missing == nil)
      throw std::runtime_error("Could not load scene missing-tile intersection function");
  }
  coverage = scene_mode && coverage_intersection_name == nil
                 ? missing
                 : [gpu.library() newFunctionWithName:coverage_intersection_name == nil
                                                          ? @"terrain_tile_intersection"
                                                          : coverage_intersection_name];
  if (coverage == nil)
    throw std::runtime_error("Could not load terrain coverage intersection function");
  if ([coverage.name isEqualToString:intersection.name])
    coverage = intersection;
  auto *linked = [NSMutableArray arrayWithObject:intersection];
  if (missing != nil && missing != intersection)
    [linked addObject:missing];
  if (coverage != intersection && coverage != missing)
    [linked addObject:coverage];
  descriptor.linkedFunctions.functions = linked;

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
  result.coverage_table = [result.state newIntersectionFunctionTableWithDescriptor:table];
  auto coverage_handle = [result.state functionHandleWithFunction:coverage];
  if (result.coverage_table == nil || coverage_handle == nil)
    throw std::runtime_error("Could not create terrain coverage intersection table");
  [result.coverage_table setFunction:coverage_handle atIndex:0];
  if (scene_mode) {
    result.missing_table = [result.state newIntersectionFunctionTableWithDescriptor:table];
    auto missing_handle = [result.state functionHandleWithFunction:missing];
    if (result.missing_table == nil || missing_handle == nil)
      throw std::runtime_error("Could not create scene missing-tile intersection table");
    [result.missing_table setFunction:missing_handle atIndex:0];
  }

  return result;
}

} // namespace panorama::bvh_resources
