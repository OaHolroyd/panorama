#pragma once
#include "metal_bvh_types.metalh"
#include <string>

namespace panorama::bvh_resources {
std::string error_text(NSError *error);
uint32_t checked_count(uint64_t count);
id<MTLBuffer> buffer(
    id<MTLDevice> device,
    uint64_t count,
    size_t stride,
    NSString *label,
    MTLResourceOptions options = MTLResourceStorageModeShared
);
double complete(id<MTLCommandBuffer> command);
MTLPrimitiveAccelerationStructureDescriptor *
primitive_descriptor(uint32_t count, id<MTLBuffer> bounds);

/// Pipeline-local function tables; acceleration structures are shared by all pipelines.
struct BvhPipeline {
  id<MTLComputePipelineState> state;
  id<MTLIntersectionFunctionTable> table;
  id<MTLIntersectionFunctionTable> missing_table;
};

BvhPipeline make_bvh_pipeline(
    GpuRaytraceResources &gpu,
    NSString *kernel_name,
    NSString *intersection_name,
    MTLFunctionConstantValues *constants = nil,
    bool scene_mode = false
);

} // namespace panorama::bvh_resources
