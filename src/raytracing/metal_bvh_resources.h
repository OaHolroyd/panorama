#pragma once
#include "metal_bvh_types.metalh"
#include <string>
#include <vector>

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

/// Index each source's coverage/ownership/blocker lists with nested XY bounds.
/// The first source_count records and the order of their leaf polygons stay intact.
void index_coverage_polygons(std::vector<BvhCoveragePolygon> &polygons, uint32_t source_count);

/// Pipeline-local function tables; acceleration structures are shared by all pipelines.
struct BvhPipeline {
  id<MTLComputePipelineState> state;
  id<MTLIntersectionFunctionTable> table;
  id<MTLIntersectionFunctionTable> missing_table;
  id<MTLIntersectionFunctionTable> coverage_table;
};

BvhPipeline make_bvh_pipeline(
    GpuRaytraceResources &gpu,
    NSString *kernel_name,
    NSString *intersection_name,
    MTLFunctionConstantValues *constants = nil,
    bool scene_mode = false,
    NSString *missing_intersection_name = nil,
    NSString *coverage_intersection_name = nil
);

} // namespace panorama::bvh_resources
