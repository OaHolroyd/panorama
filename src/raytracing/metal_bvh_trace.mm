#include "metal_bvh_trace.h"
#include "metal_bvh_types.metalh"
#include "timer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
namespace {
static_assert(sizeof(BvhTile) == 48U);
static_assert(sizeof(BvhBlock) == 20U);
static_assert(sizeof(BvhBounds) == sizeof(MTLAxisAlignedBoundingBox));
static_assert(sizeof(BvhParameters) == 52U);
static_assert(sizeof(BvhChunk) == 80U);
static_assert(sizeof(BvhRayState) == 16U);

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
    MTLResourceOptions options = MTLResourceStorageModeShared
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
void dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    uint32_t count
) {
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup),
                                1,
                                1
                            )];
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
struct Pipeline {
  id<MTLComputePipelineState> state;
  id<MTLIntersectionFunctionTable> table;
};
} // namespace

struct MetalBvhTrace::State {
  GpuRaytraceResources &gpu;
  uint32_t block_cells;
  GpuTraceOutputRequirements outputs;
  MetalBvhStatistics stats;
  TileManager *manager = nullptr;
  ObserverLocation observer = {};
  BvhParameters current = {};
  id<MTLComputePipelineState> bounds_pipeline;
  std::array<Pipeline, 4> pipelines = {};
  Pipeline selector;
  id<MTLBuffer> parameters, dummy, tiles, catalogue_bounds, rays, work;
  id<MTLAccelerationStructure> catalogue;
  std::vector<BvhTile> tile_metadata;
  std::vector<double> tile_shells;
  std::vector<uint32_t> selected_lods;
  uint32_t ray_capacity = 0;
  uint64_t clock = 0;

  struct Entry {
    TileVariant key;
    BvhTile local;
    id<MTLBuffer> vertices, blocks, bounds;
    id<MTLAccelerationStructure> acceleration;
    uint64_t bytes = 0, used = 0;
  };
  std::map<TileVariant, std::unique_ptr<Entry>> cache;

  State(
      GpuRaytraceResources &resources,
      uint32_t cells,
      GpuTraceOutputRequirements requested,
      uint64_t budget
  )
      : gpu(resources), block_cells(cells), outputs(requested) {
    if (cells == 0 || budget == 0)
      throw std::invalid_argument("BVH block and cache sizes must be positive");
    if (!gpu.device().supportsRaytracing)
      throw std::runtime_error(
          "This device does not support Metal ray tracing; use --raytracer software"
      );
    stats.budget_bytes = budget;
    NSError *error = nil;
    auto function = [gpu.library() newFunctionWithName:@"build_terrain_bvh_bounds"];
    bounds_pipeline = [gpu.device() newComputePipelineStateWithFunction:function error:&error];
    if (bounds_pipeline == nil)
      throw std::runtime_error("Could not create BVH bounds pipeline: " + error_text(error));
    parameters = buffer(gpu.device(), 1, sizeof(BvhParameters), @"parameters");
    dummy = buffer(gpu.device(), 1, sizeof(uint32_t), @"unused output");
    selector = make_pipeline(@"select_terrain_tiles", @"terrain_tile_intersection", nil);
  }

  Pipeline make_pipeline(
      NSString *kernel_name,
      NSString *intersection_name,
      MTLFunctionConstantValues *constants
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
    Pipeline result;
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
    return result;
  }
  Pipeline &pipeline() {
    const uint32_t index = (gpu.bilinear_collisions() ? 1U : 0U) | (gpu.c1_normals() ? 2U : 0U);
    auto &result = pipelines[index];
    if (result.state == nil) {
      const bool bilinear = gpu.bilinear_collisions(), smooth = gpu.c1_normals();
      auto *constants = [[MTLFunctionConstantValues alloc] init];
      [constants setConstantValue:&outputs.surface_gradients type:MTLDataTypeBool atIndex:0];
      [constants setConstantValue:&outputs.elevations type:MTLDataTypeBool atIndex:1];
      [constants setConstantValue:&outputs.debugging_info type:MTLDataTypeBool atIndex:2];
      [constants setConstantValue:&bilinear type:MTLDataTypeBool atIndex:3];
      [constants setConstantValue:&smooth type:MTLDataTypeBool atIndex:4];
      result = make_pipeline(@"trace_terrain_bvh", @"terrain_bvh_intersection", constants);
    }
    return result;
  }

  id<MTLAccelerationStructure> build(MTLAccelerationStructureDescriptor *descriptor, bool compact) {
    const auto sizes = [gpu.device() accelerationStructureSizesWithDescriptor:descriptor];
    id<MTLAccelerationStructure> original =
        [gpu.device() newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    if (original == nil)
      throw std::runtime_error("Could not allocate BVH acceleration structure");
    auto scratch = buffer(
        gpu.device(),
        sizes.buildScratchBufferSize,
        1,
        @"build scratch",
        MTLResourceStorageModePrivate
    );
    auto size_buffer = buffer(gpu.device(), 1, sizeof(uint64_t), @"compacted size");
    auto command = [gpu.command_queue() commandBuffer];
    auto encoder = [command accelerationStructureCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("Could not encode BVH build");
    [encoder buildAccelerationStructure:original
                             descriptor:descriptor
                          scratchBuffer:scratch
                    scratchBufferOffset:0];
    if (compact)
      [encoder writeCompactedAccelerationStructureSize:original
                                              toBuffer:size_buffer
                                                offset:0
                                          sizeDataType:MTLDataTypeULong];
    [encoder endEncoding];
    stats.build_gpu_ms += complete(command);
    ++stats.submissions;
    if (!compact)
      return original;
    const uint64_t size = *static_cast<const uint64_t *>(size_buffer.contents);
    if (size == 0 || size > sizes.accelerationStructureSize)
      throw std::runtime_error("Invalid compacted BVH size");
    id<MTLAccelerationStructure> result = [gpu.device() newAccelerationStructureWithSize:size];
    if (result == nil)
      throw std::runtime_error("Could not allocate compacted BVH");
    command = [gpu.command_queue() commandBuffer];
    encoder = [command accelerationStructureCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("Could not encode BVH compaction");
    [encoder copyAndCompactAccelerationStructure:original toAccelerationStructure:result];
    [encoder endEncoding];
    stats.build_gpu_ms += complete(command);
    ++stats.submissions;
    return result;
  }

  void update_catalogue() {
    const auto &grid = manager->catalogue().grid();
    const auto &sources = manager->sources();
    const auto &geometry = manager->origin_geometry();
    tile_metadata.resize(sources.size());
    tile_shells.resize(sources.size());
    const auto origin_key = sources[manager->observer_source_index()].key;
    std::vector<BvhBounds> boxes(sources.size());
    const double k = current.trace.curvature_coefficient;
    for (size_t i = 0; i < sources.size(); ++i) {
      const auto &source = sources[i];
      tile_shells[i] = std::abs(double(source.key.row) - double(origin_key.row)) +
                       std::abs(double(source.key.column) - double(origin_key.column));
      const float x = static_cast<float>(
          grid.origin_x + double(source.key.column) * grid.width - observer.easting
      );
      const float y = static_cast<float>(
          grid.origin_y - double(source.key.row + 1) * grid.width - observer.northing
      );
      const float width = static_cast<float>(grid.width);
      tile_metadata[i] = {x,
                          y,
                          static_cast<float>(geometry.cell_size),
                          geometry.cell_count,
                          0,
                          0,
                          1,
                          source.key.row,
                          source.key.column};
      const double near_x = std::clamp(0.0, double(x), double(x + width));
      const double near_y = std::clamp(0.0, double(y), double(y + width));
      const double far_x = std::max(std::abs(double(x)), std::abs(double(x + width)));
      const double far_y = std::max(std::abs(double(y)), std::abs(double(y + width)));
      const double far_lift = k * (far_x * far_x + far_y * far_y);
      const double near_lift = k * (near_x * near_x + near_y * near_y);
      // Cover the legacy triangle solver at the selected LOD as well as
      // float coordinate conversion and the allowed horizontal ray-length error.
      const double diagonal =
          std::sqrt(2.0) * geometry.cell_size * double(uint64_t(1) << (selected_lods[i] - 1U));
      const double guard =
          0.01 + 64 * std::numeric_limits<float>::epsilon() * (far_lift + 100000.0) +
          k * diagonal * (2 * std::hypot(far_x, far_y) + diagonal) + 2.01e-4 * far_lift;
      boxes[i] = {x,
                  y,
                  source.minimum_elevation.has_value()
                      ? float(*source.minimum_elevation - far_lift - guard)
                      : -1e30F,
                  x + width,
                  y + width,
                  source.maximum_elevation.has_value()
                      ? float(*source.maximum_elevation - near_lift + guard)
                      : 1e30F};
    }
    tiles = buffer(gpu.device(), sources.size(), sizeof(BvhTile), @"catalogue tiles");
    catalogue_bounds = buffer(gpu.device(), boxes.size(), sizeof(BvhBounds), @"catalogue bounds");
    std::memcpy(tiles.contents, tile_metadata.data(), tiles.length);
    std::memcpy(catalogue_bounds.contents, boxes.data(), catalogue_bounds.length);
    catalogue = build(primitive_descriptor(checked_count(boxes.size()), catalogue_bounds), false);
    ++stats.catalogue_builds;
    stats.catalogue_bytes = tiles.length + catalogue_bounds.length + catalogue.size;
  }

  bool make_room(uint64_t required, const std::vector<Entry *> &pinned) {
    if (required > stats.budget_bytes)
      throw std::runtime_error(
          "BVH cache cannot build one full-resolution tile: needs " + std::to_string(required) +
          " bytes including build/compaction workspace; increase --bvh-cache-mib"
      );
    while (stats.resident_bytes > stats.budget_bytes - required) {
      auto victim = cache.end();
      for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (std::find(pinned.begin(), pinned.end(), it->second.get()) != pinned.end())
          continue;
        if (victim == cache.end() || it->second->used < victim->second->used)
          victim = it;
      }
      if (victim == cache.end())
        return false;
      stats.resident_bytes -= victim->second->bytes;
      cache.erase(victim);
      ++stats.evictions;
    }
    stats.peak_bytes = std::max(stats.peak_bytes, stats.resident_bytes + required);
    return true;
  }

  Entry *acquire(uint32_t source, const std::vector<Entry *> &pinned, Timer &timer) {
    const TileVariant key{source, manager->lod_for_source(source)};
    if (auto found = cache.find(key); found != cache.end()) {
      found->second->used = ++clock;
      ++stats.cache_hits;
      return found->second.get();
    }
    const auto &geometry = manager->origin_geometry();
    const uint32_t cells = geometry.cell_count >> (key.lod - 1U);
    const uint64_t side = uint64_t(cells) + 1U;
    const uint64_t block_side = (uint64_t(cells) + block_cells - 1U) / block_cells;
    const uint32_t count = checked_count(block_side * block_side);
    const uint64_t vertex_bytes =
        side * side * (manager->traces_quantized() ? sizeof(uint16_t) : sizeof(float));
    const auto sizes =
        [gpu.device() accelerationStructureSizesWithDescriptor:primitive_descriptor(count, nil)];
    const uint64_t base_bytes =
        vertex_bytes + uint64_t(count) * (sizeof(BvhBlock) + sizeof(BvhBounds));
    // Reserve both acceleration structures during compaction. No terrain
    // allocations or I/O happen until this peak fits; all submissions finish
    // and autoreleased build objects drain before the next admission.
    const uint64_t peak = base_bytes + 2 * sizes.accelerationStructureSize +
                          sizes.buildScratchBufferSize + sizeof(BvhTile) + sizeof(uint64_t);
    if (!make_room(peak, pinned))
      return nullptr;
    const auto started = std::chrono::steady_clock::now();
    auto entry = std::make_unique<Entry>();
    entry->key = key;
    entry->used = ++clock;
    entry->vertices = buffer(gpu.device(), vertex_bytes, 1, @"tile vertices");
    entry->blocks = buffer(gpu.device(), count, sizeof(BvhBlock), @"tile blocks");
    entry->bounds = buffer(gpu.device(), count, sizeof(BvhBounds), @"tile bounds");
    const std::vector<uint8_t> unpinned(manager->slot_capacity(), 0U);
    manager->request(source, 0);
    while (manager->slot_for_source(source) == manager->slot_capacity()) {
      (void)manager->install_available(unpinned, timer);
      if (manager->slot_for_source(source) == manager->slot_capacity())
        manager->wait_for_available();
    }
    const uint32_t slot = manager->slot_for_source(source);
    const auto bindings = manager->bindings();
    const std::byte *data = static_cast<const std::byte *>(bindings.vertex_atlas.contents);
    int base = 0;
    if (manager->traces_quantized()) {
      data += size_t(slot) * bindings.quantized_layout.record_stride;
      std::memcpy(&base, data + bindings.quantized_layout.elevation_base_offset, sizeof(base));
      data += bindings.quantized_layout.vertex_offset;
    } else {
      const uint64_t full_side = uint64_t(geometry.cell_count) + 1U;
      data += size_t(slot) * full_side * full_side * sizeof(float);
    }
    std::memcpy(entry->vertices.contents, data, vertex_bytes);
    const float delta = float(geometry.cell_size) * float(1U << (key.lod - 1U));
    const float half_width = 0.5F * delta * float(cells);
    const auto tile_key = manager->sources()[source].key;
    entry->local =
        {-half_width, -half_width, delta, cells, 0, base, key.lod, tile_key.row, tile_key.column};
    auto *blocks = static_cast<BvhBlock *>(entry->blocks.contents);
    uint32_t index = 0;
    for (uint32_t y = 0; y < cells;) {
      const uint32_t height = std::min(block_cells, cells - y);
      for (uint32_t x = 0; x < cells;) {
        const uint32_t width = std::min(block_cells, cells - x);
        blocks[index++] = {0, x, y, width, height};
        x += width;
      }
      y += height;
    }
    auto metadata = buffer(gpu.device(), 1, sizeof(BvhTile), @"tile build metadata");
    std::memcpy(metadata.contents, &entry->local, sizeof(BvhTile));
    current.primitive_count = count;
    std::memcpy(parameters.contents, &current, sizeof(current));
    auto command = [gpu.command_queue() commandBuffer];
    auto encoder = [command computeCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("Could not encode tile bounds");
    [encoder setComputePipelineState:bounds_pipeline];
    [encoder setBuffer:entry->vertices offset:0 atIndex:0];
    [encoder setBuffer:metadata offset:0 atIndex:1];
    [encoder setBuffer:entry->blocks offset:0 atIndex:2];
    [encoder setBuffer:entry->bounds offset:0 atIndex:3];
    [encoder setBuffer:parameters offset:0 atIndex:4];
    dispatch(encoder, bounds_pipeline, count);
    [encoder endEncoding];
    stats.build_gpu_ms += complete(command);
    ++stats.submissions;
    entry->acceleration = build(primitive_descriptor(count, entry->bounds), true);
    entry->bytes = base_bytes + entry->acceleration.size;
    stats.resident_bytes += entry->bytes;
    ++stats.builds;
    Entry *result = entry.get();
    cache.emplace(key, std::move(entry));
    timer.add_work("BVH tile loading/building", std::chrono::steady_clock::now() - started);
    return result;
  }

  void trace_batch(
      const std::vector<Entry *> &batch,
      const std::vector<uint32_t> &indices,
      Timer &timer
  ) {
    const auto started = std::chrono::steady_clock::now();
    auto instances = buffer(
        gpu.device(),
        batch.size(),
        sizeof(MTLAccelerationStructureInstanceDescriptor),
        @"batch instances"
    );
    auto chunks = buffer(gpu.device(), batch.size(), sizeof(BvhChunk), @"batch resources");
    auto *transforms =
        static_cast<MTLAccelerationStructureInstanceDescriptor *>(instances.contents);
    auto *resources = static_cast<BvhChunk *>(chunks.contents);
    auto *structures = [[NSMutableArray<id<MTLAccelerationStructure>> alloc] init];
    for (size_t i = 0; i < batch.size(); ++i) {
      Entry &entry = *batch[i];
      BvhTile tile = entry.local;
      tile.x_min = tile_metadata[entry.key.source_index].x_min;
      tile.y_min = tile_metadata[entry.key.source_index].y_min;
      resources[i] = {entry.vertices.gpuAddress,
                      entry.blocks.gpuAddress,
                      entry.bounds.gpuAddress,
                      tile,
                      entry.key.source_index,
                      0};
      const double ax = double(tile.x_min) - entry.local.x_min;
      const double ay = double(tile.y_min) - entry.local.y_min;
      const double k = current.trace.curvature_coefficient;
      auto &instance = transforms[i];
      instance = {};
      // z_observer = z_anchor - 2*k*a dot (u,v) - k*|a|^2.
      // XY translation and Z shear preserve the original horizontal ray t.
      instance.transformationMatrix.columns[0] = {1, 0, float(-2 * k * ax)};
      instance.transformationMatrix.columns[1] = {0, 1, float(-2 * k * ay)};
      instance.transformationMatrix.columns[2] = {0, 0, 1};
      instance.transformationMatrix.columns[3] = {float(ax),
                                                  float(ay),
                                                  float(-k * (ax * ax + ay * ay))};
      instance.mask = 0xff;
      instance.accelerationStructureIndex = checked_count(i);
      [structures addObject:entry.acceleration];
    }
    auto *descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];
    descriptor.instanceCount = batch.size();
    descriptor.instanceDescriptorBuffer = instances;
    descriptor.instancedAccelerationStructures = structures;
    auto acceleration = build(descriptor, false);
    ++stats.instance_builds;
    timer.add_work("BVH instance setup", std::chrono::steady_clock::now() - started);
    std::memcpy(work.contents, indices.data(), indices.size() * sizeof(uint32_t));
    current.work_count = checked_count(indices.size());
    std::memcpy(parameters.contents, &current, sizeof(current));
    auto &p = pipeline();
    [p.table setBuffer:chunks offset:0 atIndex:0];
    [p.table setBuffer:parameters offset:0 atIndex:1];
    auto command = [gpu.command_queue() commandBuffer];
    auto encoder = [command computeCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("Could not encode BVH trace");
    [encoder setComputePipelineState:p.state];
    [encoder setAccelerationStructure:acceleration atBufferIndex:0];
    [encoder setIntersectionFunctionTable:p.table atBufferIndex:1];
    [encoder setBuffer:gpu.ray_directions() offset:0 atIndex:2];
    [encoder setBuffer:gpu.distances() offset:0 atIndex:3];
    [encoder setBuffer:outputs.elevations ? gpu.elevations() : dummy offset:0 atIndex:4];
    [encoder setBuffer:outputs.surface_gradients ? gpu.surface_gradients() : dummy
                offset:0
               atIndex:5];
    [encoder setBuffer:outputs.debugging_info ? gpu.num_steps() : dummy offset:0 atIndex:6];
    [encoder setBuffer:outputs.debugging_info ? gpu.num_evaluations() : dummy offset:0 atIndex:7];
    [encoder setBuffer:parameters offset:0 atIndex:8];
    [encoder setBuffer:tiles offset:0 atIndex:9];
    [encoder setBuffer:gpu.catalogue_hash() offset:0 atIndex:10];
    [encoder setBuffer:rays offset:0 atIndex:11];
    [encoder setBuffer:work offset:0 atIndex:12];
    [encoder useResource:chunks usage:MTLResourceUsageRead];
    [encoder useResource:parameters usage:MTLResourceUsageRead];
    [encoder useResource:acceleration usage:MTLResourceUsageRead];
    [encoder useResource:p.table usage:MTLResourceUsageRead];
    for (Entry *entry : batch) {
      [encoder useResource:entry->vertices usage:MTLResourceUsageRead];
      [encoder useResource:entry->blocks usage:MTLResourceUsageRead];
      [encoder useResource:entry->bounds usage:MTLResourceUsageRead];
      [encoder useResource:entry->acceleration usage:MTLResourceUsageRead];
    }
    dispatch(encoder, p.state, current.work_count);
    [encoder endEncoding];
    const double gpu_ms = complete(command);
    stats.trace_gpu_ms += gpu_ms;
    ++stats.trace_passes;
    ++stats.submissions;
    timer.add_work("GPU BVH detail traversal", gpu_ms);
    // Tables otherwise retain their previous argument buffers between batches.
    [p.table setBuffer:nil offset:0 atIndex:0];
  }

  void select(Timer &timer) {
    std::memcpy(parameters.contents, &current, sizeof(current));
    [selector.table setBuffer:tiles offset:0 atIndex:0];
    auto command = [gpu.command_queue() commandBuffer];
    auto encoder = [command computeCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("Could not encode tile selection");
    [encoder setComputePipelineState:selector.state];
    [encoder setAccelerationStructure:catalogue atBufferIndex:0];
    [encoder setIntersectionFunctionTable:selector.table atBufferIndex:1];
    [encoder setBuffer:gpu.ray_directions() offset:0 atIndex:2];
    [encoder setBuffer:rays offset:0 atIndex:3];
    [encoder setBuffer:parameters offset:0 atIndex:4];
    [encoder setBuffer:tiles offset:0 atIndex:5];
    [encoder useResource:tiles usage:MTLResourceUsageRead];
    [encoder useResource:catalogue usage:MTLResourceUsageRead];
    [encoder useResource:selector.table usage:MTLResourceUsageRead];
    dispatch(encoder, selector.state, current.trace.ray_count);
    [encoder endEncoding];
    const double gpu_ms = complete(command);
    stats.selection_gpu_ms += gpu_ms;
    ++stats.selection_passes;
    ++stats.submissions;
    timer.add_work("GPU BVH tile selection", gpu_ms);
  }
};

MetalBvhTrace::MetalBvhTrace(
    GpuRaytraceResources &gpu,
    uint32_t cells,
    GpuTraceOutputRequirements outputs,
    uint64_t budget
)
    : state_(std::make_unique<State>(gpu, cells, outputs, budget)) {}
MetalBvhTrace::~MetalBvhTrace() = default;
MetalBvhStatistics MetalBvhTrace::statistics() const { return state_->stats; }

void MetalBvhTrace::prepare(
    TileManager &tiles,
    ObserverLocation observer,
    const RaytraceParameters &parameters,
    Timer &timer
) {
  State &state = *state_;
  std::vector<uint32_t> lods;
  for (uint32_t i = 0; i < tiles.sources().size(); ++i)
    lods.push_back(tiles.lod_for_source(i));
  const bool rebuild = state.catalogue == nil || state.observer.easting != observer.easting ||
                       state.observer.northing != observer.northing || state.selected_lods != lods;
  state.selected_lods = std::move(lods);
  state.manager = &tiles;
  state.observer = observer;
  state.current.trace = parameters;
  state.current.quantized = tiles.traces_quantized();
  state.current.observer_source = tiles.observer_source_index();
  state.current.source_count = checked_count(tiles.sources().size());
  state.current.hash_capacity = state.gpu.catalogue_hash_capacity();
  @autoreleasepool {
    (void)state.pipeline();
    if (rebuild) {
      const auto started = std::chrono::steady_clock::now();
      state.update_catalogue();
      timer.add_work("BVH catalogue setup", std::chrono::steady_clock::now() - started);
    }
  }
  if (parameters.ray_count > state.ray_capacity) {
    state.rays =
        buffer(state.gpu.device(), parameters.ray_count, sizeof(BvhRayState), @"ray continuations");
    state.work =
        buffer(state.gpu.device(), parameters.ray_count, sizeof(uint32_t), @"ray work indices");
    state.ray_capacity = parameters.ray_count;
  }
}

void MetalBvhTrace::trace(const RaytraceParameters &parameters, Timer &timer) {
  State &state = *state_;
  state.current.trace = parameters;
  auto *continuations = static_cast<BvhRayState *>(state.rays.contents);
  for (uint32_t i = 0; i < parameters.ray_count; ++i)
    continuations[i] = {0, 0, 0xffffffffU, 0};
  std::memset(state.gpu.distances().contents, 0, size_t(parameters.ray_count) * sizeof(float));
  if (state.outputs.elevations)
    std::memset(state.gpu.elevations().contents, 0, size_t(parameters.ray_count) * sizeof(float));
  if (state.outputs.surface_gradients)
    std::memset(
        state.gpu.surface_gradients().contents,
        0,
        size_t(parameters.ray_count) * sizeof(uint32_t)
    );
  if (state.outputs.debugging_info) {
    std::memset(state.gpu.num_steps().contents, 0, size_t(parameters.ray_count) * sizeof(float));
    std::memset(
        state.gpu.num_evaluations().contents,
        0,
        size_t(parameters.ray_count) * sizeof(float)
    );
  }
  for (uint32_t round = 0; round <= state.current.source_count; ++round) {
    @autoreleasepool {
      state.select(timer);
    }
    const auto grouping_started = std::chrono::steady_clock::now();
    const auto *rays = static_cast<const BvhRayState *>(state.rays.contents);
    double nearest_shell = std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < parameters.ray_count; ++i) {
      if (rays[i].done)
        continue;
      if (rays[i].source >= state.current.source_count || !(rays[i].exit > rays[i].progress))
        throw std::runtime_error("BVH tile selection failed to advance a ray");
      nearest_shell = std::min(nearest_shell, state.tile_shells[rays[i].source]);
    }
    // A straight ray's grid Manhattan distance from its origin tile only
    // increases. Finish one shell before the next: every ray that can reach a
    // source then arrives before that source is processed, even with a one-tile
    // cache. This prevents rebuilding a tile for each wave of incoming rays.
    std::map<uint32_t, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < parameters.ray_count; ++i) {
      if (rays[i].done)
        continue;
      if (state.tile_shells[rays[i].source] != nearest_shell)
        continue;
      groups[rays[i].source].push_back(i);
    }
    state.stats.grouping_cpu_ms += std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - grouping_started
    )
                                       .count();
    if (groups.empty())
      return;
    auto group = groups.begin();
    while (group != groups.end()) {
      std::vector<State::Entry *> batch;
      std::vector<uint32_t> indices;
      // Bound the small temporary instance hierarchy independently of scene size.
      while (group != groups.end() && batch.size() < 64U) {
        State::Entry *entry;
        @autoreleasepool {
          entry = state.acquire(group->first, batch, timer);
        }
        if (entry == nullptr)
          break;
        batch.push_back(entry);
        indices.insert(indices.end(), group->second.begin(), group->second.end());
        ++group;
      }
      if (batch.empty())
        throw std::logic_error("BVH cache cannot admit a tile");
      @autoreleasepool {
        state.trace_batch(batch, indices, timer);
      }
    }
  }
  throw std::runtime_error("BVH streaming exceeded the finite tile traversal limit");
}
} // namespace panorama
