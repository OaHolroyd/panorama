#include "metal_bvh_trace.h"
#include "metal_bvh_types.metalh"
#include "terrain_tile_bvh.h"
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
using namespace bvh_resources;
namespace {
static_assert(sizeof(BvhTile) == 48U);
static_assert(sizeof(BvhBlock) == 20U);
static_assert(sizeof(BvhBounds) == sizeof(MTLAxisAlignedBoundingBox));
static_assert(sizeof(BvhParameters) == 52U);
static_assert(sizeof(BvhChunk) == 80U);
static_assert(sizeof(BvhRayState) == 16U);

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
enum class TraceMode { Batch, Scene, Shadows };
using Pipeline = BvhPipeline;
struct Hierarchy {
  id<MTLBuffer> instances, chunks;
  id<MTLAccelerationStructure> acceleration;
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
  std::array<Pipeline, 4> scene_pipelines = {};
  std::array<Pipeline, 4> shadow_pipelines = {};
  Pipeline selector;
  id<MTLBuffer> parameters, dummy, tiles, rays, work;
  id<MTLAccelerationStructure> catalogue;
  std::span<const BvhTile> tile_metadata;
  uint64_t catalogue_generation = 0;
  std::vector<double> tile_shells;
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
  Hierarchy scene;
  std::vector<Entry *> scene_entries;
  id<MTLBuffer> scene_resident, scene_missing_count;
  id<MTLBuffer> scene_shadows, shadow_missing_count;
  float sun[4] = {};

  // Release indirect references before eviction, and before observer/LOD
  // changes invalidate instance transforms or the selected tile variants.
  void invalidate_scene() {
    scene = {};
    scene_entries.clear();
    scene_resident = nil;
    stats.scene_bytes = 0U;
  }

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
    scene_missing_count = buffer(gpu.device(), 1, sizeof(uint32_t), @"scene missing ray count");
    shadow_missing_count = buffer(gpu.device(), 1, sizeof(uint32_t), @"shadow missing ray count");
    selector = make_bvh_pipeline(gpu, @"select_terrain_tiles", @"terrain_tile_intersection");
  }

  Pipeline &pipeline(TraceMode mode = TraceMode::Batch) {
    const bool scene_mode = mode != TraceMode::Batch;
    const uint32_t index = (gpu.bilinear_collisions() ? 1U : 0U) | (gpu.c1_normals() ? 2U : 0U);
    auto &result = mode == TraceMode::Shadows ? shadow_pipelines[index]
                   : scene_mode               ? scene_pipelines[index]
                                              : pipelines[index];
    if (result.state == nil) {
      const bool bilinear = gpu.bilinear_collisions(), smooth = gpu.c1_normals();
      auto *constants = [[MTLFunctionConstantValues alloc] init];
      const bool gradients = mode != TraceMode::Shadows && outputs.surface_gradients;
      const bool elevations = mode != TraceMode::Shadows && outputs.elevations;
      const bool debugging = mode != TraceMode::Shadows && outputs.debugging_info;
      [constants setConstantValue:&gradients type:MTLDataTypeBool atIndex:0];
      [constants setConstantValue:&elevations type:MTLDataTypeBool atIndex:1];
      [constants setConstantValue:&debugging type:MTLDataTypeBool atIndex:2];
      [constants setConstantValue:&bilinear type:MTLDataTypeBool atIndex:3];
      [constants setConstantValue:&smooth type:MTLDataTypeBool atIndex:4];
      result = make_bvh_pipeline(
          gpu,
          mode == TraceMode::Shadows ? @"trace_scene_shadows"
          : scene_mode               ? @"trace_terrain_scene"
                                     : @"trace_terrain_bvh",
          @"terrain_bvh_intersection",
          constants,
          scene_mode
      );
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
    auto size_buffer = compact ? buffer(gpu.device(), 1, sizeof(uint64_t), @"compacted size") : nil;
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
    invalidate_scene();
    auto &shared = gpu.tile_bvh();
    tiles = shared.tiles();
    catalogue = shared.acceleration();
    tile_metadata = shared.metadata();
    catalogue_generation = shared.generation();
    const auto &sources = manager->sources();
    const auto origin = sources[manager->observer_source_index()].key;
    tile_shells.resize(sources.size());
    for (size_t i = 0; i < sources.size(); ++i)
      tile_shells[i] = std::abs(double(sources[i].key.row) - double(origin.row)) +
                       std::abs(double(sources[i].key.column) - double(origin.column));
    stats.catalogue_bytes = shared.bytes();
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
      invalidate_scene();
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
    invalidate_scene();
    cache.emplace(key, std::move(entry));
    timer.add_work("BVH tile loading/building", std::chrono::steady_clock::now() - started);
    return result;
  }

  Hierarchy make_hierarchy(const std::vector<Entry *> &batch, Timer &timer) {
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
    return {instances, chunks, acceleration};
  }

  void trace_hierarchy(
      const std::vector<Entry *> &batch,
      const Hierarchy &hierarchy,
      bool scene_mode,
      Timer &timer,
      id<MTLCommandBuffer> command = nil,
      bool shadows = false
  ) {
    std::memcpy(parameters.contents, &current, sizeof(current));
    auto &p = pipeline(
        shadows      ? TraceMode::Shadows
        : scene_mode ? TraceMode::Scene
                     : TraceMode::Batch
    );
    [p.table setBuffer:hierarchy.chunks offset:0 atIndex:0];
    [p.table setBuffer:parameters offset:0 atIndex:1];
    const bool submit = command == nil;
    if (submit)
      command = [gpu.command_queue() commandBuffer];
    auto encoder = [command computeCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("Could not encode BVH trace");
    [encoder setComputePipelineState:p.state];
    [encoder setAccelerationStructure:hierarchy.acceleration atBufferIndex:0];
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
    if (scene_mode) {
      [p.missing_table setBuffer:tiles offset:0 atIndex:0];
      [p.missing_table setBuffer:scene_resident offset:0 atIndex:1];
      [encoder setAccelerationStructure:catalogue atBufferIndex:13];
      [encoder setIntersectionFunctionTable:p.missing_table atBufferIndex:14];
      [encoder setBuffer:shadows ? shadow_missing_count : scene_missing_count offset:0 atIndex:15];
      [encoder useResource:catalogue usage:MTLResourceUsageRead];
      [encoder useResource:scene_resident usage:MTLResourceUsageRead];
      [encoder useResource:p.missing_table usage:MTLResourceUsageRead];
    }
    if (shadows) {
      [encoder setBytes:sun length:sizeof(sun) atIndex:16];
      [encoder setBuffer:scene_shadows offset:0 atIndex:17];
    }
    [encoder useResource:hierarchy.chunks usage:MTLResourceUsageRead];
    [encoder useResource:parameters usage:MTLResourceUsageRead];
    [encoder useResource:hierarchy.acceleration usage:MTLResourceUsageRead];
    [encoder useResource:p.table usage:MTLResourceUsageRead];
    for (Entry *entry : batch) {
      [encoder useResource:entry->vertices usage:MTLResourceUsageRead];
      [encoder useResource:entry->blocks usage:MTLResourceUsageRead];
      [encoder useResource:entry->bounds usage:MTLResourceUsageRead];
      [encoder useResource:entry->acceleration usage:MTLResourceUsageRead];
    }
    dispatch(encoder, p.state, current.work_count);
    [encoder endEncoding];
    if (!submit)
      return;
    const double gpu_ms = complete(command);
    stats.trace_gpu_ms += gpu_ms;
    ++stats.trace_passes;
    ++stats.submissions;
    if (scene_mode)
      ++stats.scene_passes;
    timer.add_work("GPU BVH detail traversal", gpu_ms);
    // Tables otherwise retain their previous argument buffers between batches.
    [p.table setBuffer:nil offset:0 atIndex:0];
    [p.missing_table setBuffer:nil offset:0 atIndex:0];
    [p.missing_table setBuffer:nil offset:0 atIndex:1];
  }

  void trace_batch(
      const std::vector<Entry *> &batch,
      const std::vector<uint32_t> &indices,
      Timer &timer
  ) {
    const auto hierarchy = make_hierarchy(batch, timer);
    std::memcpy(work.contents, indices.data(), indices.size() * sizeof(uint32_t));
    current.work_count = checked_count(indices.size());
    trace_hierarchy(batch, hierarchy, false, timer);
  }

  // A cached scene can resolve rays across any number of resident tiles. The
  // GPU also checks the catalogue for missing candidates before accepting the
  // result, so unseen terrain cannot be mistaken for empty space.
  bool prepare_scene(Timer &timer) {
    if (scene.acceleration == nil) {
      std::vector<Entry *> entries;
      for (auto &[key, entry] : cache) {
        if (key.lod == manager->lod_for_source(key.source_index))
          entries.push_back(entry.get());
      }
      if (entries.empty())
        return false;
      auto hierarchy = make_hierarchy(entries, timer);
      auto residency =
          buffer(gpu.device(), current.source_count, sizeof(uint32_t), @"scene residency");
      std::memset(residency.contents, 0, residency.length);
      auto *resident = static_cast<uint32_t *>(residency.contents);
      for (Entry *entry : entries)
        resident[entry->key.source_index] = 1U;
      scene = std::move(hierarchy);
      scene_entries = std::move(entries);
      scene_resident = residency;
      ++stats.scene_builds;
      stats.scene_bytes = scene.instances.length + scene.chunks.length + scene.acceleration.size +
                          scene_resident.length;
    }
    return true;
  }

  bool trace_scene(Timer &timer) {
    if (!prepare_scene(timer))
      return false;
    *static_cast<uint32_t *>(scene_missing_count.contents) = 0U;
    current.work_count = current.trace.ray_count;
    trace_hierarchy(scene_entries, scene, true, timer);
    const uint32_t missing = *static_cast<const uint32_t *>(scene_missing_count.contents);
    stats.scene_fallback_rays += missing;
    return missing == 0U;
  }

  void select(Timer &timer) {
    std::memcpy(parameters.contents, &current, sizeof(current));
    [selector.table setBuffer:tiles offset:0 atIndex:0];
    [selector.table setBuffer:dummy offset:0 atIndex:1];
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

bool MetalBvhTrace::encode_scene(id<MTLCommandBuffer> command, Timer &timer) {
  State &state = *state_;
  if (command == nil || !state.prepare_scene(timer))
    return false;
  *static_cast<uint32_t *>(state.scene_missing_count.contents) = 0U;
  state.current.work_count = state.current.trace.ray_count;
  state.trace_hierarchy(state.scene_entries, state.scene, true, timer, command);
  return true;
}

bool MetalBvhTrace::scene_complete() {
  State &state = *state_;
  const uint32_t missing = *static_cast<const uint32_t *>(state.scene_missing_count.contents);
  state.stats.scene_fallback_rays += missing;
  ++state.stats.scene_passes;
  ++state.stats.trace_passes;
  ++state.stats.submissions;
  return missing == 0U;
}

bool MetalBvhTrace::encode_shadows(id<MTLCommandBuffer> command, double azimuth, double elevation) {
  State &state = *state_;
  Timer timer("Encode scene shadows");
  if (command == nil || !state.prepare_scene(timer))
    return false;
  const uint32_t count = state.current.trace.ray_count;
  // A cold primary trace leaves work_count at its final streaming batch size.
  // Shadows always cover the entire image, including sky visibility values.
  state.current.work_count = count;
  if (state.scene_shadows == nil || state.scene_shadows.length < count)
    state.scene_shadows = buffer(state.gpu.device(), count, 1U, @"scene shadow visibility");
  state.sun[0] = static_cast<float>(std::sin(azimuth));
  state.sun[1] = static_cast<float>(std::cos(azimuth));
  state.sun[2] = static_cast<float>(std::tan(elevation));
  state.sun[3] = elevation <= 0.0 ? -1.0F : std::cos(elevation) < 1e-6 ? 1.0F : 0.0F;
  *static_cast<uint32_t *>(state.shadow_missing_count.contents) = 0U;
  state.trace_hierarchy(state.scene_entries, state.scene, true, timer, command, true);
  return true;
}

bool MetalBvhTrace::shadows_complete() const {
  return *static_cast<const uint32_t *>(state_->shadow_missing_count.contents) == 0U;
}

id<MTLBuffer> MetalBvhTrace::shadow_visibility() const { return state_->scene_shadows; }

void MetalBvhTrace::prepare(
    TileManager &tiles,
    ObserverLocation observer,
    const RaytraceParameters &parameters,
    Timer &timer
) {
  State &state = *state_;
  state.manager = &tiles;
  state.observer = observer;
  state.current.trace = parameters;
  state.current.quantized = tiles.traces_quantized();
  state.current.observer_source = tiles.observer_source_index();
  state.current.source_count = checked_count(tiles.sources().size());
  state.current.hash_capacity = state.gpu.catalogue_hash_capacity();
  @autoreleasepool {
    (void)state.pipeline();
    (void)state.pipeline(TraceMode::Scene);
    auto &shared = state.gpu.tile_bvh();
    const auto started = std::chrono::steady_clock::now();
    if (shared.prepare(tiles, observer, parameters)) {
      ++state.stats.catalogue_builds;
      ++state.stats.submissions;
      state.stats.build_gpu_ms += shared.build_milliseconds();
      timer.add_work("BVH catalogue setup", std::chrono::steady_clock::now() - started);
    }
    if (state.catalogue_generation != shared.generation())
      state.update_catalogue();
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
  @autoreleasepool {
    if (state.trace_scene(timer))
      return;
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
