#include "metal_bvh_trace.h"
#include "metal_bvh_types.metalh"
#include "terrain_tile_bvh.h"
#include "threadgroup_sizes.h"
#include "timer.h"
#include "trace_activity.h"

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
static_assert(sizeof(BvhTile) == 56U);
static_assert(sizeof(BvhAffinePatch) == 64U);
static_assert(sizeof(BvhCoveragePolygon) == 56U);
static_assert(sizeof(BvhCoverageVertex) == 8U);
static_assert(sizeof(BvhBlock) == 24U);
static_assert(sizeof(BvhBounds) == sizeof(MTLAxisAlignedBoundingBox));
static_assert(sizeof(BvhParameters) == 72U);
static_assert(sizeof(BvhChunk) == 104U);
static_assert(sizeof(BvhRayState) == 20U);

void dispatch_linear(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    uint32_t count
) {
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:threadgroups::bounded_linear(pipeline.maxTotalThreadsPerThreadgroup)];
}

void dispatch_image(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const RaytraceParameters &parameters
) {
  const MTLSize group = threadgroups::bounded_bvh(pipeline.maxTotalThreadsPerThreadgroup);
  if (group.height == 0)
    throw std::runtime_error("Raytracing pipeline cannot dispatch a 32-pixel BVH row");
  [encoder dispatchThreads:MTLSizeMake(parameters.image_width, parameters.image_height, 1)
      threadsPerThreadgroup:group];
}
enum class TraceMode { Batch, Scene, Shadows };
using Pipeline = BvhPipeline;
struct Hierarchy {
  id<MTLBuffer> instances, chunks;
  id<MTLAccelerationStructure> acceleration;
};

[[nodiscard]] BvhAffinePatch
affine_patch(const TerrainTransformPatch &patch, uint32_t source, double logical_scale) {
  const auto &transform = patch.transform;
  const double determinant = transform.determinant();
  return {
      float(transform.origin.x),
      float(transform.origin.y),
      float(transform.column_step.x / logical_scale),
      float(transform.column_step.y / logical_scale),
      float(transform.row_step.x / logical_scale),
      float(transform.row_step.y / logical_scale),
      float(logical_scale * transform.row_step.y / determinant),
      float(-logical_scale * transform.row_step.x / determinant),
      float(-logical_scale * transform.column_step.y / determinant),
      float(logical_scale * transform.column_step.x / determinant),
      patch.minimum_column,
      patch.minimum_row,
      patch.minimum_column + patch.cell_width,
      patch.minimum_row + patch.cell_height,
      source,
      float(transform.maximum_residual_metres),
  };
}
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
  id<MTLComputePipelineState> initialize_pipeline;
  std::array<Pipeline, 4> pipelines = {};
  std::array<Pipeline, 4> scene_pipelines = {};
  std::array<Pipeline, 4> shadow_pipelines = {};
  Pipeline tile_selector, patch_selector;
  id<MTLBuffer> parameters, dummy, tiles, coverage_polygons, coverage_vertices, candidate_patches,
      rays, work;
  id<MTLAccelerationStructure> catalogue, candidate_catalogue;
  std::span<const BvhTile> tile_metadata;
  uint64_t catalogue_generation = 0;
  std::vector<uint32_t> selected_lods;
  std::vector<double> tile_shells;
  uint32_t ray_capacity = 0;
  uint64_t clock = 0;

  struct Entry {
    TileVariant key;
    BvhTile local;
    id<MTLBuffer> vertices, blocks, bounds, transforms;
    id<MTLAccelerationStructure> acceleration;
    uint64_t bytes = 0, used = 0;
    bool transformed = false;
    float anchor_x = 0.0F, anchor_y = 0.0F;
    float vertical_offset = 0.0F;
  };
  std::map<TileVariant, std::unique_ptr<Entry>> cache;
  Hierarchy scene;
  std::vector<Entry *> scene_entries;
  id<MTLBuffer> scene_resident, scene_missing_count;
  id<MTLBuffer> scene_requested_sources, used_sources;
  std::array<id<MTLBuffer>, 2> scene_pending_rays = {};
  uint32_t scene_pending_output = 0U;
  id<MTLBuffer> scene_shadows, shadow_missing_count;
  id<MTLBuffer> shadow_tested;
  id<MTLBuffer> shadow_requested_sources;
  bool scene_requests_ready = false;
  bool shadow_requests_ready = false;
  double shadow_azimuth = 0.0, shadow_elevation = 0.0;
  float sun[4] = {};

  enum class SceneInvalidation { Catalogue, Lod, Admission, Eviction };

  // Release indirect references before eviction, and before observer/LOD
  // changes invalidate instance transforms or the selected tile variants.
  void invalidate_scene(SceneInvalidation reason) {
    if (scene.acceleration != nil) {
      switch (reason) {
      case SceneInvalidation::Catalogue:
        ++stats.scene_catalogue_invalidations;
        break;
      case SceneInvalidation::Lod:
        ++stats.scene_lod_invalidations;
        break;
      case SceneInvalidation::Admission:
        ++stats.scene_admission_invalidations;
        break;
      case SceneInvalidation::Eviction:
        ++stats.scene_eviction_invalidations;
        break;
      }
    }
    scene_requests_ready = false;
    shadow_requests_ready = false;
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
    function = [gpu.library() newFunctionWithName:@"initialize_bvh_continuations"];
    initialize_pipeline = [gpu.device() newComputePipelineStateWithFunction:function error:&error];
    if (initialize_pipeline == nil)
      throw std::runtime_error(
          "Could not create BVH initialization pipeline: " + error_text(error)
      );
    parameters = buffer(gpu.device(), 1, sizeof(BvhParameters), @"parameters");
    dummy = buffer(gpu.device(), 1, sizeof(uint32_t), @"unused output");
    scene_missing_count = buffer(gpu.device(), 1, sizeof(uint32_t), @"scene missing ray count");
    shadow_missing_count = buffer(gpu.device(), 1, sizeof(uint32_t), @"shadow missing ray count");
    tile_selector = make_bvh_pipeline(gpu, @"select_terrain_tiles", @"terrain_tile_intersection");
    patch_selector = make_bvh_pipeline(
        gpu,
        @"select_terrain_coverage_polygons",
        @"terrain_coverage_polygon_intersection"
    );
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
          scene_mode,
          current.transformed_catalogue ? @"terrain_patch_intersection" : nil,
          current.transformed_catalogue ? @"terrain_coverage_polygon_intersection" : nil
      );
    }
    return result;
  }

  id<MTLAccelerationStructure> build(
      MTLAccelerationStructureDescriptor *descriptor,
      bool compact,
      id<MTLCommandBuffer> command = nil
  ) {
    trace_activity::Scope activity("BVH build/compaction");
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
    if (command == nil)
      command = [gpu.command_queue() commandBuffer];
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
    invalidate_scene(SceneInvalidation::Catalogue);
    auto &shared = gpu.tile_bvh();
    tiles = shared.tiles();
    coverage_polygons = shared.coverage_polygons();
    coverage_vertices = shared.coverage_vertices();
    candidate_patches = shared.candidate_patches();
    catalogue = shared.acceleration();
    candidate_catalogue = shared.candidate_acceleration();
    tile_metadata = shared.metadata();
    catalogue_generation = shared.generation();
    current.coverage_step_limit = shared.coverage_step_limit();
    const auto &sources = manager->sources();
    tile_shells.resize(sources.size());
    if (current.transformed_catalogue) {
      const Coord render_observer =
          manager->catalogue().render_coordinate({observer.easting, observer.northing});
      std::vector<double> distances(sources.size(), std::numeric_limits<double>::infinity());
      double shell_width = 1.0;
      for (size_t i = 0; i < sources.size(); ++i) {
        for (const TerrainTransformPatch &patch : sources[i].transform_patches) {
          const double x0 = patch.minimum_column;
          const double y0 = patch.minimum_row;
          const double x1 = x0 + patch.cell_width;
          const double y1 = y0 + patch.cell_height;
          const std::array<Coord, 4> corners = {
              patch.transform.apply(x0, y0),
              patch.transform.apply(x1, y0),
              patch.transform.apply(x0, y1),
              patch.transform.apply(x1, y1),
          };
          double low_x = corners[0].x, high_x = corners[0].x;
          double low_y = corners[0].y, high_y = corners[0].y;
          for (const Coord corner : corners) {
            low_x = std::min(low_x, corner.x);
            high_x = std::max(high_x, corner.x);
            low_y = std::min(low_y, corner.y);
            high_y = std::max(high_y, corner.y);
          }
          const double dx = render_observer.x < low_x    ? low_x - render_observer.x
                            : render_observer.x > high_x ? render_observer.x - high_x
                                                         : 0.0;
          const double dy = render_observer.y < low_y    ? low_y - render_observer.y
                            : render_observer.y > high_y ? render_observer.y - high_y
                                                         : 0.0;
          distances[i] = std::min(distances[i], std::hypot(dx, dy));
          shell_width = std::max(shell_width, std::hypot(high_x - low_x, high_y - low_y));
        }
      }
      for (size_t i = 0; i < sources.size(); ++i)
        tile_shells[i] = std::floor(distances[i] / shell_width);
    } else {
      const auto origin = sources[manager->observer_source_index()].key;
      for (size_t i = 0; i < sources.size(); ++i)
        tile_shells[i] = std::abs(double(sources[i].key.row) - double(origin.row)) +
                         std::abs(double(sources[i].key.column) - double(origin.column));
    }
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
        const bool obsolete = it->first.lod != manager->lod_for_source(it->first.source_index);
        const bool victim_obsolete =
            victim != cache.end() &&
            victim->first.lod != manager->lod_for_source(victim->first.source_index);
        if (victim == cache.end() || (obsolete && !victim_obsolete) ||
            (obsolete == victim_obsolete && it->second->used < victim->second->used))
          victim = it;
      }
      if (victim == cache.end())
        return false;
      stats.resident_bytes -= victim->second->bytes;
      invalidate_scene(SceneInvalidation::Eviction);
      cache.erase(victim);
      ++stats.evictions;
    }
    stats.peak_bytes = std::max(stats.peak_bytes, stats.resident_bytes + required);
    return true;
  }

  // The completed GPU pass identifies primary hits (including provisional
  // hits) and shadow occluders. Protect these during repair, rather than every
  // old tile that happens to remain in the scene. No command may be in flight.
  std::vector<Entry *> used_entries() {
    const auto *used = static_cast<const uint32_t *>(used_sources.contents);
    const uint64_t stamp = ++clock;
    std::vector<Entry *> entries;
    for (auto &[key, entry] : cache) {
      if (key.lod == manager->lod_for_source(key.source_index) && used[key.source_index]) {
        entry->used = stamp;
        entries.push_back(entry.get());
      }
    }
    return entries;
  }

  Entry *acquire(
      uint32_t source,
      const std::vector<Entry *> &pinned,
      Timer &timer,
      bool optional = false
  ) {
    const TileVariant key{source, manager->lod_for_source(source)};
    if (auto found = cache.find(key); found != cache.end()) {
      found->second->used = ++clock;
      ++stats.cache_hits;
      return found->second.get();
    }
    trace_activity::Scope activity("BVH tile admission/loading/building");
    const auto &geometry = manager->origin_geometry();
    const TerrainSource &source_metadata = manager->sources()[source];
    const bool transformed = !source_metadata.transform_patches.empty();
    const uint32_t cells = geometry.cell_count >> (key.lod - 1U);
    const uint64_t side = uint64_t(cells) + 1U;
    std::vector<BvhBlock> block_metadata;
    if (transformed) {
      const uint32_t scale = 1U << (key.lod - 1U);
      for (size_t patch_index = 0; patch_index < source_metadata.transform_patches.size();
           ++patch_index) {
        const TerrainTransformPatch &patch = source_metadata.transform_patches[patch_index];
        const uint32_t x0 = (patch.minimum_column + scale - 1U) / scale;
        const uint32_t y0 = (patch.minimum_row + scale - 1U) / scale;
        const uint32_t x1 = (patch.minimum_column + patch.cell_width) / scale;
        const uint32_t y1 = (patch.minimum_row + patch.cell_height) / scale;
        for (uint32_t y = y0; y < y1;) {
          const uint32_t height = std::min(block_cells, y1 - y);
          for (uint32_t x = x0; x < x1;) {
            const uint32_t width = std::min(block_cells, x1 - x);
            block_metadata.push_back({0U, x, y, width, height, checked_count(patch_index)});
            x += width;
          }
          y += height;
        }
      }
    } else {
      for (uint32_t y = 0; y < cells;) {
        const uint32_t height = std::min(block_cells, cells - y);
        for (uint32_t x = 0; x < cells;) {
          const uint32_t width = std::min(block_cells, cells - x);
          block_metadata.push_back({0U, x, y, width, height, 0U});
          x += width;
        }
        y += height;
      }
    }
    if (block_metadata.empty())
      throw std::logic_error("Selected terrain LOD has no owned cells");
    const uint32_t count = checked_count(block_metadata.size());
    const uint64_t vertex_bytes =
        side * side * (manager->traces_quantized() ? sizeof(uint16_t) : sizeof(float));
    const auto sizes =
        [gpu.device() accelerationStructureSizesWithDescriptor:primitive_descriptor(count, nil)];
    const uint64_t transform_count = transformed ? source_metadata.transform_patches.size() : 1U;
    const uint64_t base_bytes = vertex_bytes +
                                uint64_t(count) * (sizeof(BvhBlock) + sizeof(BvhBounds)) +
                                transform_count * sizeof(BvhAffinePatch);
    // Reserve both acceleration structures during compaction. No terrain
    // allocations or I/O happen until this peak fits; all submissions finish
    // and autoreleased build objects drain before the next admission.
    const uint64_t peak = base_bytes + 2 * sizes.accelerationStructureSize +
                          sizes.buildScratchBufferSize + sizeof(BvhTile) + sizeof(uint64_t);
    if (optional && peak > stats.budget_bytes)
      return nullptr;
    if (!make_room(peak, pinned))
      return nullptr;
    const auto started = std::chrono::steady_clock::now();
    auto entry = std::make_unique<Entry>();
    entry->key = key;
    entry->used = ++clock;
    entry->vertices = buffer(gpu.device(), vertex_bytes, 1, @"tile vertices");
    entry->blocks = buffer(gpu.device(), count, sizeof(BvhBlock), @"tile blocks");
    entry->bounds = buffer(gpu.device(), count, sizeof(BvhBounds), @"tile bounds");
    entry->transforms =
        buffer(gpu.device(), transform_count, sizeof(BvhAffinePatch), @"tile transforms");
    entry->transformed = transformed;
    entry->vertical_offset = static_cast<float>(source_metadata.vertical_offset_metres);
    const std::vector<uint8_t> unpinned(manager->slot_capacity(), 0U);
    {
      trace_activity::Scope loading("BVH terrain loading");
      manager->request(source, 0);
      while (manager->slot_for_source(source) == manager->slot_capacity()) {
        (void)manager->install_available(unpinned, timer);
        if (manager->slot_for_source(source) == manager->slot_capacity())
          manager->wait_for_available();
      }
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
    const double logical_scale = transformed ? source_metadata.effective_cell_size_metres : 1.0;
    const float delta = transformed ? float(logical_scale * double(1U << (key.lod - 1U)))
                                    : float(geometry.cell_size) * float(1U << (key.lod - 1U));
    const float half_width = 0.5F * delta * float(cells);
    const auto tile_key = manager->sources()[source].key;
    entry->local = {transformed ? 0.0F : -half_width,
                    transformed ? 0.0F : -half_width,
                    delta,
                    cells,
                    0,
                    base,
                    key.lod,
                    tile_key.row,
                    tile_key.column,
                    uint32_t(bool(source_metadata.valid_cells))};
    auto *tile_transforms = static_cast<BvhAffinePatch *>(entry->transforms.contents);
    if (transformed) {
      for (size_t patch = 0; patch < source_metadata.transform_patches.size(); ++patch)
        tile_transforms[patch] =
            affine_patch(source_metadata.transform_patches[patch], source, logical_scale);
      const Coord centre = source_metadata.transform_patches.front().transform.apply(
          0.5 * geometry.cell_count,
          0.5 * geometry.cell_count
      );
      entry->anchor_x = float(centre.x);
      entry->anchor_y = float(centre.y);
    } else {
      tile_transforms[0] = {entry->local.x_min,
                            entry->local.y_min,
                            delta,
                            0,
                            0,
                            delta,
                            1 / delta,
                            0,
                            0,
                            1 / delta,
                            0,
                            0,
                            cells,
                            cells,
                            source,
                            0};
    }
    std::memcpy(entry->blocks.contents, block_metadata.data(), entry->blocks.length);
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
    [encoder setBuffer:entry->transforms offset:0 atIndex:5];
    dispatch_linear(encoder, bounds_pipeline, count);
    [encoder endEncoding];
    // Tracked resources and encoder ordering make the generated bounds
    // available to the build without a separate submission and CPU wait.
    entry->acceleration = build(primitive_descriptor(count, entry->bounds), true, command);
    entry->bytes = base_bytes + entry->acceleration.size;
    stats.resident_bytes += entry->bytes;
    ++stats.builds;
    Entry *result = entry.get();
    invalidate_scene(SceneInvalidation::Admission);
    cache.emplace(key, std::move(entry));
    timer.add_work("BVH tile loading/building", std::chrono::steady_clock::now() - started);
    return result;
  }

  Hierarchy make_hierarchy(const std::vector<Entry *> &batch, Timer &timer) {
    trace_activity::Scope activity("BVH instance hierarchy");
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
    const Coord render_observer =
        manager->catalogue().render_coordinate({observer.easting, observer.northing});
    for (size_t i = 0; i < batch.size(); ++i) {
      Entry &entry = *batch[i];
      BvhTile tile = entry.local;
      if (!entry.transformed) {
        tile.x_min = tile_metadata[entry.key.source_index].x_min;
        tile.y_min = tile_metadata[entry.key.source_index].y_min;
      }
      resources[i] = {entry.vertices.gpuAddress,
                      entry.blocks.gpuAddress,
                      entry.bounds.gpuAddress,
                      entry.transforms.gpuAddress,
                      tile,
                      entry.key.source_index,
                      entry.vertical_offset,
                      entry.transformed ? 1U : 0U,
                      0U};
      auto &instance = transforms[i];
      instance = {};
      if (entry.transformed) {
        const double dx = double(entry.anchor_x) - render_observer.x;
        const double dy = double(entry.anchor_y) - render_observer.y;
        const double k = current.trace.curvature_coefficient;
        instance.transformationMatrix.columns[0] = {1, 0, float(-2 * k * dx)};
        instance.transformationMatrix.columns[1] = {0, 1, float(-2 * k * dy)};
        instance.transformationMatrix.columns[2] = {0, 0, 1};
        instance.transformationMatrix.columns[3] = {
            float(-render_observer.x),
            float(-render_observer.y),
            float(
                entry.vertical_offset + 2 * k * (dx * entry.anchor_x + dy * entry.anchor_y) -
                k * (dx * dx + dy * dy)
            )};
        instance.mask = 0xff;
        instance.accelerationStructureIndex = checked_count(i);
        [structures addObject:entry.acceleration];
        continue;
      }
      const double ax = double(tile.x_min) - entry.local.x_min;
      const double ay = double(tile.y_min) - entry.local.y_min;
      const double k = current.trace.curvature_coefficient;
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
    trace_activity::Scope activity("BVH detail encode/traverse");
    std::memcpy(parameters.contents, &current, sizeof(current));
    auto &p = pipeline(
        shadows      ? TraceMode::Shadows
        : scene_mode ? TraceMode::Scene
                     : TraceMode::Batch
    );
    [p.table setBuffer:hierarchy.chunks offset:0 atIndex:0];
    [p.table setBuffer:parameters offset:0 atIndex:1];
    [p.table setBuffer:coverage_polygons offset:0 atIndex:2];
    [p.table setBuffer:coverage_vertices offset:0 atIndex:3];
    const bool submit = command == nil;
    if (submit)
      command = [gpu.command_queue() commandBuffer];
    if (scene_mode && !shadows && !current.scene_resume) {
      auto clear = [command blitCommandEncoder];
      if (clear == nil)
        throw std::runtime_error("Could not clear terrain usage");
      [clear fillBuffer:used_sources range:NSMakeRange(0, used_sources.length) value:0];
      [clear endEncoding];
    }
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
    [encoder setBuffer:hierarchy.chunks offset:0 atIndex:9];
    [encoder setBuffer:used_sources offset:0 atIndex:10];
    [encoder setBuffer:rays offset:0 atIndex:11];
    [encoder setBuffer:work offset:0 atIndex:12];
    if (scene_mode && !shadows) {
      [encoder setBuffer:scene_pending_rays[1U - scene_pending_output] offset:0 atIndex:12];
      [encoder setBuffer:scene_pending_rays[scene_pending_output] offset:0 atIndex:18];
    }
    if (scene_mode) {
      [p.missing_table setBuffer:current.transformed_catalogue ? candidate_patches : tiles
                          offset:0
                         atIndex:0];
      const id<MTLBuffer> excluded = shadows && shadow_tested ? shadow_tested : scene_resident;
      [p.missing_table setBuffer:excluded offset:0 atIndex:1];
      [encoder setAccelerationStructure:candidate_catalogue atBufferIndex:13];
      [encoder setIntersectionFunctionTable:p.missing_table atBufferIndex:14];
      [encoder setBuffer:shadows ? shadow_missing_count : scene_missing_count offset:0 atIndex:15];
      [encoder useResource:candidate_catalogue usage:MTLResourceUsageRead];
      [encoder useResource:excluded usage:MTLResourceUsageRead];
      [encoder useResource:p.missing_table usage:MTLResourceUsageRead];
      if (!shadows) {
        [encoder setBuffer:candidate_patches offset:0 atIndex:16];
        [encoder setBuffer:scene_requested_sources offset:0 atIndex:17];
      }
    }
    [p.coverage_table setBuffer:current.transformed_catalogue ? coverage_polygons : tiles
                         offset:0
                        atIndex:0];
    [p.coverage_table setBuffer:current.transformed_catalogue ? coverage_vertices : dummy
                         offset:0
                        atIndex:1];
    [p.coverage_table setBuffer:dummy offset:0 atIndex:2];
    [encoder setAccelerationStructure:catalogue atBufferIndex:20];
    [encoder setIntersectionFunctionTable:p.coverage_table atBufferIndex:21];
    [encoder setBuffer:coverage_polygons offset:0 atIndex:22];
    [encoder setBuffer:coverage_vertices offset:0 atIndex:23];
    [encoder useResource:catalogue usage:MTLResourceUsageRead];
    [encoder useResource:p.coverage_table usage:MTLResourceUsageRead];
    [encoder useResource:coverage_polygons usage:MTLResourceUsageRead];
    [encoder useResource:coverage_vertices usage:MTLResourceUsageRead];
    if (shadows) {
      [encoder setBytes:sun length:sizeof(sun) atIndex:16];
      [encoder setBuffer:scene_shadows offset:0 atIndex:17];
      [encoder setBuffer:shadow_requested_sources offset:0 atIndex:18];
      [encoder setBuffer:candidate_patches offset:0 atIndex:19];
    }
    [encoder useResource:coverage_polygons usage:MTLResourceUsageRead];
    [encoder useResource:coverage_vertices usage:MTLResourceUsageRead];
    [encoder useResource:hierarchy.chunks usage:MTLResourceUsageRead];
    [encoder useResource:parameters usage:MTLResourceUsageRead];
    [encoder useResource:hierarchy.acceleration usage:MTLResourceUsageRead];
    [encoder useResource:p.table usage:MTLResourceUsageRead];
    for (Entry *entry : batch) {
      [encoder useResource:entry->vertices usage:MTLResourceUsageRead];
      [encoder useResource:entry->blocks usage:MTLResourceUsageRead];
      [encoder useResource:entry->bounds usage:MTLResourceUsageRead];
      [encoder useResource:entry->transforms usage:MTLResourceUsageRead];
      [encoder useResource:entry->acceleration usage:MTLResourceUsageRead];
    }
    if (scene_mode && (shadows || !current.scene_resume))
      dispatch_image(encoder, p.state, current.trace);
    else
      dispatch_linear(encoder, p.state, current.work_count);
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
    [p.coverage_table setBuffer:nil offset:0 atIndex:0];
    [p.coverage_table setBuffer:nil offset:0 atIndex:1];
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
    trace_activity::Scope activity("BVH resident scene preparation");
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

  bool trace_scene(Timer &timer, bool resume) {
    if (!prepare_scene(timer))
      return false;
    current.scene_resume = resume;
    current.work_count = resume ? *static_cast<const uint32_t *>(scene_missing_count.contents)
                                : current.trace.ray_count;
    if (resume) {
      scene_pending_output = 1U - scene_pending_output;
      ++stats.scene_repair_passes;
      stats.scene_repair_rays += current.work_count;
    } else {
      scene_pending_output = 0U;
    }
    *static_cast<uint32_t *>(scene_missing_count.contents) = 0U;
    std::memset(scene_requested_sources.contents, 0, scene_requested_sources.length);
    scene_requests_ready = false;
    trace_hierarchy(scene_entries, scene, true, timer);
    const uint32_t missing = *static_cast<const uint32_t *>(scene_missing_count.contents);
    scene_requests_ready = true;
    stats.scene_fallback_rays += missing;
    (void)used_entries();
    return missing == 0U;
  }

  void select(const std::vector<uint32_t> &indices, Timer &timer) {
    trace_activity::Scope activity("BVH streaming tile selection");
    current.work_count = checked_count(indices.size());
    std::memcpy(work.contents, indices.data(), indices.size() * sizeof(uint32_t));
    std::memcpy(parameters.contents, &current, sizeof(current));
    Pipeline &selector = current.transformed_catalogue ? patch_selector : tile_selector;
    id<MTLBuffer> selection_metadata = current.transformed_catalogue ? coverage_polygons : tiles;
    [selector.table setBuffer:selection_metadata offset:0 atIndex:0];
    [selector.table setBuffer:current.transformed_catalogue ? coverage_vertices : dummy
                       offset:0
                      atIndex:1];
    [selector.table setBuffer:dummy offset:0 atIndex:2];
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
    [encoder setBuffer:selection_metadata offset:0 atIndex:5];
    if (current.transformed_catalogue)
      [encoder setBuffer:coverage_vertices offset:0 atIndex:6];
    [encoder setBuffer:work offset:0 atIndex:7];
    [encoder useResource:selection_metadata usage:MTLResourceUsageRead];
    if (current.transformed_catalogue)
      [encoder useResource:coverage_vertices usage:MTLResourceUsageRead];
    [encoder useResource:catalogue usage:MTLResourceUsageRead];
    [encoder useResource:selector.table usage:MTLResourceUsageRead];
    dispatch_linear(encoder, selector.state, current.work_count);
    [encoder endEncoding];
    const double gpu_ms = complete(command);
    stats.selection_gpu_ms += gpu_ms;
    ++stats.selection_passes;
    stats.selection_rays += indices.size();
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
MetalBvhStatistics MetalBvhTrace::statistics() const {
  auto result = state_->stats;
  result.cached_tiles = state_->cache.size();
  result.scene_tiles = state_->scene_entries.size();
  return result;
}

bool MetalBvhTrace::prepare_scene(Timer &timer) { return state_->prepare_scene(timer); }

bool MetalBvhTrace::encode_scene(id<MTLCommandBuffer> command, Timer &timer) {
  State &state = *state_;
  state.scene_requests_ready = false;
  state.shadow_requests_ready = false;
  if (command == nil || !state.prepare_scene(timer))
    return false;
  *static_cast<uint32_t *>(state.scene_missing_count.contents) = 0U;
  auto clear = [command blitCommandEncoder];
  if (clear == nil)
    throw std::runtime_error("Could not clear primary terrain requests");
  [clear fillBuffer:state.scene_requested_sources
              range:NSMakeRange(0, state.scene_requested_sources.length)
              value:0];
  [clear endEncoding];
  state.current.work_count = state.current.trace.ray_count;
  state.current.scene_resume = 0U;
  state.scene_pending_output = 0U;
  state.trace_hierarchy(state.scene_entries, state.scene, true, timer, command);
  return true;
}

bool MetalBvhTrace::scene_complete() {
  State &state = *state_;
  state.scene_requests_ready = true;
  const uint32_t missing = *static_cast<const uint32_t *>(state.scene_missing_count.contents);
  state.stats.scene_fallback_rays += missing;
  ++state.stats.scene_passes;
  ++state.stats.trace_passes;
  ++state.stats.submissions;
  (void)state.used_entries();
  return missing == 0U;
}

bool MetalBvhTrace::encode_shadows(id<MTLCommandBuffer> command, double azimuth, double elevation) {
  State &state = *state_;
  state.shadow_requests_ready = false;
  state.shadow_azimuth = azimuth;
  state.shadow_elevation = elevation;
  Timer timer("Encode scene shadows");
  if (command == nil || !state.prepare_scene(timer))
    return false;
  const uint32_t count = state.current.trace.ray_count;
  // A cold primary trace leaves work_count at its final streaming batch size.
  // Shadows always cover the entire image, including sky visibility values.
  state.current.work_count = count;
  if (state.scene_shadows == nil || state.scene_shadows.length < count)
    state.scene_shadows = buffer(state.gpu.device(), count, 1U, @"scene shadow visibility");
  if (state.shadow_requested_sources == nil)
    state.shadow_requested_sources = buffer(
        state.gpu.device(),
        state.current.source_count,
        sizeof(uint32_t),
        @"shadow terrain requests"
    );
  const auto basis =
      state.manager->catalogue().render_basis({state.observer.easting, state.observer.northing});
  const double east = std::sin(azimuth), north = std::cos(azimuth);
  state.sun[0] = static_cast<float>(basis[0].x * east + basis[1].x * north);
  state.sun[1] = static_cast<float>(basis[0].y * east + basis[1].y * north);
  state.sun[2] = static_cast<float>(std::tan(elevation));
  state.sun[3] = elevation <= 0.0 ? -1.0F : std::cos(elevation) < 1e-6 ? 1.0F : 0.0F;
  *static_cast<uint32_t *>(state.shadow_missing_count.contents) = 0U;
  // Clear on the GPU: the producer may already contain work using this scene.
  auto clear = [command blitCommandEncoder];
  if (clear == nil)
    throw std::runtime_error("Could not clear shadow terrain requests");
  [clear fillBuffer:state.shadow_requested_sources
              range:NSMakeRange(0, state.shadow_requested_sources.length)
              value:0];
  [clear endEncoding];
  state.trace_hierarchy(state.scene_entries, state.scene, true, timer, command, true);
  return true;
}

bool MetalBvhTrace::shadows_complete() {
  state_->shadow_requests_ready = true;
  (void)state_->used_entries();
  return *static_cast<const uint32_t *>(state_->shadow_missing_count.contents) == 0U;
}

id<MTLBuffer> MetalBvhTrace::shadow_visibility() const { return state_->scene_shadows; }

bool MetalBvhTrace::trace_shadows(double azimuth, double elevation, Timer &timer) {
  State &state = *state_;
  trace_activity::Scope activity("BVH shadow terrain repair");
  std::vector<State::Entry *> pinned = state.used_entries();
  for (uint32_t round = 0; round <= state.current.source_count; ++round) {
    // Reuse the completed producer's requests when its primary outputs and
    // sun are unchanged. Primary repair/relocation invalidate this snapshot.
    if (state.shadow_requests_ready && state.shadow_azimuth == azimuth &&
        state.shadow_elevation == elevation) {
      if (shadows_complete())
        return true;
    } else
      @autoreleasepool {
        auto command = [state.gpu.command_queue() commandBuffer];
        if (!encode_shadows(command, azimuth, elevation))
          break;
        const double gpu_ms = complete(command);
        state.stats.shadow_gpu_ms += gpu_ms;
        ++state.stats.shadow_passes;
        ++state.stats.submissions;
        timer.add_work("GPU BVH shadow repair", gpu_ms);
        if (shadows_complete())
          return true;
      }
    const auto *requested = static_cast<const uint32_t *>(state.shadow_requested_sources.contents);
    bool loaded = false;
    for (uint32_t source = 0; source < state.current.source_count; ++source) {
      if (requested[source] == 0)
        continue;
      @autoreleasepool {
        // Each request names a source absent from the preceding scene. Loading
        // can invalidate that scene, but never modifies the request buffer.
        State::Entry *entry = state.acquire(source, pinned, timer, true);
        if (entry == nullptr) {
          ++state.stats.shadow_cache_fallbacks;
          return state.current.transformed_catalogue ? stream_shadows(azimuth, elevation, timer)
                                                     : false;
        }
        pinned.push_back(entry);
        ++state.stats.shadow_tiles_built;
        loaded = true;
      }
    }
    if (!loaded)
      break;
  }
  ++state.stats.shadow_cache_fallbacks;
  return state.current.transformed_catalogue ? stream_shadows(azimuth, elevation, timer) : false;
}

bool MetalBvhTrace::stream_shadows(double azimuth, double elevation, Timer &timer) {
  State &state = *state_;
  // Each pass tests a cache-sized set of sources. Keep proven occlusion and
  // remember tested sources even after eviction; primary hit outputs stay valid.
  state.shadow_tested = buffer(
      state.gpu.device(),
      state.current.source_count,
      sizeof(uint32_t),
      @"tested shadow sources"
  );
  struct ClearTested {
    State &state;
    ~ClearTested() { state.shadow_tested = nil; }
  } clear{state};
  auto *tested = static_cast<uint32_t *>(state.shadow_tested.contents);
  std::memset(tested, 0, state.shadow_tested.length);
  std::vector<uint8_t> visibility(state.current.trace.ray_count, 1U);
  for (uint32_t round = 0; round <= state.current.source_count; ++round) {
    @autoreleasepool {
      if (!state.prepare_scene(timer))
        throw std::runtime_error("No resident BVH is available for shadow streaming");
      for (const auto *entry : state.scene_entries)
        tested[entry->key.source_index] = 1U;
      auto command = [state.gpu.command_queue() commandBuffer];
      if (!encode_shadows(command, azimuth, elevation))
        throw std::runtime_error("Could not encode streamed BVH shadows");
      const double gpu_ms = complete(command);
      state.stats.shadow_gpu_ms += gpu_ms;
      ++state.stats.shadow_passes;
      ++state.stats.submissions;
      timer.add_work("GPU BVH shadow streaming", gpu_ms);
      const auto *current_visibility = static_cast<const uint8_t *>(state.scene_shadows.contents);
      for (size_t index = 0; index < visibility.size(); ++index)
        visibility[index] &= current_visibility[index];
      if (shadows_complete()) {
        std::memcpy(state.scene_shadows.contents, visibility.data(), visibility.size());
        return true;
      }
      const auto *requested =
          static_cast<const uint32_t *>(state.shadow_requested_sources.contents);
      std::vector<State::Entry *> batch;
      for (uint32_t source = 0; source < state.current.source_count; ++source) {
        if (!requested[source] || tested[source])
          continue;
        auto *entry = state.acquire(source, batch, timer, true);
        if (!entry) {
          if (batch.empty())
            throw std::runtime_error(
                "BVH cache cannot hold one shadow terrain tile; increase --bvh-cache-mib"
            );
          break;
        }
        batch.push_back(entry);
        ++state.stats.shadow_tiles_built;
      }
      if (batch.empty())
        throw std::runtime_error("BVH shadow streaming did not advance to an untested source");
    }
  }
  throw std::runtime_error("BVH shadow streaming exceeded the source count");
}

void MetalBvhTrace::prepare(
    TileManager &tiles,
    ObserverLocation observer,
    const RaytraceParameters &parameters,
    Timer &timer
) {
  State &state = *state_;
  state.shadow_requests_ready = false;
  state.manager = &tiles;
  state.observer = observer;
  state.current.trace = parameters;
  state.current.quantized = tiles.traces_quantized();
  state.current.observer_source = tiles.observer_source_index();
  state.current.source_count = checked_count(tiles.sources().size());
  state.current.hash_capacity = state.gpu.catalogue_hash_capacity();
  state.current.transformed_catalogue =
      !tiles.sources().empty() && !tiles.sources().front().transform_patches.empty();
  if (state.scene_requested_sources == nil ||
      state.scene_requested_sources.length <
          uint64_t(state.current.source_count) * sizeof(uint32_t)) {
    state.scene_requested_sources = buffer(
        state.gpu.device(),
        state.current.source_count,
        sizeof(uint32_t),
        @"primary terrain requests"
    );
    state.used_sources = buffer(
        state.gpu.device(),
        state.current.source_count,
        sizeof(uint32_t),
        @"used terrain sources"
    );
    std::memset(state.used_sources.contents, 0, state.used_sources.length);
  }
  uint32_t changed_lods = state.selected_lods.size() == tiles.sources().size()
                              ? 0U
                              : checked_count(tiles.sources().size());
  if (changed_lods == 0U) {
    for (uint32_t source = 0U; source < state.selected_lods.size(); ++source)
      changed_lods += state.selected_lods[source] != tiles.lod_for_source(source);
  }
  if (changed_lods != 0U) {
    state.selected_lods.resize(tiles.sources().size());
    for (uint32_t source = 0U; source < state.selected_lods.size(); ++source)
      state.selected_lods[source] = tiles.lod_for_source(source);
    // Detailed scenes contain one chosen LOD per source. Camera changes must
    // refresh that scene, while the LOD-independent catalogue remains reusable.
    ++state.stats.lod_plan_changes;
    state.stats.lod_sources_changed += changed_lods;
    state.invalidate_scene(State::SceneInvalidation::Lod);
  }
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
    for (auto &pending : state.scene_pending_rays)
      pending = buffer(
          state.gpu.device(),
          parameters.ray_count,
          sizeof(uint32_t),
          @"unresolved scene rays"
      );
    state.ray_capacity = parameters.ray_count;
  }
}

void MetalBvhTrace::trace(const RaytraceParameters &parameters, Timer &timer, bool resume) {
  State &state = *state_;
  state.current.trace = parameters;
  // Only an explicitly resumed producer may reuse requests and completed rays.
  // Ordinary synchronous traces own a fresh ray field, even if the scene stays.
  if (!resume)
    state.scene_requests_ready = false;
  @autoreleasepool {
    bool has_scene = state.prepare_scene(timer);
    if (!has_scene) {
      const std::vector<State::Entry *> none;
      (void)state.acquire(state.current.observer_source, none, timer);
      has_scene = state.prepare_scene(timer);
    }
    if (has_scene) {
      // A scene miss identifies catalogue geometry absent from the detailed
      // hierarchy. Admit those sources directly and retry the scene instead
      // of walking every already-resident tile again in one full-image pass
      // per distance shell. The bounded streaming path below remains the
      // exact fallback when the complete working set cannot fit the cache.
      std::vector<State::Entry *> repair_pins;
      for (uint32_t repair = 0; repair <= state.current.source_count; ++repair) {
        const uint32_t missing =
            state.scene_requests_ready
                ? *static_cast<const uint32_t *>(state.scene_missing_count.contents)
                : std::numeric_limits<uint32_t>::max();
        if ((state.scene_requests_ready && missing == 0U) ||
            (!state.scene_requests_ready && state.trace_scene(timer, resume)))
          return;
        resume = true;
        // Include the provisional winners of every completed repair pass:
        // their hit bounds are reused by the next pass, so they must survive.
        for (auto *entry : state.used_entries()) {
          if (std::find(repair_pins.begin(), repair_pins.end(), entry) == repair_pins.end())
            repair_pins.push_back(entry);
        }
        const auto *requested =
            static_cast<const uint32_t *>(state.scene_requested_sources.contents);
        bool loaded = false;
        bool fits = true;
        for (uint32_t source = 0; source < state.current.source_count; ++source) {
          if (requested[source] == 0U)
            continue;
          const TileVariant key{source, state.manager->lod_for_source(source)};
          if (state.cache.contains(key))
            continue;
          State::Entry *entry = state.acquire(source, repair_pins, timer, true);
          if (entry == nullptr) {
            fits = false;
            break;
          }
          repair_pins.push_back(entry);
          loaded = true;
        }
        if (!fits || !loaded)
          break;
        state.scene_requests_ready = false;
        (void)state.prepare_scene(timer);
      }
    }
    if (!has_scene) {
      // A cold streaming pass needs fresh continuations. A partially resolved
      // scene already marks its complete rays and leaves only seam/missing
      // rays active for the streaming fallback below.
      auto command = [state.gpu.command_queue() commandBuffer];
      if (command == nil)
        throw std::runtime_error("Could not create BVH initialization command");
      state.gpu.encode_clear_outputs(command);
      auto encoder = [command computeCommandEncoder];
      if (encoder == nil)
        throw std::runtime_error("Could not encode BVH initialization");
      [encoder setComputePipelineState:state.initialize_pipeline];
      [encoder setBuffer:state.rays offset:0 atIndex:0];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
      dispatch_image(encoder, state.initialize_pipeline, parameters);
      [encoder endEncoding];
      state.stats.trace_gpu_ms += complete(command);
      ++state.stats.submissions;
    }
  }
  const uint32_t traversal_limit = state.current.transformed_catalogue
                                       ? state.current.coverage_step_limit
                                       : state.current.source_count;
  // Keep the sparse unresolved tail compact through selection and CPU grouping,
  // including when eviction forces us out of the resident-scene repair path.
  const auto *rays = static_cast<const BvhRayState *>(state.rays.contents);
  std::vector<uint32_t> active;
  for (uint32_t i = 0; i < parameters.ray_count; ++i) {
    if (!rays[i].done)
      active.push_back(i);
  }
  for (uint32_t round = 0; round <= traversal_limit; ++round) {
    std::erase_if(active, [&](uint32_t i) { return rays[i].done != 0U; });
    if (active.empty())
      return;
    @autoreleasepool {
      state.select(active, timer);
    }
    const auto grouping_started = std::chrono::steady_clock::now();
    double nearest_shell = std::numeric_limits<double>::infinity();
    uint32_t active_rays = 0U;
    uint32_t failed_ray = parameters.ray_count;
    for (uint32_t i : active) {
      if (rays[i].done)
        continue;
      ++active_rays;
      if (failed_ray == parameters.ray_count &&
          (rays[i].source >= state.current.source_count || !(rays[i].exit > rays[i].progress)))
        failed_ray = i;
      if (rays[i].source >= state.current.source_count)
        continue;
      nearest_shell = std::min(nearest_shell, state.tile_shells[rays[i].source]);
    }
    if (failed_ray != parameters.ray_count) {
      const BvhRayState failed = rays[failed_ray];
      const auto *directions =
          static_cast<const RayDirection *>(state.gpu.ray_directions().contents);
      const RayDirection direction = directions[failed_ray];
      char diagnostic[1024];
      if (failed.source < state.current.source_count) {
        const TerrainSource &source = state.manager->sources()[failed.source];
        double maximum_residual = 0.0;
        for (const TerrainTransformPatch &patch : source.transform_patches)
          maximum_residual = std::max(maximum_residual, patch.transform.maximum_residual_metres);
        std::snprintf(
            diagnostic,
            sizeof(diagnostic),
            "BVH tile selection failed to advance ray %u/%u in round %u: "
            "progress=%.9g exit=%.9g source=%u dataset=%u tile=(%lld,%lld) LOD=%u "
            "patches=%zu max-residual=%.6g m direction=(%.9g,%.9g) slope=%.9g; "
            "active=%u catalogue-sources=%u transformed=%u",
            failed_ray,
            parameters.ray_count,
            round,
            failed.progress,
            failed.exit,
            failed.source,
            source.dataset_index,
            static_cast<long long>(source.key.row),
            static_cast<long long>(source.key.column),
            state.manager->lod_for_source(failed.source),
            source.transform_patches.size(),
            maximum_residual,
            direction.x,
            direction.y,
            direction.slope,
            active_rays,
            state.current.source_count,
            state.current.transformed_catalogue
        );
      } else {
        std::snprintf(
            diagnostic,
            sizeof(diagnostic),
            "BVH tile selection returned invalid source for ray %u/%u in round %u: "
            "progress=%.9g exit=%.9g source=%u direction=(%.9g,%.9g) slope=%.9g; "
            "active=%u catalogue-sources=%u transformed=%u",
            failed_ray,
            parameters.ray_count,
            round,
            failed.progress,
            failed.exit,
            failed.source,
            direction.x,
            direction.y,
            direction.slope,
            active_rays,
            state.current.source_count,
            state.current.transformed_catalogue
        );
      }
      throw std::runtime_error(diagnostic);
    }
    ++state.stats.streaming_rounds;
    // Finish one spatial shell before the next so all rays that can reach a
    // source arrive before it is processed. Native grids use Manhattan shells;
    // mixed-CRS catalogues use radial bands in their common render frame.
    std::map<uint32_t, std::vector<uint32_t>> groups;
    for (uint32_t i : active) {
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
    state.stats.streaming_groups += groups.size();
    for (const auto &[source, indices] : groups) {
      (void)source;
      state.stats.streaming_rays += indices.size();
    }
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
