#include "terrain_tile_bvh.h"
#include "trace_activity.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
using namespace bvh_resources;
struct TerrainTileBvh::State {
  GpuRaytraceResources &gpu;
  ObserverLocation observer = {};
  float curvature = 0;
  std::vector<BvhTile> tile_metadata;
  id<MTLBuffer> tiles, patches, catalogue_bounds, candidate_bounds;
  id<MTLAccelerationStructure> catalogue, candidates;
  uint64_t generation = 0;
  double build_ms = 0;
  explicit State(GpuRaytraceResources &resources) : gpu(resources) {}

  void rebuild(TileManager &manager, const RaytraceParameters &parameters) {
    trace_activity::Scope activity("BVH catalogue rebuild");
    const auto &grid = manager.catalogue().grid();
    const auto &sources = manager.sources();
    const auto &geometry = manager.origin_geometry();
    tile_metadata.resize(sources.size());
    std::vector<BvhAffinePatch> patch_metadata;
    std::vector<BvhBounds> boxes, candidate_boxes;
    const Coord render_observer =
        manager.catalogue().render_coordinate({observer.easting, observer.northing});
    const double k = parameters.curvature_coefficient;
    for (size_t i = 0; i < sources.size(); ++i) {
      const auto &source = sources[i];
      const bool transformed = !source.transform_patches.empty();
      const float x =
          transformed
              ? static_cast<float>(
                    source.transform_patches.front().transform.bounds[0] - render_observer.x
                )
              : static_cast<float>(
                    grid.origin_x + double(source.key.column) * grid.width - observer.easting
                );
      const float y =
          transformed
              ? static_cast<float>(
                    source.transform_patches.front().transform.bounds[1] - render_observer.y
                )
              : static_cast<float>(
                    grid.origin_y - double(source.key.row + 1) * grid.width - observer.northing
                );
      const float width =
          transformed ? static_cast<float>(source.effective_cell_size_metres * geometry.cell_count)
                      : static_cast<float>(grid.width);
      tile_metadata[i] = {x,
                          y,
                          transformed ? static_cast<float>(source.effective_cell_size_metres)
                                      : static_cast<float>(geometry.cell_size),
                          geometry.cell_count,
                          0,
                          0,
                          1,
                          source.key.row,
                          source.key.column};
      const auto append_patch = [&](const TerrainTileTransform &transform,
                                    uint32_t minimum_column,
                                    uint32_t minimum_row,
                                    uint32_t maximum_column,
                                    uint32_t maximum_row) {
        const double ox = transform.origin.x - render_observer.x;
        const double oy = transform.origin.y - render_observer.y;
        const double determinant = transform.determinant();
        patch_metadata.push_back(
            {float(ox),
             float(oy),
             float(transform.column_step.x),
             float(transform.column_step.y),
             float(transform.row_step.x),
             float(transform.row_step.y),
             float(transform.row_step.y / determinant),
             float(-transform.row_step.x / determinant),
             float(-transform.column_step.y / determinant),
             float(transform.column_step.x / determinant),
             minimum_column,
             minimum_row,
             maximum_column,
             maximum_row,
             checked_count(i),
             float(transform.maximum_residual_metres)}
        );
        const double bx0 = transform.bounds[0] - render_observer.x;
        const double by0 = transform.bounds[1] - render_observer.y;
        const double bx1 = transform.bounds[2] - render_observer.x;
        const double by1 = transform.bounds[3] - render_observer.y;
        const double near_x = std::clamp(0.0, bx0, bx1);
        const double near_y = std::clamp(0.0, by0, by1);
        const double far_x = std::max(std::abs(bx0), std::abs(bx1));
        const double far_y = std::max(std::abs(by0), std::abs(by1));
        const double far_lift = k * (far_x * far_x + far_y * far_y);
        const double near_lift = k * (near_x * near_x + near_y * near_y);
        const double diagonal = std::hypot(bx1 - bx0, by1 - by0);
        const double guard =
            1.01 + 64 * std::numeric_limits<float>::epsilon() * (far_lift + 100000.0) +
            k * diagonal * (2 * std::hypot(far_x, far_y) + diagonal) + 2.01e-4 * far_lift;
        const BvhBounds candidate = {float(bx0),
                                     float(by0),
                                     source.minimum_elevation.has_value()
                                         ? float(*source.minimum_elevation - far_lift - guard)
                                         : -1e30F,
                                     float(bx1),
                                     float(by1),
                                     source.maximum_elevation.has_value()
                                         ? float(*source.maximum_elevation - near_lift + guard)
                                         : 1e30F};
        candidate_boxes.push_back(candidate);
        // Transformed tile selection also proves continuous source coverage.
        // Keep that hierarchy independent of elevation so safely culled low
        // terrain is never mistaken for a hole between datasets.
        boxes.push_back(
            transformed ? BvhBounds{float(bx0), float(by0), -1e30F, float(bx1), float(by1), 1e30F}
                        : candidate
        );
      };
      if (transformed) {
        for (const TerrainTransformPatch &patch : source.transform_patches) {
          append_patch(
              patch.transform,
              patch.minimum_column,
              patch.minimum_row,
              patch.minimum_column + patch.cell_width,
              patch.minimum_row + patch.cell_height
          );
        }
      } else {
        TerrainTileTransform transform = {
            {double(x) + render_observer.x, double(y) + render_observer.y},
            {geometry.cell_size, 0.0},
            {0.0, geometry.cell_size},
            0.0,
            {double(x) + render_observer.x,
             double(y) + render_observer.y,
             double(x + width) + render_observer.x,
             double(y + width) + render_observer.y}};
        append_patch(transform, 0U, 0U, geometry.cell_count, geometry.cell_count);
      }
    }
    tiles = buffer(gpu.device(), sources.size(), sizeof(BvhTile), @"catalogue tiles");
    patches =
        buffer(gpu.device(), patch_metadata.size(), sizeof(BvhAffinePatch), @"catalogue patches");
    catalogue_bounds = buffer(gpu.device(), boxes.size(), sizeof(BvhBounds), @"catalogue bounds");
    candidate_bounds =
        buffer(gpu.device(), candidate_boxes.size(), sizeof(BvhBounds), @"candidate bounds");
    std::memcpy(tiles.contents, tile_metadata.data(), tiles.length);
    std::memcpy(patches.contents, patch_metadata.data(), patches.length);
    std::memcpy(catalogue_bounds.contents, boxes.data(), catalogue_bounds.length);
    std::memcpy(candidate_bounds.contents, candidate_boxes.data(), candidate_bounds.length);

    auto descriptor = primitive_descriptor(checked_count(patch_metadata.size()), catalogue_bounds);
    auto candidate_descriptor =
        primitive_descriptor(checked_count(patch_metadata.size()), candidate_bounds);
    const auto sizes = [gpu.device() accelerationStructureSizesWithDescriptor:descriptor];
    const auto candidate_sizes =
        [gpu.device() accelerationStructureSizesWithDescriptor:candidate_descriptor];
    auto next = [gpu.device() newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    auto next_candidates =
        [gpu.device() newAccelerationStructureWithSize:candidate_sizes.accelerationStructureSize];
    if (next == nil || next_candidates == nil)
      throw std::runtime_error("Could not allocate catalogue BVH");
    next.label = @"Shared terrain catalogue BVH";
    next_candidates.label = @"Shared terrain candidate BVH";
    auto scratch = buffer(
        gpu.device(),
        sizes.buildScratchBufferSize,
        1,
        @"catalogue build scratch",
        MTLResourceStorageModePrivate
    );
    auto command = [gpu.command_queue() commandBuffer];
    command.label = @"Shared terrain catalogue build";
    auto encoder = [command accelerationStructureCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("Could not encode catalogue BVH build");
    [encoder buildAccelerationStructure:next
                             descriptor:descriptor
                          scratchBuffer:scratch
                    scratchBufferOffset:0];
    auto candidate_scratch = buffer(
        gpu.device(),
        candidate_sizes.buildScratchBufferSize,
        1,
        @"candidate build scratch",
        MTLResourceStorageModePrivate
    );
    [encoder buildAccelerationStructure:next_candidates
                             descriptor:candidate_descriptor
                          scratchBuffer:candidate_scratch
                    scratchBufferOffset:0];
    [encoder endEncoding];
    build_ms = complete(command);
    catalogue = next;
    candidates = next_candidates;
    ++generation;
  }
};

TerrainTileBvh::TerrainTileBvh(GpuRaytraceResources &gpu) : state_(std::make_unique<State>(gpu)) {}
TerrainTileBvh::~TerrainTileBvh() = default;
bool TerrainTileBvh::prepare(
    TileManager &manager,
    ObserverLocation observer,
    const RaytraceParameters &parameters
) {
  State &state = *state_;
  if (state.catalogue != nil && state.observer.easting == observer.easting &&
      state.observer.northing == observer.northing &&
      state.curvature == parameters.curvature_coefficient)
    return false;
  state.observer = observer;
  state.curvature = parameters.curvature_coefficient;
  state.catalogue = nil; // Failed construction must be retried on the next frame.
  state.rebuild(manager, parameters);
  return true;
}
id<MTLAccelerationStructure> TerrainTileBvh::acceleration() const { return state_->catalogue; }
id<MTLAccelerationStructure> TerrainTileBvh::candidate_acceleration() const {
  return state_->candidates;
}
id<MTLBuffer> TerrainTileBvh::tiles() const { return state_->tiles; }
id<MTLBuffer> TerrainTileBvh::patches() const { return state_->patches; }
std::span<const BvhTile> TerrainTileBvh::metadata() const { return state_->tile_metadata; }
uint64_t TerrainTileBvh::bytes() const {
  return state_->tiles.length + state_->patches.length + state_->catalogue_bounds.length +
         state_->candidate_bounds.length + state_->catalogue.size + state_->candidates.size;
}
uint64_t TerrainTileBvh::generation() const { return state_->generation; }
double TerrainTileBvh::build_milliseconds() const { return state_->build_ms; }
} // namespace panorama
