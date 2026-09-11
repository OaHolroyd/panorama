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
  id<MTLBuffer> tiles, coverage_polygons, coverage_vertices, candidate_patches, catalogue_bounds,
      candidate_bounds;
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
    // The first records are full source footprints and correspond to BVH
    // primitive IDs. Ownership polygons follow as metadata for streaming.
    std::vector<BvhCoveragePolygon> coverage_metadata(sources.size());
    std::vector<BvhCoverageVertex> coverage_vertex_metadata;
    std::vector<BvhAffinePatch> candidate_patch_metadata;
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
                          source.key.column,
                          uint32_t(bool(source.valid_cells))};
      const auto append_candidate_patch = [&](const TerrainTileTransform &transform,
                                              uint32_t minimum_column,
                                              uint32_t minimum_row,
                                              uint32_t maximum_column,
                                              uint32_t maximum_row) {
        const double ox = transform.origin.x - render_observer.x;
        const double oy = transform.origin.y - render_observer.y;
        const double determinant = transform.determinant();
        candidate_patch_metadata.push_back(
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
        // Coverage clipping can leave many small rectangles sharing one
        // affine transform. Bound this rectangle, not the transform's entire
        // original region, or those candidates overlap almost everywhere.
        double bx0 = INFINITY, by0 = INFINITY, bx1 = -INFINITY, by1 = -INFINITY;
        for (uint32_t column : {minimum_column, maximum_column}) {
          for (uint32_t row : {minimum_row, maximum_row}) {
            const Coord corner = transform.apply(column, row);
            bx0 = std::min(bx0, corner.x - render_observer.x);
            by0 = std::min(by0, corner.y - render_observer.y);
            bx1 = std::max(bx1, corner.x - render_observer.x);
            by1 = std::max(by1, corner.y - render_observer.y);
          }
        }
        // Cover projection residual and the Float32 forward/inverse affine
        // arithmetic used by the shaders, including cancellation at long range.
        const double coordinate_scale =
            1.0 + std::abs(ox) + std::abs(oy) +
            maximum_column *
                (std::abs(transform.column_step.x) + std::abs(transform.column_step.y)) +
            maximum_row * (std::abs(transform.row_step.x) + std::abs(transform.row_step.y));
        const double xy_guard = transform.maximum_residual_metres +
                                32 * std::numeric_limits<float>::epsilon() * coordinate_scale;
        bx0 = std::nextafter(float(bx0 - xy_guard), -INFINITY);
        by0 = std::nextafter(float(by0 - xy_guard), -INFINITY);
        bx1 = std::nextafter(float(bx1 + xy_guard), INFINITY);
        by1 = std::nextafter(float(by1 + xy_guard), INFINITY);
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
      };
      if (transformed) {
        const auto append_vertices = [&](const TerrainCoveragePolygon &polygon) {
          const uint32_t offset = checked_count(coverage_vertex_metadata.size());
          float minimum_x = std::numeric_limits<float>::infinity();
          float minimum_y = std::numeric_limits<float>::infinity();
          float maximum_x = -std::numeric_limits<float>::infinity();
          float maximum_y = -std::numeric_limits<float>::infinity();
          for (const Coord vertex : polygon.vertices) {
            const float vertex_x = float(vertex.x - render_observer.x);
            const float vertex_y = float(vertex.y - render_observer.y);
            coverage_vertex_metadata.push_back({vertex_x, vertex_y});
            minimum_x = std::min(minimum_x, vertex_x);
            minimum_y = std::min(minimum_y, vertex_y);
            maximum_x = std::max(maximum_x, vertex_x);
            maximum_y = std::max(maximum_y, vertex_y);
          }
          return std::pair{BvhCoveragePolygon{offset,
                                              checked_count(polygon.vertices.size()),
                                              checked_count(i),
                                              0U,
                                              0U,
                                              0U,
                                              0U,
                                              0U,
                                              0U,
                                              0U,
                                              minimum_x,
                                              minimum_y,
                                              maximum_x,
                                              maximum_y},
                           BvhBounds{minimum_x, minimum_y, -1e30F, maximum_x, maximum_y, 1e30F}};
        };
        BvhCoveragePolygon footprint = {};
        footprint.source = checked_count(i);
        footprint.coverage_offset = checked_count(coverage_metadata.size());
        footprint.coverage_count = checked_count(source.coverage_polygons.size());
        BvhBounds footprint_bounds = {INFINITY, INFINITY, -1e30F, -INFINITY, -INFINITY, 1e30F};
        for (const auto &polygon : source.coverage_polygons) {
          const auto [record, box] = append_vertices(polygon);
          coverage_metadata.push_back(record);
          footprint_bounds.min_x = std::min(footprint_bounds.min_x, box.min_x);
          footprint_bounds.min_y = std::min(footprint_bounds.min_y, box.min_y);
          footprint_bounds.max_x = std::max(footprint_bounds.max_x, box.max_x);
          footprint_bounds.max_y = std::max(footprint_bounds.max_y, box.max_y);
        }
        footprint.ownership_offset = footprint.coverage_offset;
        footprint.ownership_count = footprint.coverage_count;
        if (!source.ownership_polygons.empty()) {
          footprint.ownership_offset = checked_count(coverage_metadata.size());
          footprint.ownership_count = checked_count(source.ownership_polygons.size());
          for (const TerrainCoveragePolygon &polygon : source.ownership_polygons)
            coverage_metadata.push_back(append_vertices(polygon).first);
        }
        footprint.minimum_x = footprint_bounds.min_x;
        footprint.minimum_y = footprint_bounds.min_y;
        footprint.maximum_x = footprint_bounds.max_x;
        footprint.maximum_y = footprint_bounds.max_y;
        coverage_metadata[i] = footprint;
        boxes.push_back(footprint_bounds);
        for (const TerrainTransformPatch &patch : source.transform_patches) {
          append_candidate_patch(
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
        coverage_vertex_metadata.push_back({});
        append_candidate_patch(transform, 0U, 0U, geometry.cell_count, geometry.cell_count);
        boxes.push_back(candidate_boxes.back());
      }
    }
    // Precompute only overlapping higher-priority regions. Their vertex
    // ranges are shared, and a cheap XY check rejects most at each exact hit.
    const auto overlaps = [](const BvhCoveragePolygon &a, const BvhCoveragePolygon &b) {
      return a.minimum_x < b.maximum_x && a.maximum_x > b.minimum_x && a.minimum_y < b.maximum_y &&
             a.maximum_y > b.minimum_y;
    };
    for (size_t i = 0; i < sources.size(); ++i) {
      if (sources[i].transform_patches.empty())
        continue;
      const auto footprint = coverage_metadata[i];
      const uint32_t offset = checked_count(coverage_metadata.size());
      for (size_t j = 0; j < sources.size(); ++j) {
        if (sources[j].dataset_index >= sources[i].dataset_index)
          continue;
        const auto higher = coverage_metadata[j];
        if (!overlaps(footprint, higher))
          continue;
        for (uint32_t r = 0; r < higher.coverage_count; ++r) {
          const auto region = coverage_metadata[higher.coverage_offset + r];
          if (overlaps(footprint, region))
            coverage_metadata.push_back(region);
        }
      }
      coverage_metadata[i].blocker_offset = offset;
      coverage_metadata[i].blocker_count = checked_count(coverage_metadata.size()) - offset;
    }
    tiles = buffer(gpu.device(), sources.size(), sizeof(BvhTile), @"catalogue tiles");
    coverage_polygons = buffer(
        gpu.device(),
        coverage_metadata.size(),
        sizeof(BvhCoveragePolygon),
        @"catalogue coverage polygons"
    );
    coverage_vertices = buffer(
        gpu.device(),
        coverage_vertex_metadata.size(),
        sizeof(BvhCoverageVertex),
        @"catalogue coverage vertices"
    );
    candidate_patches = buffer(
        gpu.device(),
        candidate_patch_metadata.size(),
        sizeof(BvhAffinePatch),
        @"candidate patches"
    );
    catalogue_bounds = buffer(gpu.device(), boxes.size(), sizeof(BvhBounds), @"catalogue bounds");
    candidate_bounds =
        buffer(gpu.device(), candidate_boxes.size(), sizeof(BvhBounds), @"candidate bounds");
    std::memcpy(tiles.contents, tile_metadata.data(), tiles.length);
    std::memcpy(coverage_polygons.contents, coverage_metadata.data(), coverage_polygons.length);
    std::memcpy(
        coverage_vertices.contents,
        coverage_vertex_metadata.data(),
        coverage_vertices.length
    );
    std::memcpy(
        candidate_patches.contents,
        candidate_patch_metadata.data(),
        candidate_patches.length
    );
    std::memcpy(catalogue_bounds.contents, boxes.data(), catalogue_bounds.length);
    std::memcpy(candidate_bounds.contents, candidate_boxes.data(), candidate_bounds.length);

    auto descriptor = primitive_descriptor(checked_count(boxes.size()), catalogue_bounds);
    auto candidate_descriptor =
        primitive_descriptor(checked_count(candidate_patch_metadata.size()), candidate_bounds);
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
id<MTLBuffer> TerrainTileBvh::coverage_polygons() const { return state_->coverage_polygons; }
id<MTLBuffer> TerrainTileBvh::coverage_vertices() const { return state_->coverage_vertices; }
id<MTLBuffer> TerrainTileBvh::candidate_patches() const { return state_->candidate_patches; }
std::span<const BvhTile> TerrainTileBvh::metadata() const { return state_->tile_metadata; }
uint32_t TerrainTileBvh::coverage_step_limit() const {
  return checked_count(state_->coverage_vertices.length / sizeof(BvhCoverageVertex));
}
uint64_t TerrainTileBvh::bytes() const {
  return state_->tiles.length + state_->coverage_polygons.length +
         state_->coverage_vertices.length + state_->candidate_patches.length +
         state_->catalogue_bounds.length + state_->candidate_bounds.length +
         state_->catalogue.size + state_->candidates.size;
}
uint64_t TerrainTileBvh::generation() const { return state_->generation; }
double TerrainTileBvh::build_milliseconds() const { return state_->build_ms; }
} // namespace panorama
