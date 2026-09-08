#include "terrain_tile_bvh.h"

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
  std::vector<uint32_t> selected_lods;
  std::vector<BvhTile> tile_metadata;
  id<MTLBuffer> tiles, catalogue_bounds;
  id<MTLAccelerationStructure> catalogue;
  uint64_t generation = 0;
  double build_ms = 0;
  explicit State(GpuRaytraceResources &resources) : gpu(resources) {}

  void rebuild(TileManager &manager, const RaytraceParameters &parameters) {
    const auto &grid = manager.catalogue().grid();
    const auto &sources = manager.sources();
    const auto &geometry = manager.origin_geometry();
    tile_metadata.resize(sources.size());
    std::vector<BvhBounds> boxes(sources.size());
    const double k = parameters.curvature_coefficient;
    for (size_t i = 0; i < sources.size(); ++i) {
      const auto &source = sources[i];
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

    auto descriptor = primitive_descriptor(checked_count(tile_metadata.size()), catalogue_bounds);
    const auto sizes = [gpu.device() accelerationStructureSizesWithDescriptor:descriptor];
    auto next = [gpu.device() newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    if (next == nil)
      throw std::runtime_error("Could not allocate catalogue BVH");
    next.label = @"Shared terrain catalogue BVH";
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
    [encoder endEncoding];
    build_ms = complete(command);
    catalogue = next;
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
  bool lods_changed = state.selected_lods.size() != manager.sources().size();
  for (uint32_t i = 0; !lods_changed && i < state.selected_lods.size(); ++i)
    lods_changed = state.selected_lods[i] != manager.lod_for_source(i);
  if (state.catalogue != nil && state.observer.easting == observer.easting &&
      state.observer.northing == observer.northing && !lods_changed &&
      state.curvature == parameters.curvature_coefficient)
    return false;
  state.selected_lods.resize(manager.sources().size());
  for (uint32_t i = 0; i < state.selected_lods.size(); ++i)
    state.selected_lods[i] = manager.lod_for_source(i);
  state.observer = observer;
  state.curvature = parameters.curvature_coefficient;
  state.catalogue = nil; // Failed construction must be retried on the next frame.
  state.rebuild(manager, parameters);
  return true;
}
id<MTLAccelerationStructure> TerrainTileBvh::acceleration() const { return state_->catalogue; }
id<MTLBuffer> TerrainTileBvh::tiles() const { return state_->tiles; }
std::span<const BvhTile> TerrainTileBvh::metadata() const { return state_->tile_metadata; }
uint64_t TerrainTileBvh::bytes() const {
  return state_->tiles.length + state_->catalogue_bounds.length + state_->catalogue.size;
}
uint64_t TerrainTileBvh::generation() const { return state_->generation; }
double TerrainTileBvh::build_milliseconds() const { return state_->build_ms; }
} // namespace panorama
