#include "terrain_tile_bvh.metalh"

/// Select the next tile in XY order among the conservative 3D candidates.
/// Tile footprints are disjoint except at edges, so completing this tile
/// proves all earlier terrain intervals have been examined or culled.
[[intersection(bounding_box)]]
BvhIntersection terrain_tile_intersection(
    float min_distance [[min_distance]],
    float max_distance [[max_distance]],
    uint primitive [[primitive_id]],
    ray_data TileSelection &payload [[payload]],
    device const BvhTile *tiles [[buffer(0)]],
    device const uint *resident [[buffer(1)]]
) {
  if (primitive == payload.excluded_source || primitive == payload.excluded_primitive ||
      (payload.skip_resident && resident[primitive] != 0U))
    return {false, 0};
  const BvhTile tile = tiles[primitive];
  const float width = tile.cell_size * float(tile.cell_count);
  float entry = 0.0F, exit = max_distance;
  if (!bvh_slab(
          payload.origin.x,
          payload.world_direction.x,
          tile.x_min,
          tile.x_min + width,
          entry,
          exit
      ) ||
      !bvh_slab(
          payload.origin.y,
          payload.world_direction.y,
          tile.y_min,
          tile.y_min + width,
          entry,
          exit
      ) ||
      entry >= exit || exit <= min_distance)
    return {false, 0};
  // Half-open ownership for rays parallel to a shared tile edge.
  if ((payload.world_direction.x == 0 && tile.x_min + width <= payload.origin.x) ||
      (payload.world_direction.y == 0 && tile.y_min + width <= payload.origin.y))
    return {false, 0};
  return {true, max(entry, min_distance)};
}

/// Exact ray/parallelogram test for transformed catalogue patches.
[[intersection(bounding_box)]]
BvhIntersection terrain_patch_intersection(
    float min_distance [[min_distance]],
    float max_distance [[max_distance]],
    uint primitive [[primitive_id]],
    ray_data TileSelection &payload [[payload]],
    device const BvhAffinePatch *patches [[buffer(0)]],
    device const uint *resident [[buffer(1)]]
) {
  const BvhAffinePatch patch = patches[primitive];
  if (primitive == payload.excluded_primitive || patch.source == payload.excluded_source ||
      (payload.skip_resident && resident[patch.source] != 0U))
    return {false, 0};
  const float2 relative = payload.origin - float2(patch.origin_x, patch.origin_y);
  const float2 origin = float2(
      patch.inverse_xx * relative.x + patch.inverse_xy * relative.y,
      patch.inverse_yx * relative.x + patch.inverse_yy * relative.y
  );
  const float2 direction = float2(
      patch.inverse_xx * payload.world_direction.x + patch.inverse_xy * payload.world_direction.y,
      patch.inverse_yx * payload.world_direction.x + patch.inverse_yy * payload.world_direction.y
  );
  float entry = 0.0F, exit = max_distance;
  if (!bvh_slab(origin.x, direction.x, patch.minimum_column, patch.maximum_column, entry, exit) ||
      !bvh_slab(origin.y, direction.y, patch.minimum_row, patch.maximum_row, entry, exit) ||
      entry >= exit || exit <= min_distance)
    return {false, 0};
  if ((direction.x == 0 && origin.x >= patch.maximum_column) ||
      (direction.y == 0 && origin.y >= patch.maximum_row))
    return {false, 0};
  return {true, max(entry, min_distance)};
}

/// Exact horizontal ray/polygon test for the shared coverage tessellation.
[[intersection(bounding_box)]]
BvhIntersection terrain_coverage_polygon_intersection(
    float min_distance [[min_distance]],
    float max_distance [[max_distance]],
    uint primitive [[primitive_id]],
    ray_data TileSelection &payload [[payload]],
    device const BvhCoveragePolygon *polygons [[buffer(0)]],
    device const BvhCoverageVertex *vertices [[buffer(1)]],
    device const uint *resident [[buffer(2)]]
) {
  const BvhCoveragePolygon polygon = polygons[primitive];
  if (polygon.source == payload.excluded_source ||
      (payload.skip_resident && resident[polygon.source] != 0U))
    return {false, 0};
  // Begin at the requested progress: a non-convex footprint or a source's
  // separate ownership regions can be entered more than once by the same ray.
  float entry = min_distance, exit = max_distance;
  if (!bvh_source_interval(
          polygon,
          polygons,
          vertices,
          payload.ownership_only,
          payload.origin,
          payload.world_direction.xy,
          entry,
          exit
      ) ||
      entry >= exit || exit <= min_distance)
    return {false, 0};
  return {true, max(entry, min_distance)};
}

kernel void select_terrain_coverage_polygons(
    primitive_acceleration_structure catalogue [[buffer(0)]],
    intersection_function_table<> functions [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device BvhRayState *states [[buffer(3)]],
    constant BvhParameters &params [[buffer(4)]],
    device const BvhCoveragePolygon *polygons [[buffer(5)]],
    device const BvhCoverageVertex *vertices [[buffer(6)]],
    device const uint *work [[buffer(7)]],
    uint work_index [[thread_position_in_grid]]
) {
  if (work_index >= params.work_count)
    return;
  const uint index = work[work_index];
  if (states[index].done || states[index].source != 0xffffffffU)
    return;
  auto selected = select_next_coverage_polygon(
      catalogue,
      functions,
      polygons,
      vertices,
      rays[index],
      params.trace.observer_elevation,
      states[index].progress,
      params.trace.max_distance,
      0xffffffffU,
      states[index].primitive,
      float2(0.0F),
      true,
      float2(params.catalogue_x, params.catalogue_y)
  );
  states[index].source = selected.source;
  states[index].primitive = selected.primitive;
  states[index].exit = selected.exit;
  if (selected.source == 0xffffffffU)
    states[index].done = 1U;
}

kernel void select_terrain_tiles(
    primitive_acceleration_structure catalogue [[buffer(0)]],
    intersection_function_table<> functions [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device BvhRayState *states [[buffer(3)]],
    constant BvhParameters &params [[buffer(4)]],
    device const BvhTile *tiles [[buffer(5)]],
    device const uint *work [[buffer(7)]],
    uint work_index [[thread_position_in_grid]]
) {
  if (work_index >= params.work_count)
    return;
  const uint index = work[work_index];
  if (states[index].done || states[index].source != 0xffffffffU)
    return;
  const auto selected = select_next_terrain_tile(
      catalogue,
      functions,
      tiles,
      rays[index],
      params.trace.observer_elevation,
      states[index].progress,
      params.trace.max_distance,
      0xffffffffU,
      states[index].primitive
  );
  states[index].source = selected.source;
  states[index].primitive = selected.primitive;
  states[index].exit = selected.exit;
  if (selected.source == 0xffffffffU)
    states[index].done = 1U;
}

// Continue the mipmap frontier through the same catalogue used by BVH tracing.
// The within-tile kernel and CPU residency scheduler are unchanged.
kernel void emit_bvh_tile_frontier(
    device const RayWorkItem *active_items [[buffer(0)]],
    device const ResidentTile *resident_tiles [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device const float *continuations [[buffer(3)]],
    constant RaytraceParameters &params [[buffer(4)]],
    constant uint &frontier_capacity [[buffer(5)]],
    device DeferredRayWork *deferred_items [[buffer(6)]],
    device atomic_uint *deferred_count [[buffer(7)]],
    device const CatalogueTileHashEntry *catalogue_hash [[buffer(8)]],
    constant uint &hash_capacity [[buffer(9)]],
    primitive_acceleration_structure catalogue [[buffer(12)]],
    intersection_function_table<> functions [[buffer(13)]],
    device const BvhTile *tiles [[buffer(15)]],
    uint work_index [[thread_position_in_grid]]
) {
  const float progress = continuations[work_index];
  if (!isfinite(progress) || progress >= params.max_distance)
    return;
  const RayWorkItem active = active_items[work_index];
  const RayDirection direction = rays[active.ray_index];
  const ResidentTile current = resident_tiles[active.slot];
  const auto current_source =
      lookup_catalogue_tile(catalogue_hash, hash_capacity, current.row, current.column);
  const auto selected = select_next_terrain_tile(
      catalogue,
      functions,
      tiles,
      direction,
      params.observer_elevation,
      progress,
      params.max_distance,
      current_source->source_index
  );
  if (selected.source == 0xffffffffU)
    return;
  const uint index = atomic_fetch_add_explicit(deferred_count, 1U, memory_order_relaxed);
  if (index < frontier_capacity)
    deferred_items[index] = {active.ray_index, selected.source, selected.entry};
}
