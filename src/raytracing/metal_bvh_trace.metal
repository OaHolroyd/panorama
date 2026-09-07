#include "metal_bvh_types.metalh"
#include "terrain_intersection.metalh"
#include <metal_raytracing>

using namespace metal::raytracing;

inline float bvh_height(device const uchar *vertices, BvhTile tile, uint index, bool quantized) {
  device const uchar *data = vertices + tile.vertex_offset;
  return quantized ? sample_elevation(
                         reinterpret_cast<device const ushort *>(data)[index],
                         tile.base_decimeters
                     )
                   : reinterpret_cast<device const float *>(data)[index];
}

/// Immutable bounds use a tile-centred anchor: h(u,v) - k*(u*u+v*v).
/// Instance shear/translation applies observer curvature without rebuilding.
kernel void build_terrain_bvh_bounds(
    device const uchar *vertices [[buffer(0)]],
    device const BvhTile *tiles [[buffer(1)]],
    device const BvhBlock *blocks [[buffer(2)]],
    device BvhBounds *bounds [[buffer(3)]],
    constant BvhParameters &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
  if (index >= params.primitive_count)
    return;
  const BvhBlock block = blocks[index];
  const BvhTile tile = tiles[block.tile];
  float low = INFINITY, high = -INFINITY;
  for (uint y = block.y; y <= block.y + block.height; ++y) {
    for (uint x = block.x; x <= block.x + block.width; ++x) {
      const float z = bvh_height(vertices, tile, y * (tile.cell_count + 1U) + x, params.quantized);
      low = min(low, z);
      high = max(high, z);
    }
  }
  const float x0 = tile.x_min + float(block.x) * tile.cell_size;
  const float y0 = tile.y_min + float(block.y) * tile.cell_size;
  const float x1 = tile.x_min + float(block.x + block.width) * tile.cell_size;
  const float y1 = tile.y_min + float(block.y + block.height) * tile.cell_size;
  const float2 closest = clamp(float2(0.0F), float2(x0, y0), float2(x1, y1));
  const float2 farthest = max(abs(float2(x0, y0)), abs(float2(x1, y1)));
  const float k = params.trace.curvature_coefficient;
  const float far_radius = params.trace.max_distance + sqrt(2.0F) * tile.cell_size;
  const float far_lift = k * dot(farthest, farthest);
  const float near_lift = k * dot(closest, closest);
  // Float sample reconstruction and curvature arithmetic, plus the legacy
  // triangle solver's frozen curvature over at most one diagonal cell length.
  // This remains centimetres at ordinary ranges, rather than a terrain column.
  const float cell_diagonal = sqrt(2.0F) * tile.cell_size;
  const float range_lift = k * far_radius * far_radius;
  const float guard =
      max(0.002F, 32.0F * FLT_EPSILON * (max(abs(low), abs(high)) + range_lift + 1.0F)) +
      k * cell_diagonal * (2.0F * far_radius + cell_diagonal) +
      // validate_ray_field permits |length(direction.xy)-1| <= 1e-4.
      // Cover that parameter-space/radial-space difference without changing rays.
      2.01e-4F * range_lift;
  bounds[index] = {x0, y0, low - far_lift - guard, x1, y1, high - near_lift + guard};
}

/// No reciprocal of zero, and no 0*infinity at a boundary. Bounds are inclusive.
inline bool bvh_slab(
    float origin,
    float direction,
    float low,
    float high,
    thread float &near_t,
    thread float &far_t
) {
  if (direction == 0.0F)
    return origin >= low && origin <= high;
  const float a = (low - origin) / direction;
  const float b = (high - origin) / direction;
  near_t = max(near_t, min(a, b));
  far_t = min(far_t, max(a, b));
  // Round outward instead of dropping a tangent because two axes rounded apart.
  return near_t <= far_t + 8.0F * FLT_EPSILON * max(1.0F, max(abs(near_t), abs(far_t)));
}

struct BvhPayload {
  float closest;
  uint gradients;
  uint steps;
  uint evaluations;
  uint source;
  float3 world_direction;
};

struct BvhIntersection {
  bool accept [[accept_intersection]];
  float distance [[distance]];
};

template <typename Sample>
inline BvhIntersection intersect_terrain_block(
    device const Sample *vertices,
    BvhTile tile,
    BvhBlock block,
    float3 direction,
    float min_distance,
    float max_distance,
    constant BvhParameters &params,
    ray_data BvhPayload &payload
) {
  const float delta = tile.cell_size;
  const float inverse_delta = 1.0F / delta;
  const int sx = int(direction.x > 0.0F) - int(direction.x < 0.0F);
  const int sy = int(direction.y > 0.0F) - int(direction.y < 0.0F);
  float entry = 0.0F, exit = params.trace.max_distance;
  if (!bvh_slab(
          0.0F,
          direction.x,
          tile.x_min + float(block.x) * delta,
          tile.x_min + float(block.x + block.width) * delta,
          entry,
          exit
      ) ||
      !bvh_slab(
          0.0F,
          direction.y,
          tile.y_min + float(block.y) * delta,
          tile.y_min + float(block.y + block.height) * delta,
          entry,
          exit
      ) ||
      entry >= exit) {
    return {false, 0.0F};
  }
  const float nudge = max(1e-3F * delta, 8.0F * FLT_EPSILON * max(1.0F, entry));
  const float x_classify = entry * direction.x + float(sx) * nudge;
  const float y_classify = entry * direction.y + float(sy) * nudge;
  int x = clamp(
      int(floor((x_classify - tile.x_min) * inverse_delta)),
      int(block.x),
      int(block.x + block.width) - 1
  );
  int y = clamp(
      int(floor((y_classify - tile.y_min) * inverse_delta)),
      int(block.y),
      int(block.y + block.height) - 1
  );
  const float dtx = sx == 0 ? INFINITY : delta * abs(1.0F / direction.x);
  const float dty = sy == 0 ? INFINITY : delta * abs(1.0F / direction.y);
  float tx = sx == 0 ? INFINITY
                     : float(sx) * (tile.x_min * inverse_delta + float(x) + float(sx > 0)) * dtx;
  float ty = sy == 0 ? INFINITY
                     : float(sy) * (tile.y_min * inverse_delta + float(y) + float(sy > 0)) * dty;
  if (sx != 0)
    tx = next_boundary_after(tx, entry, dtx);
  if (sy != 0)
    ty = next_boundary_after(ty, entry, dty);

  while (x >= int(block.x) && x < int(block.x + block.width) && y >= int(block.y) &&
         y < int(block.y + block.height)) {
    const float end = min(min(tx, ty), params.trace.max_distance);
    // Use the entire cell interval, not the Z slab or a previous candidate's
    // closest distance: the established triangle routine is entry-dependent.
    float start = 0.0F;
    if (sx != 0)
      start = max(start, tx - dtx);
    if (sy != 0)
      start = max(start, ty - dty);
    start = min(start, end);
    const float cell_x = tile.x_min + float(x) * delta;
    const float cell_y = tile.y_min + float(y) * delta;
    const uint side = tile.cell_count + 1U;
    const uint sample_index = uint(y) * side + uint(x);
    const float maximum =
        max(max(sample_elevation(vertices[sample_index], tile.base_decimeters),
                sample_elevation(vertices[sample_index + 1U], tile.base_decimeters)),
            max(sample_elevation(vertices[sample_index + side], tile.base_decimeters),
                sample_elevation(vertices[sample_index + side + 1U], tile.base_decimeters)));
    const float stationary = -direction.z / (2.0F * params.trace.curvature_coefficient);
    if (minimum_curved_ray_elevation(
            params.trace.observer_elevation,
            direction.z,
            params.trace.curvature_coefficient,
            stationary,
            start,
            end
        ) <= maximum) {
      if (store_debugging_info)
        payload.evaluations++;
      Collision collision = use_bilinear_collisions ? bilinear_collision(
                                                          vertices,
                                                          side,
                                                          tile.base_decimeters,
                                                          cell_x,
                                                          cell_y,
                                                          inverse_delta,
                                                          uint(y),
                                                          uint(x),
                                                          params.trace.observer_elevation,
                                                          direction.xy,
                                                          direction.z,
                                                          params.trace.curvature_coefficient,
                                                          start,
                                                          end
                                                      )
                                                    : triangle_collision(
                                                          vertices,
                                                          side,
                                                          tile.base_decimeters,
                                                          cell_x,
                                                          cell_y,
                                                          delta,
                                                          uint(y),
                                                          uint(x),
                                                          params.trace.observer_elevation,
                                                          direction.xy,
                                                          direction.z,
                                                          params.trace.curvature_coefficient,
                                                          start
                                                      );
      // The catalogue DDA and cell DDA can round a shared far edge differently.
      // Clamp only a few ULPs so an edge hit remains in Metal's legal ray range.
      const float range_guard = 8.0F * FLT_EPSILON * max(1.0F, max_distance);
      if (collision.hit && collision.distance >= min_distance - range_guard &&
          collision.distance <= max_distance + range_guard) {
        collision.distance = clamp(collision.distance, min_distance, max_distance);
      }
      if (collision.hit && collision.distance >= min_distance &&
          collision.distance <= max_distance && collision.distance <= params.trace.max_distance) {
        if (collision.distance < payload.closest) {
          payload.closest = collision.distance;
          if (compute_surface_gradients) {
            if (use_c1_normals && x >= 1 && y >= 1 && x < int(tile.cell_count) - 1 &&
                y < int(tile.cell_count) - 1) {
              payload.gradients = interpolated_packed_surface_gradients(
                  vertices,
                  side,
                  tile.base_decimeters,
                  cell_x,
                  cell_y,
                  inverse_delta,
                  uint(y),
                  uint(x),
                  direction.xy,
                  collision.distance
              );
            } else if (use_bilinear_collisions) {
              payload.gradients = bilinear_packed_surface_gradients(
                  vertices,
                  side,
                  tile.base_decimeters,
                  cell_x,
                  cell_y,
                  inverse_delta,
                  uint(y),
                  uint(x),
                  direction.xy,
                  collision.distance
              );
            } else {
              payload.gradients = triangle_packed_surface_gradients(
                  vertices,
                  side,
                  tile.base_decimeters,
                  cell_x,
                  cell_y,
                  inverse_delta,
                  uint(y),
                  uint(x),
                  direction.xy,
                  collision.distance
              );
            }
          }
        }
        return {true, collision.distance};
      }
    }
    if (min(tx, ty) >= exit)
      break;
    if (ty < tx) {
      ty += dty;
      y += sy;
    } else {
      tx += dtx;
      x += sx;
    }
  }
  return {false, 0.0F};
}

[[intersection(bounding_box, instancing)]]
BvhIntersection terrain_bvh_intersection(
    float3 origin [[origin]],
    float3 direction [[direction]],
    float min_distance [[min_distance]],
    float max_distance [[max_distance]],
    uint primitive [[primitive_id]],
    uint instance [[instance_id]],
    ray_data BvhPayload &payload [[payload]],
    device const BvhChunk *chunks [[buffer(0)]],
    constant BvhParameters &params [[buffer(1)]]
) {
  if (store_debugging_info)
    payload.steps++;
  const BvhChunk chunk = chunks[instance];
  if (chunk.source != payload.source)
    return {false, 0};
  const BvhBounds box = chunk.bounds[primitive];
  float near_t = min_distance, far_t = max_distance;
  if (!bvh_slab(origin.x, direction.x, box.min_x, box.max_x, near_t, far_t) ||
      !bvh_slab(origin.y, direction.y, box.min_y, box.max_y, near_t, far_t) ||
      !bvh_slab(origin.z, direction.z, box.min_z, box.max_z, near_t, far_t)) {
    return {false, 0.0F};
  }
  const BvhBlock block = chunk.blocks[primitive];
  const BvhTile tile = chunk.tile;
  device const uchar *data = chunk.vertices;
  return params.quantized ? intersect_terrain_block(
                                reinterpret_cast<device const ushort *>(data),
                                tile,
                                block,
                                payload.world_direction,
                                min_distance,
                                max_distance,
                                params,
                                payload
                            )
                          : intersect_terrain_block(
                                reinterpret_cast<device const float *>(data),
                                tile,
                                block,
                                payload.world_direction,
                                min_distance,
                                max_distance,
                                params,
                                payload
                            );
}

/// Match the software frontier's termination at the first missing source.
inline float bvh_coverage_limit(
    RayDirection ray,
    constant BvhParameters &params,
    device const BvhTile *tiles,
    device const CatalogueTileHashEntry *catalogue,
    float limit
) {
  BvhTile tile = tiles[params.observer_source];
  const float width = tile.cell_size * float(tile.cell_count);
  float entry = 0.0F;
  for (uint crossed = 0; crossed <= params.source_count; ++crossed) {
    const float boundary = tile_exit_distance(
        tile.x_min,
        tile.y_min,
        width,
        1U,
        float4(ray.x, ray.y, ray.inverse_x, ray.inverse_y),
        entry
    );
    if (boundary >= limit)
      return limit;
    const float nudge =
        max(1e-3F * params.trace.cell_size, 8.0F * FLT_EPSILON * max(1.0F, boundary));
    const float x = boundary * ray.x + (ray.x == 0.0F ? 0.0F : copysign(nudge, ray.x));
    const float y = boundary * ray.y + (ray.y == 0.0F ? 0.0F : copysign(nudge, ray.y));
    const long dx = x < tile.x_min ? -1L : x >= tile.x_min + width ? 1L : 0L;
    const long dy = y < tile.y_min ? 1L : y >= tile.y_min + width ? -1L : 0L;
    if (dx == 0L && dy == 0L)
      return boundary;
    tile.row += dy;
    tile.column += dx;
    tile.x_min += float(dx) * width;
    tile.y_min -= float(dy) * width;
    if (lookup_catalogue_tile(catalogue, params.hash_capacity, tile.row, tile.column) == nullptr)
      return boundary;
    entry = boundary;
  }
  return entry;
}

kernel void trace_terrain_bvh(
    instance_acceleration_structure terrain [[buffer(0)]],
    intersection_function_table<instancing> functions [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device float *distances [[buffer(3)]],
    device float *elevations [[buffer(4)]],
    device uint *gradients [[buffer(5)]],
    device float *steps [[buffer(6)]],
    device float *evaluations [[buffer(7)]],
    constant BvhParameters &params [[buffer(8)]],
    device const BvhTile *tiles [[buffer(9)]],
    device const CatalogueTileHashEntry *catalogue [[buffer(10)]],
    device BvhRayState *states [[buffer(11)]],
    device const uint *work [[buffer(12)]],
    uint work_index [[thread_position_in_grid]]
) {
  if (work_index >= params.work_count)
    return;
  const uint index = work[work_index];
  const RayDirection direction = rays[index];
  ray r;
  r.origin = float3(0.0F, 0.0F, params.trace.observer_elevation);
  r.direction = float3(direction.x, direction.y, direction.slope);
  r.min_distance = states[index].progress;
  r.max_distance = min(states[index].exit, params.trace.max_distance);
  intersector<instancing> tracer;
  tracer.assume_geometry_type(geometry_type::bounding_box);
  BvhPayload payload = {INFINITY, 0U, 0U, 0U, states[index].source, r.direction};
  const auto result = tracer.intersect(r, terrain, functions, payload);
  states[index].source = 0xffffffffU;
  bool hit = result.type != intersection_type::none;
  states[index].progress = states[index].exit;
  states[index].done = hit || states[index].progress >= params.trace.max_distance;
  // Only successful rays need coverage verification, and only up to the hit.
  // Walking the entire catalogue first wastes work on sky and nearby terrain.
  if (hit) {
    const float limit = bvh_coverage_limit(direction, params, tiles, catalogue, result.distance);
    hit = result.distance <= limit + 8.0F * FLT_EPSILON * max(1.0F, limit);
  }
  distances[index] = hit ? result.distance : 0.0F;
  if (store_collision_elevations)
    elevations[index] = hit ? curved_ray_elevation(
                                  params.trace.observer_elevation,
                                  direction.slope,
                                  params.trace.curvature_coefficient,
                                  result.distance
                              )
                            : 0.0F;
  if (compute_surface_gradients)
    gradients[index] = hit ? payload.gradients : 0U;
  if (store_debugging_info) {
    steps[index] += float(payload.steps);
    evaluations[index] += float(payload.evaluations);
  }
}

struct TileSelection {
  uint source;
  float exit;
  float3 world_direction;
};

/// Select the next tile in XY order among the conservative 3D candidates.
/// Tile footprints are disjoint except at edges, so completing this tile
/// proves all earlier terrain intervals have been examined or culled.
[[intersection(bounding_box)]]
BvhIntersection terrain_tile_intersection(
    float min_distance [[min_distance]],
    float max_distance [[max_distance]],
    uint primitive [[primitive_id]],
    ray_data TileSelection &payload [[payload]],
    device const BvhTile *tiles [[buffer(0)]]
) {
  const BvhTile tile = tiles[primitive];
  const float width = tile.cell_size * float(tile.cell_count);
  float entry = 0.0F, exit = max_distance;
  if (!bvh_slab(0, payload.world_direction.x, tile.x_min, tile.x_min + width, entry, exit) ||
      !bvh_slab(0, payload.world_direction.y, tile.y_min, tile.y_min + width, entry, exit) ||
      entry >= exit || exit <= min_distance)
    return {false, 0};
  // Half-open ownership for rays parallel to a shared tile edge.
  if ((payload.world_direction.x == 0 && tile.x_min + width <= 0) ||
      (payload.world_direction.y == 0 && tile.y_min + width <= 0))
    return {false, 0};
  payload.source = primitive;
  payload.exit = exit;
  return {true, max(entry, min_distance)};
}

kernel void select_terrain_tiles(
    primitive_acceleration_structure catalogue [[buffer(0)]],
    intersection_function_table<> functions [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device BvhRayState *states [[buffer(3)]],
    constant BvhParameters &params [[buffer(4)]],
    device const BvhTile *tiles [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
  if (index >= params.trace.ray_count || states[index].done || states[index].source != 0xffffffffU)
    return;
  ray r;
  r.origin = float3(0, 0, params.trace.observer_elevation);
  r.direction = float3(rays[index].x, rays[index].y, rays[index].slope);
  r.min_distance = states[index].progress;
  r.max_distance = params.trace.max_distance;
  intersector<> tracer;
  tracer.assume_geometry_type(geometry_type::bounding_box);
  TileSelection payload = {0xffffffffU, 0, r.direction};
  const auto hit = tracer.intersect(r, catalogue, functions, payload);
  if (hit.type == intersection_type::none) {
    states[index].done = 1U;
    states[index].source = 0xffffffffU;
  } else {
    // The closest primitive ID is authoritative: callbacks need not arrive
    // in distance order, and may be invoked for farther candidates too.
    states[index].source = hit.primitive_id;
    const BvhTile tile = tiles[hit.primitive_id];
    const float width = tile.cell_size * float(tile.cell_count);
    float entry = 0, exit = params.trace.max_distance;
    bvh_slab(0, r.direction.x, tile.x_min, tile.x_min + width, entry, exit);
    bvh_slab(0, r.direction.y, tile.y_min, tile.y_min + width, entry, exit);
    states[index].exit = exit;
  }
}
