#include "metal_bvh_types.metalh"
#include "terrain_intersection.metalh"
#include "terrain_tile_bvh.metalh"

using namespace metal::raytracing;

kernel void initialize_bvh_continuations(
    device BvhRayState *states [[buffer(0)]],
    constant RaytraceParameters &params [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
  const uint index = position.y * params.image_width + position.x;
  states[index] = {0, 0, 0xffffffffU, 0xffffffffU, 0};
}

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
    device const BvhAffinePatch *transforms [[buffer(5)]],
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
  float x0 = tile.x_min + float(block.x) * tile.cell_size;
  float y0 = tile.y_min + float(block.y) * tile.cell_size;
  float x1 = tile.x_min + float(block.x + block.width) * tile.cell_size;
  float y1 = tile.y_min + float(block.y + block.height) * tile.cell_size;
  if (params.transformed_catalogue) {
    const BvhAffinePatch transform = transforms[block.transform];
    const float2 p00 = float2(transform.origin_x, transform.origin_y) +
                       x0 * float2(transform.column_x, transform.column_y) +
                       y0 * float2(transform.row_x, transform.row_y);
    const float2 p10 = float2(transform.origin_x, transform.origin_y) +
                       x1 * float2(transform.column_x, transform.column_y) +
                       y0 * float2(transform.row_x, transform.row_y);
    const float2 p01 = float2(transform.origin_x, transform.origin_y) +
                       x0 * float2(transform.column_x, transform.column_y) +
                       y1 * float2(transform.row_x, transform.row_y);
    const float2 p11 = float2(transform.origin_x, transform.origin_y) +
                       x1 * float2(transform.column_x, transform.column_y) +
                       y1 * float2(transform.row_x, transform.row_y);
    const float2 low_xy = min(min(p00, p10), min(p01, p11));
    const float2 high_xy = max(max(p00, p10), max(p01, p11));
    x0 = low_xy.x;
    y0 = low_xy.y;
    x1 = high_xy.x;
    y1 = high_xy.y;
    const float base_side = float(tile.cell_count) * tile.cell_size;
    const float2 anchor =
        float2(transform.origin_x, transform.origin_y) +
        0.5F * base_side *
            float2(transform.column_x + transform.row_x, transform.column_y + transform.row_y);
    const float radius =
        max(max(distance(p00, anchor), distance(p10, anchor)),
            max(distance(p01, anchor), distance(p11, anchor)));
    const float curvature_guard = params.trace.curvature_coefficient * radius * radius + 1.01F;
    bounds[index] = {x0, y0, low - curvature_guard, x1, y1, high + curvature_guard};
    return;
  }
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

struct BvhPayload {
  float closest;
  uint gradients;
  uint steps;
  uint evaluations;
  uint source;
  float3 world_direction;
  float3 world_origin;
};

template <typename Sample>
inline BvhIntersection intersect_terrain_block(
    device const Sample *vertices,
    BvhTile tile,
    BvhBlock block,
    float2 horizontal_origin,
    float3 direction,
    float observer_elevation,
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
          horizontal_origin.x,
          direction.x,
          tile.x_min + float(block.x) * delta,
          tile.x_min + float(block.x + block.width) * delta,
          entry,
          exit
      ) ||
      !bvh_slab(
          horizontal_origin.y,
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
  const float x_classify = horizontal_origin.x + entry * direction.x + float(sx) * nudge;
  const float y_classify = horizontal_origin.y + entry * direction.y + float(sy) * nudge;
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
  float tx =
      sx == 0 ? INFINITY
              : (tile.x_min + float(x + int(sx > 0)) * delta - horizontal_origin.x) / direction.x;
  float ty =
      sy == 0 ? INFINITY
              : (tile.y_min + float(y + int(sy > 0)) * delta - horizontal_origin.y) / direction.y;
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
            observer_elevation,
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
                                                          observer_elevation,
                                                          direction.xy,
                                                          direction.z,
                                                          params.trace.curvature_coefficient,
                                                          start,
                                                          end,
                                                          horizontal_origin
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
                                                          observer_elevation,
                                                          direction.xy,
                                                          direction.z,
                                                          params.trace.curvature_coefficient,
                                                          start,
                                                          horizontal_origin
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
                  collision.distance,
                  horizontal_origin
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
                  collision.distance,
                  horizontal_origin
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
                  collision.distance,
                  horizontal_origin
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
  if (payload.source != 0xffffffffU && chunk.source != payload.source)
    return {false, 0};
  // The whole-scene path must preserve the streaming selector's half-open
  // ownership when a ray lies exactly on a shared tile edge.
  if (!chunk.transformed) {
    const float width = chunk.tile.cell_size * float(chunk.tile.cell_count);
    if ((payload.world_direction.x == 0 && chunk.tile.x_min + width <= payload.world_origin.x) ||
        (payload.world_direction.y == 0 && chunk.tile.y_min + width <= payload.world_origin.y))
      return {false, 0};
  }
  const BvhBounds box = chunk.bounds[primitive];
  float near_t = min_distance, far_t = max_distance;
  if (!bvh_slab(origin.x, direction.x, box.min_x, box.max_x, near_t, far_t) ||
      !bvh_slab(origin.y, direction.y, box.min_y, box.max_y, near_t, far_t) ||
      !bvh_slab(origin.z, direction.z, box.min_z, box.max_z, near_t, far_t)) {
    return {false, 0.0F};
  }
  const BvhBlock block = chunk.blocks[primitive];
  BvhTile tile = chunk.tile;
  float2 horizontal_origin(0.0F);
  float3 collision_direction = payload.world_direction;
  BvhAffinePatch transform = {};
  if (chunk.transformed) {
    transform = chunk.transforms[block.transform];
    const float2 relative = origin.xy - float2(transform.origin_x, transform.origin_y);
    horizontal_origin = float2(
        transform.inverse_xx * relative.x + transform.inverse_xy * relative.y,
        transform.inverse_yx * relative.x + transform.inverse_yy * relative.y
    );
    collision_direction.xy = float2(
        transform.inverse_xx * direction.x + transform.inverse_xy * direction.y,
        transform.inverse_yx * direction.x + transform.inverse_yy * direction.y
    );
  } else {
    tile.x_min -= payload.world_origin.x;
    tile.y_min -= payload.world_origin.y;
  }
  device const uchar *data = chunk.vertices;
  const float raw_observer_elevation = payload.world_origin.z - chunk.vertical_offset;
  const BvhIntersection result = params.quantized
                                     ? intersect_terrain_block(
                                           reinterpret_cast<device const ushort *>(data),
                                           tile,
                                           block,
                                           horizontal_origin,
                                           collision_direction,
                                           raw_observer_elevation,
                                           min_distance,
                                           max_distance,
                                           params,
                                           payload
                                       )
                                     : intersect_terrain_block(
                                           reinterpret_cast<device const float *>(data),
                                           tile,
                                           block,
                                           horizontal_origin,
                                           collision_direction,
                                           raw_observer_elevation,
                                           min_distance,
                                           max_distance,
                                           params,
                                           payload
                                       );
  if (result.accept && chunk.transformed && compute_surface_gradients) {
    const float2 logical = float2(as_type<half2>(payload.gradients));
    constexpr float kMaximumHalf = 65504.0F;
    const float2 world = clamp(
        float2(
            transform.inverse_xx * logical.x + transform.inverse_yx * logical.y,
            transform.inverse_xy * logical.x + transform.inverse_yy * logical.y
        ),
        -kMaximumHalf,
        kMaximumHalf
    );
    payload.gradients = as_type<uint>(half2(world));
  }
  return result;
}

/// Match the software frontier's termination at the first missing source.
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
  BvhPayload payload = {INFINITY, 0U, 0U, 0U, states[index].source, r.direction, r.origin};
  const auto result = tracer.intersect(r, terrain, functions, payload);
  states[index].source = 0xffffffffU;
  bool hit = result.type != intersection_type::none;
  states[index].progress = states[index].exit;
  states[index].done = hit || states[index].progress >= params.trace.max_distance;
  // Only successful rays need coverage verification, and only up to the hit.
  // Walking the entire catalogue first wastes work on sky and nearby terrain.
  if (hit && !params.transformed_catalogue) {
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

// Traverse all cached tiles in one GPU dispatch. A second, conservative
// catalogue query proves no missing tile can occlude the tentative hit. Rays
// without that proof remain at progress zero for the bounded streaming path.
kernel void trace_terrain_scene(
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
    primitive_acceleration_structure missing_tiles [[buffer(13)]],
    intersection_function_table<> missing_functions [[buffer(14)]],
    device atomic_uint *missing_count [[buffer(15)]],
    device const BvhAffinePatch *candidate_patches [[buffer(16)]],
    device atomic_uint *requested_sources [[buffer(17)]],
    primitive_acceleration_structure coverage [[buffer(20)]],
    intersection_function_table<> coverage_functions [[buffer(21)]],
    device const BvhCoveragePolygon *coverage_polygons [[buffer(22)]],
    device const BvhCoverageVertex *coverage_vertices [[buffer(23)]],
    uint2 position [[thread_position_in_grid]]
) {
  const uint index = position.y * params.trace.image_width + position.x;
  if (index >= params.trace.ray_count)
    return;
  states[index] = {0, 0, 0xffffffffU, 0xffffffffU, 0};
  distances[index] = 0.0F;
  if (store_collision_elevations)
    elevations[index] = 0.0F;
  if (compute_surface_gradients)
    gradients[index] = 0U;
  if (store_debugging_info) {
    steps[index] = 0.0F;
    evaluations[index] = 0.0F;
  }
  const RayDirection direction = rays[index];
  ray r;
  r.origin = float3(0.0F, 0.0F, params.trace.observer_elevation);
  r.direction = float3(direction.x, direction.y, direction.slope);
  r.min_distance = 0.0F;
  r.max_distance = params.trace.max_distance;
  intersector<instancing> tracer;
  tracer.assume_geometry_type(geometry_type::bounding_box);
  BvhPayload payload = {INFINITY, 0U, 0U, 0U, 0xffffffffU, r.direction, r.origin};
  const auto result = tracer.intersect(r, terrain, functions, payload);
  bool hit = result.type != intersection_type::none;
  if (hit) {
    r.max_distance =
        min(params.trace.max_distance,
            result.distance + 8.0F * FLT_EPSILON * max(1.0F, result.distance));
  }
  intersector<> missing_tracer;
  missing_tracer.assume_geometry_type(geometry_type::bounding_box);
  missing_tracer.accept_any_intersection(true);
  TileSelection missing_payload = {r.direction, true, float2(0), 0xffffffffU, 0xffffffffU, false};
  const auto missing =
      missing_tracer.intersect(r, missing_tiles, missing_functions, missing_payload);
  if (missing.type != intersection_type::none) {
    const uint source = params.transformed_catalogue
                            ? candidate_patches[missing.primitive_id].source
                            : missing.primitive_id;
    atomic_store_explicit(requested_sources + source, 1U, memory_order_relaxed);
    atomic_fetch_add_explicit(missing_count, 1U, memory_order_relaxed);
    return;
  }
  if (hit) {
    const float limit =
        params.transformed_catalogue
            ? transformed_coverage_limit(
                  coverage,
                  coverage_functions,
                  coverage_polygons,
                  coverage_vertices,
                  direction,
                  params.trace.observer_elevation,
                  result.distance,
                  params.coverage_step_limit
              )
            : bvh_coverage_limit(direction, params, tiles, catalogue, result.distance);
    hit = result.distance <= limit + 8.0F * FLT_EPSILON * max(1.0F, limit);
  }
  states[index].done = 1U;
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
    steps[index] = float(payload.steps);
    evaluations[index] = float(payload.evaluations);
  }
}

// Secondary rays use the primary scene's curvature frame. The collision
// solver still receives the original, shadow-origin-relative ray, matching
// the mipmap shadow path's bias and horizontal-distance parameterization.
kernel void trace_scene_shadows(
    instance_acceleration_structure terrain [[buffer(0)]],
    intersection_function_table<instancing> functions [[buffer(1)]],
    device const RayDirection *camera_rays [[buffer(2)]],
    device const float *distances [[buffer(3)]],
    device const float *elevations [[buffer(4)]],
    device const uint *gradients [[buffer(5)]],
    constant BvhParameters &params [[buffer(8)]],
    device const BvhTile *tiles [[buffer(9)]],
    device const CatalogueTileHashEntry *catalogue [[buffer(10)]],
    device const BvhRayState *states [[buffer(11)]],
    primitive_acceleration_structure missing_tiles [[buffer(13)]],
    intersection_function_table<> missing_functions [[buffer(14)]],
    device atomic_uint *missing_count [[buffer(15)]],
    constant float4 &sun [[buffer(16)]],
    device uchar *visibility [[buffer(17)]],
    device atomic_uint *requested_sources [[buffer(18)]],
    device const BvhAffinePatch *candidate_patches [[buffer(19)]],
    primitive_acceleration_structure coverage [[buffer(20)]],
    intersection_function_table<> coverage_functions [[buffer(21)]],
    device const BvhCoveragePolygon *coverage_polygons [[buffer(22)]],
    device const BvhCoverageVertex *coverage_vertices [[buffer(23)]],
    uint2 position [[thread_position_in_grid]]
) {
  const uint index = position.y * params.trace.image_width + position.x;
  if (index >= params.trace.ray_count)
    return;
  visibility[index] = sun.w < 0 ? 0U : 1U;
  if (sun.w != 0 || !states[index].done || !(distances[index] > 0))
    return;
  const float2 gradient = float2(as_type<half2>(gradients[index]));
  const float3 normal = normalize(float3(-gradient, 1));
  if (dot(normal, normalize(sun.xyz)) <= 0)
    return;
  const float horizontal_bias = max(0.1F, 0.05F * params.trace.cell_size);
  const float vertical_bias = max(0.05F, 0.005F * params.trace.cell_size);
  const RayDirection camera = camera_rays[index];
  const float3 origin = float3(
      distances[index] * float2(camera.x, camera.y) + horizontal_bias * sun.xy,
      elevations[index] + horizontal_bias * sun.z + vertical_bias
  );
  const RayDirection direction = {sun.x,
                                  sun.y,
                                  sun.x == 0 ? INFINITY : 1 / sun.x,
                                  sun.y == 0 ? INFINITY : 1 / sun.y,
                                  sun.z};
  float limit = params.trace.max_distance;
  if (!params.transformed_catalogue) {
    const BvhTile observer_tile = tiles[params.observer_source];
    const float width = observer_tile.cell_size * float(observer_tile.cell_count);
    const long column =
        observer_tile.column + long(floor((origin.x - observer_tile.x_min) / width));
    const long row = observer_tile.row - long(floor((origin.y - observer_tile.y_min) / width));
    device const CatalogueTileHashEntry *source =
        lookup_catalogue_tile(catalogue, params.hash_capacity, row, column);
    if (source == nullptr)
      return;
    limit = bvh_coverage_limit(
        direction,
        params,
        tiles,
        catalogue,
        params.trace.max_distance,
        origin.xy,
        source->source_index
    );
  }
  if (!(limit > 0))
    return;
  const float k = params.trace.curvature_coefficient;
  ray r;
  r.origin = float3(origin.xy, origin.z - k * dot(origin.xy, origin.xy));
  r.direction = float3(sun.xy, sun.z - 2 * k * dot(origin.xy, sun.xy));
  r.min_distance = 0;
  r.max_distance = limit;
  intersector<instancing> tracer;
  tracer.assume_geometry_type(geometry_type::bounding_box);
  tracer.accept_any_intersection(true);
  BvhPayload payload = {INFINITY, 0U, 0U, 0U, 0xffffffffU, sun.xyz, origin};
  const auto hit = tracer.intersect(r, terrain, functions, payload);
  if (hit.type != intersection_type::none) {
    const float coverage_limit = params.transformed_catalogue ? transformed_coverage_limit(
                                                                    coverage,
                                                                    coverage_functions,
                                                                    coverage_polygons,
                                                                    coverage_vertices,
                                                                    direction,
                                                                    origin.z,
                                                                    hit.distance,
                                                                    params.coverage_step_limit,
                                                                    origin.xy
                                                                )
                                                              : limit;
    if (hit.distance <= coverage_limit + 8.0F * FLT_EPSILON * max(1.0F, coverage_limit)) {
      visibility[index] = 0U;
      return; // A reachable known occluder proves shadow even if later tiles are absent.
    }
  }
  intersector<> missing_tracer;
  missing_tracer.assume_geometry_type(geometry_type::bounding_box);
  missing_tracer.accept_any_intersection(true);
  TileSelection missing_payload = {sun.xyz, true, origin.xy, 0xffffffffU, 0xffffffffU, false};
  const auto missing =
      missing_tracer.intersect(r, missing_tiles, missing_functions, missing_payload);
  if (missing.type != intersection_type::none) {
    const uint source = params.transformed_catalogue
                            ? candidate_patches[missing.primitive_id].source
                            : missing.primitive_id;
    atomic_store_explicit(requested_sources + source, 1U, memory_order_relaxed);
    atomic_fetch_add_explicit(missing_count, 1U, memory_order_relaxed);
  }
}
