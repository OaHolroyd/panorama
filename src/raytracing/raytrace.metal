#include "terrain_intersection.metalh"

/// Expand fixed-point tile vertices from a Metal I/O staging buffer into
/// their final Float32 atlas slots. Each staged record contains the complete
/// logical tile stream, allowing this kernel to read the per-tile base from
/// its header without a separate host-side metadata request.
kernel void convert_quantized_vertices(
    device const uchar *source_records [[buffer(0)]],
    device float *destination [[buffer(1)]],
    device const uint *destination_slots [[buffer(2)]],
    constant uint &source_record_stride [[buffer(3)]],
    constant uint &vertex_offset [[buffer(4)]],
    constant uint &elevation_base_offset [[buffer(5)]],
    constant uint &vertex_count [[buffer(6)]],
    constant uint &tile_count [[buffer(7)]],
    constant uint &no_data_offset [[buffer(8)]],
    constant uint &destination_stride [[buffer(9)]],
    uint2 output_index [[thread_position_in_grid]]
) {
  if (output_index.x >= vertex_count || output_index.y >= tile_count) {
    return;
  }

  device const uchar *record = source_records + output_index.y * source_record_stride;
  device const int *base = reinterpret_cast<device const int *>(record + elevation_base_offset);
  device const ushort *vertices = reinterpret_cast<device const ushort *>(record + vertex_offset);
  const uint slot = destination_slots[output_index.y];
  destination[slot * destination_stride + output_index.x] =
      (*reinterpret_cast<device const uint *>(record + no_data_offset) &&
       vertices[output_index.x] == 0U)
          ? NAN
          : (float(*base) + float(vertices[output_index.x])) * 0.1F;
}

/// Shared traversal specialized at compile time for Float32 or uint16 terrain.
template <typename Sample, bool shadow_trace>
inline float trace_tile_frontier_impl(
    device const Sample *mipmap,
    device const Sample *vertices,
    int base_decimeters,
    RayDirection ray,
    RayWorkItem input,
    device const ResidentTile *tiles,
    RaytraceParameters params,
    device float *distances,
    device float *elevations,
    device uint *surface_gradients,
    device float *num_steps,
    device float *num_evaluations,
    float3 ray_origin,
    device uchar *visibility
) {
  const ResidentTile resident_tile = tiles[input.slot];
  const float tile_x_min = resident_tile.tile_x_min;
  const float tile_y_min = resident_tile.tile_y_min;
  if (input.ray_index >= params.ray_count) {
    return INFINITY;
  }
  const uint output_index = input.ray_index;

  // Slots retain an LOD-1 stride, while metadata describes the coarser
  // representation stored at the beginning of this slot.
  uint num_cell = 0U;
  uint num_levels = 0U;
  float cell_size = 0.0F;
  uint lod_mipmap_value_count = 0U;
  if (!resident_lod_geometry(
          resident_tile,
          params,
          num_cell,
          num_levels,
          cell_size,
          lod_mipmap_value_count
      )) {
    return INFINITY;
  }

  // Get ray parameters. Horizontal directions use the compass convention:
  // x is eastward, y is northward, and `dz` is the vertical slope.
  const float4 horizontal_direction = float4(ray.x, ray.y, ray.inverse_x, ray.inverse_y);
  const float2 direction = horizontal_direction.xy;
  const int stepx = int(direction.x > 0.0F) - int(direction.x < 0.0F);
  const int stepy = int(direction.y > 0.0F) - int(direction.y < 0.0F);
  const float delta = cell_size;
  const float inverse_delta = 1.0F / delta;
  const float dtx = stepx == 0 ? INFINITY : delta * fabs(horizontal_direction.z);
  const float dty = stepy == 0 ? INFINITY : delta * fabs(horizontal_direction.w);
  const float dz = ray.slope;
  const float observer_elevation = ray_origin.z;
  const float curvature = params.curvature_coefficient;
  const float stationary_distance =
      curvature > 0.0F ? -dz / (2.0F * curvature) : (dz >= 0.0F ? -INFINITY : INFINITY);
  const int n = int(num_cell);

  // The observer tile begins at level 1; incoming tiles begin at their
  // coarsest maximum so clear terrain can be rejected immediately.
  uint level = clamp(input.start_level, 1U, num_levels);
  uint scale = 1 << (level - 1);
  // Host-created items start at either the full level-1 field or the final
  // one-value level, whose flattened offsets are known without rebuilding the
  // intervening geometric series in every ray.
  uint offset = level == 1U ? 0U : lod_mipmap_value_count - 1U;

  // The exact near boundary of the active DDA segment. It remains separate
  // from points nudged only to assign deterministic cell ownership.
  float t_start = input.entry_distance;

  // The local observer is exactly at (0, 0), which may lie on one or both
  // shared cell boundaries. Nudge only the coordinate used for ownership so a
  // south/west ray starts in its forward cell; all DDA distances still use the
  // exact observer position and therefore retain t = 0 as their geometry.
  const float boundary_nudge = 1e-3F * delta;
  const float x_entry = ray_origin.x + t_start * direction.x;
  const float y_entry = ray_origin.y + t_start * direction.y;
  const float x_classify = stepx == 0 ? x_entry : x_entry + copysign(boundary_nudge, direction.x);
  const float y_classify = stepy == 0 ? y_entry : y_entry + copysign(boundary_nudge, direction.y);
  // An observer on a tile edge may point immediately into its neighbour.
  // Hand off at t=0 instead of clamping an outside ray into this tile's cells.
  if (t_start == 0.0F && (x_classify < tile_x_min || x_classify >= tile_x_min + float(n) * delta ||
                          y_classify < tile_y_min || y_classify >= tile_y_min + float(n) * delta))
    return 0.0F;
  int i = clamp(int(floor((y_classify - tile_y_min) * inverse_delta)), 0, n - 1);
  int j = clamp(int(floor((x_classify - tile_x_min) * inverse_delta)), 0, n - 1);

  // align to the correct level boundary
  i = align_to_level(i, scale);
  j = align_to_level(j, scale);

  // Cell-traversal distances: `tx` and `ty` are the distances to the next
  // vertical and horizontal cell boundary respectively.
  float ty =
      stepy == 0
          ? INFINITY
          : stepy * ((tile_y_min - ray_origin.y) * inverse_delta + i + (stepy > 0) * scale) * dty;
  float tx =
      stepx == 0
          ? INFINITY
          : stepx * ((tile_x_min - ray_origin.x) * inverse_delta + j + (stepx > 0) * scale) * dtx;
  // A coarse incoming segment can begin inside the other axis's aligned
  // block. Reposition both so neither moves behind the true hand-off.
  if (stepy != 0) {
    ty = next_boundary_after(ty, t_start, scale * dty);
  }
  if (stepx != 0) {
    tx = next_boundary_after(tx, t_start, scale * dtx);
  }

  const uint vertex_count = num_cell + 1U;
  const float tile_exit = tile_exit_distance(
      tile_x_min - ray_origin.x,
      tile_y_min - ray_origin.y,
      delta,
      num_cell,
      horizontal_direction,
      t_start
  );
  const float segment_limit = min(tile_exit, params.max_distance);

  // A resident tile's manifest maximum can reject the complete segment before
  // any mipmap or vertex data is touched. The one-metre margin matches
  // successor culling and keeps quantization/rounding conservative.
  if (isfinite(resident_tile.maximum_elevation) && tile_exit <= params.max_distance &&
      minimum_curved_ray_elevation(
          observer_elevation,
          dz,
          curvature,
          stationary_distance,
          t_start,
          tile_exit
      ) > resident_tile.maximum_elevation + 1.0F) {
    return tile_exit;
  }

  // Step the ray across the mipmap cell-by-cell until we go off the edge or find an internal
  // collision
  while (i >= 0 && j >= 0 && i < n && j < n) {
    // shift t to the edge of the next cell/the edge of the tile/max distance, whichever is closest
    const float t_exit = min(tx, ty);
    const float interval_end = min(t_exit, segment_limit);

    // Derive the exact near edge of this DDA block. The curved ray is convex,
    // so its minimum can occur at either edge or at its stationary point.
    float interval_start = t_start;
    if (stepx != 0) {
      interval_start = max(interval_start, tx - scale * dtx);
    }
    if (stepy != 0) {
      interval_start = max(interval_start, ty - scale * dty);
    }
    interval_start = min(interval_start, interval_end);

    // Once this ray is both rising and above the complete catalogue's upper
    // bound, curvature guarantees it can never intersect a later cell or tile.
    if (above_global_terrain(
            observer_elevation,
            dz,
            curvature,
            stationary_distance,
            interval_start,
            params.global_maximum_elevation
        )) {
      return INFINITY;
    }
    const float z = minimum_curved_ray_elevation(
        observer_elevation,
        dz,
        curvature,
        stationary_distance,
        interval_start,
        interval_end
    );

    // find the index of the relevant cell (correct level and location) inside the flattened mipmap
    const uint level_side = mipmap_level_side(num_cell, level);
    const uint cell_index =
        offset + (uint(i) >> (level - 1U)) * level_side + (uint(j) >> (level - 1U));

    // Missing cells contain no terrain; advance the DDA through them normally.
    if (z <= sample_elevation(mipmap[cell_index], base_decimeters) &&
        (level != 1U ||
         valid_cell(vertices, vertex_count, uint(j), uint(i), resident_tile.no_data))) {
      if (level == 1) {
        // Finest level collision check. Restrict the bilinear root search to this cell's
        // actual DDA interval, including its near boundary.
        Collision collision;

        if (use_bilinear_collisions) {
          collision = bilinear_collision(
              vertices,
              vertex_count,
              base_decimeters,
              tile_x_min + float(j) * delta - ray_origin.x,
              tile_y_min + float(i) * delta - ray_origin.y,
              inverse_delta,
              uint(i),
              uint(j),
              observer_elevation,
              direction,
              dz,
              curvature,
              interval_start,
              interval_end
          );
        } else {
          collision = triangle_collision(
              vertices,
              vertex_count,
              base_decimeters,
              tile_x_min + float(j) * delta - ray_origin.x,
              tile_y_min + float(i) * delta - ray_origin.y,
              delta,
              uint(i),
              uint(j),
              observer_elevation,
              direction,
              dz,
              curvature,
              interval_start
          );
        }

        // Store debugging details if requested
        if (!shadow_trace && store_debugging_info) {
          num_evaluations[output_index] += 1.0F;
        }

        // The triangle solver is not given the far interval boundary. Reject
        // roots beyond this cell or the configured range, matching BVH queries.
        const float range_guard = 8.0F * FLT_EPSILON * max(1.0F, interval_end);
        if (collision.hit && collision.distance >= interval_start - range_guard &&
            collision.distance <= interval_end + range_guard) {
          collision.distance = clamp(collision.distance, interval_start, interval_end);
          if (shadow_trace) {
            visibility[output_index] = 0U;
          } else {
            distances[output_index] = collision.distance;
          }
          if (!shadow_trace && store_collision_elevations) {
            elevations[output_index] =
                curved_ray_elevation(observer_elevation, dz, curvature, collision.distance);
          }
          if (!shadow_trace && compute_surface_gradients) {
            if (use_c1_normals && (i >= 1 && j >= 1 && i < n - 1 && j < n - 1) &&
                valid_normal_stencil(
                    vertices,
                    vertex_count,
                    uint(j),
                    uint(i),
                    resident_tile.no_data
                )) {
              surface_gradients[output_index] = interpolated_packed_surface_gradients(
                  vertices,
                  vertex_count,
                  base_decimeters,
                  tile_x_min + float(j) * delta - ray_origin.x,
                  tile_y_min + float(i) * delta - ray_origin.y,
                  inverse_delta,
                  uint(i),
                  uint(j),
                  direction,
                  collision.distance
              );
            } else {
              // TODO: give option to do better normals
              if (use_bilinear_collisions) {
                surface_gradients[output_index] = bilinear_packed_surface_gradients(
                    vertices,
                    vertex_count,
                    base_decimeters,
                    tile_x_min + float(j) * delta - ray_origin.x,
                    tile_y_min + float(i) * delta - ray_origin.y,
                    inverse_delta,
                    uint(i),
                    uint(j),
                    direction,
                    collision.distance
                );
              } else {
                surface_gradients[output_index] = triangle_packed_surface_gradients(
                    vertices,
                    vertex_count,
                    base_decimeters,
                    tile_x_min + float(j) * delta - ray_origin.x,
                    tile_y_min + float(i) * delta - ray_origin.y,
                    inverse_delta,
                    uint(i),
                    uint(j),
                    direction,
                    collision.distance
                );
              }
            }
          }
          return INFINITY;
        }
      } else {
        // There might be a real collision inside this coarse cell. Descend to
        // the child containing the ray at the coarse cell's true near edge.
        const float cell_entry = interval_start;

        // Reclassify a point just inside the child. The nudge controls shared
        // boundary ownership only; `cell_entry` remains the exact geometry.
        float cell_x = ray_origin.x + cell_entry * direction.x;
        float cell_y = ray_origin.y + cell_entry * direction.y;
        const float cell_nudge =
            max(1e-3F * delta, 8.0F * FLT_EPSILON * max(1.0F, max(fabs(cell_x), fabs(cell_y))));
        if (stepx != 0) {
          cell_x += copysign(cell_nudge, direction.x);
        }
        if (stepy != 0) {
          cell_y += copysign(cell_nudge, direction.y);
        }
        i = clamp(int(floor((cell_y - tile_y_min) * inverse_delta)), 0, n - 1);
        j = clamp(int(floor((cell_x - tile_x_min) * inverse_delta)), 0, n - 1);

        // Fine indices remain in level-1 coordinates. Returning to a child
        // level requires their lower-left child-cell alignment before the
        // DDA timers are rebuilt; retaining an arbitrary fine index would
        // make a scale-sized timer skip an internal child boundary.
        t_start = cell_entry;
        level -= 1U;
        scale /= 2U;
        i = align_to_level(i, scale);
        j = align_to_level(j, scale);

        // The preceding level occupies `child_side` squared entries directly
        // before this one, so move backward without rebuilding the offset.
        const uint child_side = mipmap_level_side(num_cell, level);
        offset -= child_side * child_side;
        ty = stepy == 0
                 ? INFINITY
                 : stepy * ((tile_y_min - ray_origin.y) * inverse_delta + i + (stepy > 0) * scale) *
                       dty;
        tx = stepx == 0
                 ? INFINITY
                 : stepx * ((tile_x_min - ray_origin.x) * inverse_delta + j + (stepx > 0) * scale) *
                       dtx;

        // An entry through only one edge of a coarse cell may leave the
        // other axis part-way across the selected child. Advance its timer
        // rather than allowing a stale aligned boundary to move backward.
        if (stepy != 0) {
          ty = next_boundary_after(ty, t_start, scale * dty);
        }
        if (stepx != 0) {
          tx = next_boundary_after(tx, t_start, scale * dtx);
        }

        continue;
      }
    }

    // The ray reached either the far edge of this tile or the configured
    // global range. Report the distinction explicitly to the CPU scheduler.
    if (t_exit >= segment_limit) {
      if (tile_exit <= params.max_distance) {
        return tile_exit;
      }
      return INFINITY;
    }

    // Go up to a coarser level whenever possible
    if (level < num_levels) {
      if (ty < tx) {
        if (at_level_boundary(i, stepy, scale)) {
          // Crossing a Y boundary joins two vertically adjacent blocks. The
          // X timer must therefore be adjusted from the X cell's sibling.
          tx += offset_jump(j, stepx, scale, dtx);

          // The current level immediately precedes the coarser level in the
          // flat buffer, so advance by its square element count.
          offset += level_side * level_side;
          level += 1;
          scale *= 2;
        }
      } else {
        if (at_level_boundary(j, stepx, scale)) {
          // Crossing an X boundary joins two horizontally adjacent blocks.
          // Adjust the Y timer from the Y cell's sibling before coarsening.
          ty += offset_jump(i, stepy, scale, dty);

          // The level transition is identical regardless of the stepped axis.
          offset += level_side * level_side;
          level += 1;
          scale *= 2;
        }
      }
    }

    // Step forwards
    if (ty < tx) {
      ty += scale * dty;
      i += scale * stepy;
    } else {
      tx += scale * dtx;
      j += scale * stepx;
    }
    // Store debugging details if requested
    if (!shadow_trace && store_debugging_info) {
      num_steps[output_index] += 1.0F;
    }
  }
  // The DDA normally returns through the segment-limit check above. Retain a
  // defensive continuation for an unexpected fine-cell exit discrepancy.
  if (tile_exit <= params.max_distance) {
    return tile_exit;
  }
  return INFINITY;
}

/// Trace one arbitrary output ray through Float32 resident terrain.
kernel void trace_tile_frontier(
    device const float *mipmap_atlas [[buffer(0)]],
    device const float *vertex_atlas [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device const RayWorkItem *work_items [[buffer(3)]],
    device const ResidentTile *tiles [[buffer(4)]],
    constant RaytraceParameters &shared_parameters [[buffer(5)]],
    constant uint &mipmap_value_count [[buffer(6)]],
    device float *distances [[buffer(7)]],
    device float *elevations [[buffer(8)]],
    device float *continuations [[buffer(9)]],
    device uint *surface_gradients [[buffer(11)]],
    device float *num_steps [[buffer(12)]],
    device float *num_evaluations [[buffer(13)]],
    uint work_index [[thread_position_in_grid]]
) {
  const RayWorkItem input = work_items[work_index];
  const uint num_cell = mipmap_finest_side(shared_parameters.num_levels);
  const uint vertex_value_count = (num_cell + 1U) * (num_cell + 1U);
  continuations[work_index] = trace_tile_frontier_impl<float, false>(
      mipmap_atlas + input.slot * mipmap_value_count,
      vertex_atlas + input.slot * vertex_value_count,
      0,
      rays[input.ray_index],
      input,
      tiles,
      shared_parameters,
      distances,
      elevations,
      surface_gradients,
      num_steps,
      num_evaluations,
      float3(0.0F, 0.0F, shared_parameters.observer_elevation),
      nullptr
  );
}

/// Trace one arbitrary output ray while decoding uint16 elevations on demand.
kernel void trace_tile_frontier_quantized(
    device const ushort *mipmap_atlas [[buffer(0)]],
    device const uchar *vertex_records [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device const RayWorkItem *work_items [[buffer(3)]],
    device const ResidentTile *tiles [[buffer(4)]],
    constant RaytraceParameters &shared_parameters [[buffer(5)]],
    constant uint &mipmap_value_count [[buffer(6)]],
    device float *distances [[buffer(7)]],
    device float *elevations [[buffer(8)]],
    device float *continuations [[buffer(9)]],
    constant QuantizedTerrainLayout &layout [[buffer(10)]],
    device uint *surface_gradients [[buffer(11)]],
    device float *num_steps [[buffer(12)]],
    device float *num_evaluations [[buffer(13)]],
    uint work_index [[thread_position_in_grid]]
) {
  const RayWorkItem input = work_items[work_index];
  device const uchar *record = vertex_records + input.slot * layout.record_stride;
  device const int *base =
      reinterpret_cast<device const int *>(record + layout.elevation_base_offset);
  device const ushort *vertices =
      reinterpret_cast<device const ushort *>(record + layout.vertex_offset);
  continuations[work_index] = trace_tile_frontier_impl<ushort, false>(
      mipmap_atlas + input.slot * mipmap_value_count,
      vertices,
      *base,
      rays[input.ray_index],
      input,
      tiles,
      shared_parameters,
      distances,
      elevations,
      surface_gradients,
      num_steps,
      num_evaluations,
      float3(0.0F, 0.0F, shared_parameters.observer_elevation),
      nullptr
  );
}

/// Build the compact initial frontier for collision points which can receive
/// direct sunlight. Pixels not needing a query remain visible; presentation
/// already rejects sky and back-facing terrain independently.
kernel void initialise_shadow_rays(
    device const RayDirection *camera_rays [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    device const float *elevations [[buffer(2)]],
    device const uint *surface_gradients [[buffer(3)]],
    constant ShadowTraceParameters &params [[buffer(4)]],
    device ShadowRay *shadow_rays [[buffer(5)]],
    device uchar *visibility [[buffer(6)]],
    device DeferredRayWork *deferred_items [[buffer(7)]],
    device atomic_uint *deferred_count [[buffer(8)]],
    device const CatalogueTileHashEntry *catalogue_hash [[buffer(9)]],
    uint2 position [[thread_position_in_grid]]
) {
  const uint ray_index = position.y * params.trace.image_width + position.x;
  if (ray_index >= params.trace.ray_count) {
    return;
  }
  visibility[ray_index] = 1U;
  const float distance = distances[ray_index];
  if (!(distance > 0.0F) || !isfinite(distance)) {
    return;
  }

  const float2 gradient = float2(as_type<half2>(surface_gradients[ray_index]));
  const float3 normal = normalize(float3(-gradient.x, -gradient.y, 1.0F));
  const RayDirection sun = params.direction;
  if (dot(normal, normalize(float3(sun.x, sun.y, sun.slope))) <= 0.0F) {
    return;
  }

  // Move along the ray by a fraction of a DEM cell, and slightly upward, so
  // the collision's own bilinear patch cannot immediately shadow itself.
  const float horizontal_bias = max(0.1F, 0.05F * params.trace.cell_size);
  const float vertical_bias = max(0.05F, 0.005F * params.trace.cell_size);
  const RayDirection camera = camera_rays[ray_index];
  const float x = distance * camera.x + horizontal_bias * sun.x;
  const float y = distance * camera.y + horizontal_bias * sun.y;
  const float z = elevations[ray_index] + horizontal_bias * sun.slope + vertical_bias;
  shadow_rays[ray_index].origin = float4(x, y, z, 0.0F);

  long column = long(floor((x - params.grid_x_min) / params.tile_width));
  long row = long(floor((params.grid_y_max - y) / params.tile_width));
  float entry = 0.0F;
  while (entry < params.trace.max_distance) {
    device const CatalogueTileHashEntry *source =
        lookup_catalogue_tile(catalogue_hash, params.catalogue_hash_capacity, row, column);
    if (source != nullptr) {
      const uint output = atomic_fetch_add_explicit(deferred_count, 1U, memory_order_relaxed);
      if (output < params.trace.ray_count)
        deferred_items[output] = {ray_index, source->source_index, entry};
      return;
    }
    // The self-shadow bias can place the origin in an empty neighbouring tile.
    // Find the next known source rather than declaring the entire ray clear.
    const float tile_x = params.grid_x_min + float(column) * params.tile_width;
    const float tile_y = params.grid_y_max - float(row + 1L) * params.tile_width;
    const float next = tile_exit_distance(
        tile_x - x,
        tile_y - y,
        params.tile_width,
        1U,
        float4(sun.x, sun.y, sun.inverse_x, sun.inverse_y),
        entry
    );
    if (!isfinite(next) || !(next > entry) || next >= params.trace.max_distance)
      return;
    const float nudge =
        max(1e-3F * params.trace.cell_size,
            8.0F * FLT_EPSILON * max(1.0F, max(abs(x), abs(y)) + next));
    const float next_x = x + next * sun.x + (sun.x == 0 ? 0 : copysign(nudge, sun.x));
    const float next_y = y + next * sun.y + (sun.y == 0 ? 0 : copysign(nudge, sun.y));
    column = long(floor((next_x - params.grid_x_min) / params.tile_width));
    row = long(floor((params.grid_y_max - next_y) / params.tile_width));
    entry = next;
  }
}

template <typename Sample>
inline void trace_shadow_frontier_impl(
    device const Sample *mipmap_atlas,
    device const Sample *vertices,
    int base_decimeters,
    device const RayWorkItem *work_items,
    device const ResidentTile *tiles,
    constant ShadowTraceParameters &params,
    device const ShadowRay *shadow_rays,
    device uchar *visibility,
    device float *continuations,
    uint work_index
) {
  // Shadow rays share the primary DDA and bilinear intersection code. The
  // template flag changes a confirmed collision into an any-hit visibility
  // update and suppresses primary collision products.
  const RayWorkItem input = work_items[work_index];
  continuations[work_index] = trace_tile_frontier_impl<Sample, true>(
      mipmap_atlas,
      vertices,
      base_decimeters,
      params.direction,
      input,
      tiles,
      params.trace,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      shadow_rays[input.ray_index].origin.xyz,
      visibility
  );
}

/// Trace active hard-shadow segments through Float32 resident terrain.
kernel void trace_shadow_tile_frontier(
    device const float *mipmap_atlas [[buffer(0)]],
    device const float *vertex_atlas [[buffer(1)]],
    device const RayWorkItem *work_items [[buffer(2)]],
    device const ResidentTile *tiles [[buffer(3)]],
    constant ShadowTraceParameters &params [[buffer(4)]],
    constant uint &mipmap_value_count [[buffer(5)]],
    device const ShadowRay *shadow_rays [[buffer(6)]],
    device uchar *visibility [[buffer(7)]],
    device float *continuations [[buffer(8)]],
    uint work_index [[thread_position_in_grid]]
) {
  const RayWorkItem input = work_items[work_index];
  const uint side = mipmap_finest_side(params.trace.num_levels) + 1U;
  trace_shadow_frontier_impl(
      mipmap_atlas + input.slot * mipmap_value_count,
      vertex_atlas + input.slot * side * side,
      0,
      work_items,
      tiles,
      params,
      shadow_rays,
      visibility,
      continuations,
      work_index
  );
}

/// Trace active hard-shadow segments through retained uint16 terrain.
kernel void trace_shadow_tile_frontier_quantized(
    device const ushort *mipmap_atlas [[buffer(0)]],
    device const uchar *vertex_records [[buffer(1)]],
    device const RayWorkItem *work_items [[buffer(2)]],
    device const ResidentTile *tiles [[buffer(3)]],
    constant ShadowTraceParameters &params [[buffer(4)]],
    constant uint &mipmap_value_count [[buffer(5)]],
    device const ShadowRay *shadow_rays [[buffer(6)]],
    device uchar *visibility [[buffer(7)]],
    device float *continuations [[buffer(8)]],
    constant QuantizedTerrainLayout &layout [[buffer(9)]],
    uint work_index [[thread_position_in_grid]]
) {
  const RayWorkItem input = work_items[work_index];
  device const uchar *record = vertex_records + input.slot * layout.record_stride;
  device const int *base =
      reinterpret_cast<device const int *>(record + layout.elevation_base_offset);
  device const ushort *vertices =
      reinterpret_cast<device const ushort *>(record + layout.vertex_offset);
  trace_shadow_frontier_impl(
      mipmap_atlas + input.slot * mipmap_value_count,
      vertices,
      *base,
      work_items,
      tiles,
      params,
      shadow_rays,
      visibility,
      continuations,
      work_index
  );
}

/// Cull clear successors and return every required ray to a host source bucket.
kernel void emit_tile_frontier(
    device const RayWorkItem *active_items [[buffer(0)]],
    device const ResidentTile *tiles [[buffer(1)]],
    device const RayDirection *rays [[buffer(2)]],
    device const float *continuations [[buffer(3)]],
    constant RaytraceParameters &shared_parameters [[buffer(4)]],
    constant uint &frontier_capacity [[buffer(5)]],
    device DeferredRayWork *deferred_items [[buffer(6)]],
    device atomic_uint *deferred_count [[buffer(7)]],
    device const CatalogueTileHashEntry *catalogue_hash [[buffer(8)]],
    constant uint &catalogue_hash_capacity [[buffer(9)]],
    device atomic_uint *local_skip_count [[buffer(10)]],
    device atomic_uint *global_skip_count [[buffer(11)]],
    uint work_index [[thread_position_in_grid]]
) {
  float entry_distance = continuations[work_index];
  if (!isfinite(entry_distance)) {
    return;
  }
  const RayWorkItem active = active_items[work_index];
  const ResidentTile current_tile = tiles[active.slot];
  const RaytraceParameters params = shared_parameters;
  const RayDirection ray = rays[active.ray_index];
  const float2 direction = float2(ray.x, ray.y);
  const float tile_width = float(mipmap_finest_side(params.num_levels)) * params.cell_size;
  float tile_x_min = current_tile.tile_x_min;
  float tile_y_min = current_tile.tile_y_min;
  long tile_row = current_tile.row;
  long tile_column = current_tile.column;
  uint local_skips = 0U;
  uint global_skips = 0U;

  for (;;) {
    // The outward nudge deterministically selects a neighbour at shared edges
    // and corners while retaining the exact distance for its trace segment.
    float x = entry_distance * direction.x;
    float y = entry_distance * direction.y;
    const float nudge =
        max(1e-3F * params.cell_size, 8.0F * FLT_EPSILON * max(1.0F, max(fabs(x), fabs(y))));
    if (direction.x != 0.0F) {
      x += copysign(nudge, direction.x);
    }
    if (direction.y != 0.0F) {
      y += copysign(nudge, direction.y);
    }
    const long row_offset = y < tile_y_min ? 1L : y >= tile_y_min + tile_width ? -1L : 0L;
    const long column_offset = x < tile_x_min ? -1L : x >= tile_x_min + tile_width ? 1L : 0L;
    if (row_offset == 0L && column_offset == 0L) {
      break;
    }
    tile_row += row_offset;
    tile_column += column_offset;
    tile_x_min += float(column_offset) * tile_width;
    tile_y_min -= float(row_offset) * tile_width;

    device const CatalogueTileHashEntry *source =
        lookup_catalogue_tile(catalogue_hash, catalogue_hash_capacity, tile_row, tile_column);
    const float elevation_at_entry = curved_ray_elevation(
        params.observer_elevation,
        ray.slope,
        params.curvature_coefficient,
        entry_distance
    );
    const float elevation_derivative =
        ray.slope + 2.0F * params.curvature_coefficient * entry_distance;
    if (isfinite(params.global_maximum_elevation) && elevation_derivative >= 0.0F &&
        elevation_at_entry > params.global_maximum_elevation + 1.0F) {
      global_skips++;
      break;
    }

    const float exit_distance = tile_exit_distance(
        tile_x_min,
        tile_y_min,
        params.cell_size,
        mipmap_finest_side(params.num_levels),
        float4(ray.x, ray.y, ray.inverse_x, ray.inverse_y),
        entry_distance
    );
    // No catalogue entry means confirmed empty space, not a loading request.
    // Keep walking the grid until known terrain or the configured range.
    if (source == nullptr) {
      if (!isfinite(exit_distance) || !(exit_distance > entry_distance) ||
          exit_distance >= params.max_distance)
        break;
      entry_distance = exit_distance;
      continue;
    }
    const float stationary_distance = params.curvature_coefficient > 0.0F
                                          ? -ray.slope / (2.0F * params.curvature_coefficient)
                                          : (ray.slope >= 0.0F ? -INFINITY : INFINITY);
    if (isfinite(source->maximum_elevation) && isfinite(exit_distance) &&
        minimum_curved_ray_elevation(
            params.observer_elevation,
            ray.slope,
            params.curvature_coefficient,
            stationary_distance,
            entry_distance,
            exit_distance
        ) > source->maximum_elevation + 1.0F) {
      local_skips++;
      entry_distance = exit_distance;
      if (entry_distance >= params.max_distance) {
        break;
      }
      continue;
    }

    const uint deferred_index = atomic_fetch_add_explicit(deferred_count, 1U, memory_order_relaxed);
    if (deferred_index < frontier_capacity) {
      deferred_items[deferred_index] = {
          active.ray_index,
          source->source_index,
          entry_distance,
      };
    }
    break;
  }
  if (local_skips != 0U) {
    atomic_fetch_add_explicit(local_skip_count, local_skips, memory_order_relaxed);
  }
  if (global_skips != 0U) {
    atomic_fetch_add_explicit(global_skip_count, global_skips, memory_order_relaxed);
  }
}

/// Emit shadow-ray successors. This is the primary continuation logic with
/// every position and tile boundary translated by the ray's collision origin.
kernel void emit_shadow_tile_frontier(
    device const RayWorkItem *active_items [[buffer(0)]],
    device const ResidentTile *tiles [[buffer(1)]],
    device const float *continuations [[buffer(2)]],
    constant ShadowTraceParameters &shared_parameters [[buffer(3)]],
    constant uint &frontier_capacity [[buffer(4)]],
    device DeferredRayWork *deferred_items [[buffer(5)]],
    device atomic_uint *deferred_count [[buffer(6)]],
    device const CatalogueTileHashEntry *catalogue_hash [[buffer(7)]],
    device const ShadowRay *shadow_rays [[buffer(8)]],
    uint work_index [[thread_position_in_grid]]
) {
  float entry_distance = continuations[work_index];
  if (!isfinite(entry_distance)) {
    return;
  }
  const RayWorkItem active = active_items[work_index];
  const ResidentTile current_tile = tiles[active.slot];
  const ShadowTraceParameters shadow = shared_parameters;
  const RaytraceParameters params = shadow.trace;
  const RayDirection ray = shadow.direction;
  const float3 origin = shadow_rays[active.ray_index].origin.xyz;
  const float2 direction = float2(ray.x, ray.y);
  const float tile_width = float(mipmap_finest_side(params.num_levels)) * params.cell_size;
  float tile_x_min = current_tile.tile_x_min;
  float tile_y_min = current_tile.tile_y_min;
  long tile_row = current_tile.row;
  long tile_column = current_tile.column;

  for (;;) {
    float x = origin.x + entry_distance * direction.x;
    float y = origin.y + entry_distance * direction.y;
    const float nudge =
        max(1e-3F * params.cell_size, 8.0F * FLT_EPSILON * max(1.0F, max(fabs(x), fabs(y))));
    if (direction.x != 0.0F) {
      x += copysign(nudge, direction.x);
    }
    if (direction.y != 0.0F) {
      y += copysign(nudge, direction.y);
    }
    const long row_offset = y < tile_y_min ? 1L : y >= tile_y_min + tile_width ? -1L : 0L;
    const long column_offset = x < tile_x_min ? -1L : x >= tile_x_min + tile_width ? 1L : 0L;
    if (row_offset == 0L && column_offset == 0L) {
      break;
    }
    tile_row += row_offset;
    tile_column += column_offset;
    tile_x_min += float(column_offset) * tile_width;
    tile_y_min -= float(row_offset) * tile_width;

    device const CatalogueTileHashEntry *source = lookup_catalogue_tile(
        catalogue_hash,
        shadow.catalogue_hash_capacity,
        tile_row,
        tile_column
    );
    const float elevation_at_entry =
        curved_ray_elevation(origin.z, ray.slope, params.curvature_coefficient, entry_distance);
    const float elevation_derivative =
        ray.slope + 2.0F * params.curvature_coefficient * entry_distance;
    if (isfinite(params.global_maximum_elevation) && elevation_derivative >= 0.0F &&
        elevation_at_entry > params.global_maximum_elevation + 1.0F) {
      break;
    }

    const float exit_distance = tile_exit_distance(
        tile_x_min - origin.x,
        tile_y_min - origin.y,
        params.cell_size,
        mipmap_finest_side(params.num_levels),
        float4(ray.x, ray.y, ray.inverse_x, ray.inverse_y),
        entry_distance
    );
    // No catalogue entry means confirmed empty space, not a loading request.
    // Keep walking the grid until known terrain or the configured range.
    if (source == nullptr) {
      if (!isfinite(exit_distance) || !(exit_distance > entry_distance) ||
          exit_distance >= params.max_distance)
        break;
      entry_distance = exit_distance;
      continue;
    }
    const float stationary_distance = params.curvature_coefficient > 0.0F
                                          ? -ray.slope / (2.0F * params.curvature_coefficient)
                                          : (ray.slope >= 0.0F ? -INFINITY : INFINITY);
    if (isfinite(source->maximum_elevation) && isfinite(exit_distance) &&
        minimum_curved_ray_elevation(
            origin.z,
            ray.slope,
            params.curvature_coefficient,
            stationary_distance,
            entry_distance,
            exit_distance
        ) > source->maximum_elevation + 1.0F) {
      entry_distance = exit_distance;
      if (entry_distance >= params.max_distance) {
        break;
      }
      continue;
    }

    const uint deferred_index = atomic_fetch_add_explicit(deferred_count, 1U, memory_order_relaxed);
    if (deferred_index < frontier_capacity) {
      deferred_items[deferred_index] = {active.ray_index, source->source_index, entry_distance};
    }
    break;
  }
}
