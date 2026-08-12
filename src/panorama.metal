#include <metal_stdlib>

// Metal Shading Language provides GPU-specific types and functions in this
// namespace, including `uint`, `device`, and the `kernel` entry-point keyword.
using namespace metal;

// The terrain-tracing ABI. This scalar-only structure intentionally mirrors
// RaytraceParameters in raytrace_setup.mm; all projected tile coordinates have
// already been rebased around the observer before upload.
struct RaytraceParameters {
  float tile_x_min;
  float tile_y_min;
  float cell_size;
  float observer_elevation;
  uint num_levels;
  uint num_cell;
  uint num_azimuth;
  uint num_polar;
  float max_distance;
};

struct Collision {
  bool hit;
  float distance;
};

/// Return a root in the current DDA interval, allowing a small tolerance at a
/// shared cell edge before clamping it back into this patch.
inline bool valid_root(
    float local_t,
    float interval_length,
    float sx0,
    float sy0,
    float sx1,
    float sy1,
    float coordinate_tolerance,
    float t_tolerance,
    thread float &clamped_t
) {
  if (local_t < -t_tolerance || local_t > interval_length + t_tolerance) {
    return false;
  }
  clamped_t = clamp(local_t, 0.0F, interval_length);
  const float sx = sx0 + sx1 * clamped_t;
  const float sy = sy0 + sy1 * clamped_t;
  return sx >= -coordinate_tolerance && sx <= 1.0F + coordinate_tolerance &&
         sy >= -coordinate_tolerance && sy <= 1.0F + coordinate_tolerance;
}

/// Solve the exact intersection with one bilinear level-0 terrain patch. The
/// local parameter begins at t_entry to avoid cancellation on distant cells.
inline Collision bilinear_collision(
    device const float *vertices,
    uint vertex_count,
    float cell_x,
    float cell_y,
    float delta,
    uint i,
    uint j,
    float3 ray_origin,
    float3 ray_direction,
    float t_entry,
    float t_exit
) {
  constexpr float kPolynomialEpsilon = 1e-12F;
  const uint lower_left = i * vertex_count + j;
  const float z00 = vertices[lower_left];
  const float z01 = vertices[lower_left + 1U];
  const float z10 = vertices[lower_left + vertex_count];
  const float z11 = vertices[lower_left + vertex_count + 1U];
  const float interval_length = t_exit - t_entry;

  const float sx0 = (ray_origin.x + t_entry * ray_direction.x - cell_x) / delta;
  const float sy0 = (ray_origin.y + t_entry * ray_direction.y - cell_y) / delta;
  const float sx1 = ray_direction.x / delta;
  const float sy1 = ray_direction.y / delta;
  const float dzdx = z01 - z00;
  const float dzdy = z10 - z00;
  const float twist = z11 - z10 - z01 + z00;

  const float a = -twist * sx1 * sy1;
  const float b = ray_direction.z - dzdx * sx1 - dzdy * sy1 - twist * (sx0 * sy1 + sx1 * sy0);
  const float c =
      ray_origin.z + t_entry * ray_direction.z - z00 - dzdx * sx0 - dzdy * sy0 - twist * sx0 * sy0;

  const float cell_speed = max(max(fabs(sx1), fabs(sy1)), 1e-12F);
  const float direction_error = FLT_EPSILON * max(fabs(t_entry), fabs(t_exit)) * cell_speed;
  const float coordinate_tolerance = max(5e-5F, max(128.0F * direction_error, 0.05F / delta));
  const float t_tolerance = coordinate_tolerance / cell_speed;

  // A near-corner DDA transition can leave the root in an adjacent cell. If
  // the ray is below all four vertices by this interval's end, recover the
  // intersection on the near boundary as in the Python reference.
  const auto conservative_boundary_hit = [&]() -> Collision {
    if (ray_origin.z + t_exit * ray_direction.z <= min(min(z00, z01), min(z10, z11))) {
      return {true, t_entry};
    }
    return {false, 0.0F};
  };

  float local_t = 0.0F;
  if (fabs(a) < kPolynomialEpsilon) {
    if (fabs(b) < kPolynomialEpsilon) {
      return conservative_boundary_hit();
    }
    local_t = -c / b;
    if (valid_root(
            local_t,
            interval_length,
            sx0,
            sy0,
            sx1,
            sy1,
            coordinate_tolerance,
            t_tolerance,
            local_t
        )) {
      return {true, t_entry + local_t};
    }
    return conservative_boundary_hit();
  }

  float discriminant = b * b - 4.0F * a * c;
  if (discriminant < 0.0F) {
    if (discriminant > -kPolynomialEpsilon) {
      discriminant = 0.0F;
    } else {
      return conservative_boundary_hit();
    }
  }

  const float root = sqrt(discriminant);
  float local_t0 = (-b - root) / (2.0F * a);
  float local_t1 = (-b + root) / (2.0F * a);
  if (local_t0 > local_t1) {
    const float temporary = local_t0;
    local_t0 = local_t1;
    local_t1 = temporary;
  }
  if (valid_root(
          local_t0,
          interval_length,
          sx0,
          sy0,
          sx1,
          sy1,
          coordinate_tolerance,
          t_tolerance,
          local_t
      )) {
    return {true, t_entry + local_t};
  }
  if (valid_root(
          local_t1,
          interval_length,
          sx0,
          sy0,
          sx1,
          sy1,
          coordinate_tolerance,
          t_tolerance,
          local_t
      )) {
    return {true, t_entry + local_t};
  }
  return conservative_boundary_hit();
}

/// Return whether the index is a at the boundary of a level+1 block (and would thus be permitted to
/// go up a level)
inline bool at_level_boundary(int index, int direction, uint level) {
  const uint block_size = 1 << level; // the size of a block in the level above
  const uint boundary_remainder =
      direction > 0 ? block_size - 1 : 0; // remainder at a level boundary
  return (direction != 0) && (index % block_size == boundary_remainder);
}

/// Return the offset required in the opposite component when going up from `level` to `level+1`
inline float offset_jump(int index, int direction, uint level, uint scale, float dt) {
  const int parity = (index >> (level - 1)) + (direction > 0);
  return (parity % 2) * scale * dt;
}

/// Return the row-major side length of a one-indexed mipmap level.
inline uint mipmap_level_side(uint cell_count, uint level) { return cell_count >> (level - 1U); }

/// Move a periodic DDA boundary to the first occurrence strictly after the
/// current segment entry. Refinement can begin part-way through a child cell,
/// so its nominal aligned boundary may already be behind `entry_distance`.
inline float next_boundary_after(float boundary, float entry_distance, float interval) {
  if (boundary <= entry_distance) {
    boundary += (floor((entry_distance - boundary) / interval) + 1.0F) * interval;
  }
  return boundary;
}

/// Trace one independent polar ray through one level-0 terrain tile. Level-1
/// cell maxima reject empty cells before the bounded bilinear solve reads the
/// original vertex elevations.
kernel void trace_single_tile(
    device const float *mipmap [[buffer(0)]],
    device const float *vertices [[buffer(1)]],
    device const float2 *azimuth_directions [[buffer(2)]],
    device const float *polar_slopes [[buffer(3)]],
    constant RaytraceParameters &params [[buffer(4)]],
    device float *distances [[buffer(5)]],
    device float *elevations [[buffer(6)]],
    uint2 ray_index [[thread_position_in_grid]]
) {
  // Bounds check and convert to flat indexing.
  if (ray_index.x >= params.num_azimuth || ray_index.y >= params.num_polar) {
    return;
  }
  const uint output_index = ray_index.y * params.num_azimuth + ray_index.x;

  // Get ray parameters. Horizontal directions use the compass convention:
  // x is eastward, y is northward, and `dz` is the vertical slope.
  const float2 direction = azimuth_directions[ray_index.x];
  const int stepx = int(direction.x > 0.0F) - int(direction.x < 0.0F);
  const int stepy = int(direction.y > 0.0F) - int(direction.y < 0.0F);
  const float delta = params.cell_size;
  const float dtx = stepx == 0 ? INFINITY : fabs(delta / direction.x);
  const float dty = stepy == 0 ? INFINITY : fabs(delta / direction.y);
  const float dz = polar_slopes[ray_index.y];
  const float3 ray_origin = {0.0F, 0.0F, params.observer_elevation};
  const float3 ray_direction = {direction.x, direction.y, dz};
  const int n = int(params.num_cell);

  // Decide starting level: the origin tile always starts at level 1, all subsequent tiles start at
  // the max level
  // uint level = params.num_level; // for non-origin tiles
  uint level = 1;
  uint scale = 1 << (level - 1);
  uint offset = 0;
  // The exact near boundary of the active DDA segment. It remains separate
  // from points nudged only to assign deterministic cell ownership.
  float t_start = 0.0F;

  // The local observer is exactly at (0, 0), which may lie on one or both
  // shared cell boundaries. Nudge only the coordinate used for ownership so a
  // south/west ray starts in its forward cell; all DDA distances still use the
  // exact observer position and therefore retain t = 0 as their geometry.
  const float boundary_nudge = 1e-3F * delta;
  const float x_classify = stepx == 0 ? 0.0F : copysign(boundary_nudge, direction.x);
  const float y_classify = stepy == 0 ? 0.0F : copysign(boundary_nudge, direction.y);
  int i = clamp(int(floor((y_classify - params.tile_y_min) / delta)), 0, n - 1);
  int j = clamp(int(floor((x_classify - params.tile_x_min) / delta)), 0, n - 1);

  // align to the correct level boundary
  i = (i / scale) * scale;
  j = (j / scale) * scale;

  // Cell-traversal distances: `tx` and `ty` are the distances to the next
  // vertical and horizontal cell boundary respectively.
  float ty =
      stepy == 0 ? INFINITY : stepy * (params.tile_y_min / delta + i + (stepy > 0) * scale) * dty;
  float tx =
      stepx == 0 ? INFINITY : stepx * (params.tile_x_min / delta + j + (stepx > 0) * scale) * dtx;
  int previous_axis = -1;
  const uint vertex_count = params.num_cell + 1U;

  // Step the ray across the mipmap cell-by-cell until we go off the edge or find an internal
  // collision
  while (i >= 0 && j >= 0 && i < n && j < n) {
    const float t_exit = min(tx, ty);

    // Find the minimum height that the ray has within the cell we've just crossed
    float z = ray_origin.z + t_exit * dz;
    if (dz > 0.0F) {
      // Since an upward ray will be higher at the exit boundary, rewind to find the minimum height
      // as the ray crosses the cell.
      z = previous_axis == -1
              ? ray_origin.z + t_start * dz
              : ray_origin.z + (t_exit - scale * (previous_axis == 0 ? dty : dtx)) * dz;
    }

    bool refined = false;
    const uint level_side = mipmap_level_side(params.num_cell, level);
    const uint cell_index =
        offset + (uint(i) >> (level - 1U)) * level_side + (uint(j) >> (level - 1U));
    if (z <= mipmap[cell_index]) {
      if (level == 1) {
        // Collision check. Restrict the bilinear root search to this cell's
        // actual DDA interval, including its near boundary.
        float t_entry = t_start;
        if (stepx != 0) {
          t_entry = max(t_entry, tx - dtx);
        }
        if (stepy != 0) {
          t_entry = max(t_entry, ty - dty);
        }
        const Collision collision = bilinear_collision(
            vertices,
            vertex_count,
            params.tile_x_min + float(j) * delta,
            params.tile_y_min + float(i) * delta,
            delta,
            uint(i),
            uint(j),
            ray_origin,
            ray_direction,
            min(t_entry, t_exit),
            t_exit
        );
        if (collision.hit) {
          distances[output_index] = collision.distance;
          elevations[output_index] = ray_origin.z + collision.distance * dz;
          return;
        }
      } else {
        // There might be a real collision inside this coarse cell. Descend to
        // the child containing the ray at the coarse cell's true near edge.
        float cell_entry = t_start;
        if (stepx != 0) {
          cell_entry = max(cell_entry, tx - scale * dtx);
        }
        if (stepy != 0) {
          cell_entry = max(cell_entry, ty - scale * dty);
        }

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
        i = clamp(int(floor((cell_y - params.tile_y_min) / delta)), 0, n - 1);
        j = clamp(int(floor((cell_x - params.tile_x_min) / delta)), 0, n - 1);

        // Fine indices remain in level-1 coordinates. Returning to a child
        // level requires their lower-left child-cell alignment before the
        // DDA timers are rebuilt; retaining an arbitrary fine index would
        // make a scale-sized timer skip an internal child boundary.
        t_start = cell_entry;
        level -= 1U;
        scale /= 2U;
        i = (i / int(scale)) * int(scale);
        j = (j / int(scale)) * int(scale);
        // The preceding level occupies `child_side` squared entries directly
        // before this one, so move backward without rebuilding the offset.
        const uint child_side = mipmap_level_side(params.num_cell, level);
        offset -= child_side * child_side;
        ty = stepy == 0 ? INFINITY
                        : stepy * (params.tile_y_min / delta + i + (stepy > 0) * scale) * dty;
        tx = stepx == 0 ? INFINITY
                        : stepx * (params.tile_x_min / delta + j + (stepx > 0) * scale) * dtx;
        // An entry through only one edge of a coarse cell may leave the
        // other axis part-way across the selected child. Advance its timer
        // rather than allowing a stale aligned boundary to move backward.
        if (stepy != 0) {
          ty = next_boundary_after(ty, t_start, scale * dty);
        }
        if (stepx != 0) {
          tx = next_boundary_after(tx, t_start, scale * dtx);
        }
        // The child begins a fresh DDA segment, so an upward ray's near-edge
        // test must use `t_start` rather than a preceding X or Y step.
        previous_axis = -1;
        refined = true;
        continue;
      }
    }

    // go up to a coarser level whenever possible
    if (!refined && level < params.num_levels) {
      if (ty < tx) {
        if (at_level_boundary(i, stepy, level)) {
          // Crossing a Y boundary joins two vertically adjacent blocks. The
          // X timer must therefore be adjusted from the X cell's sibling.
          tx += offset_jump(j, stepx, level, scale, dtx);
          // The current level immediately precedes the coarser level in the
          // flat buffer, so advance by its square element count.
          offset += level_side * level_side;
          level += 1;
          scale *= 2;
        }
      } else {
        if (at_level_boundary(j, stepx, level)) {
          // Crossing an X boundary joins two horizontally adjacent blocks.
          // Adjust the Y timer from the Y cell's sibling before coarsening.
          ty += offset_jump(i, stepy, level, scale, dty);
          // The level transition is identical regardless of the stepped axis.
          offset += level_side * level_side;
          level += 1;
          scale *= 2;
        }
      }
    }

    // Step forwards
    if (ty < tx) {
      previous_axis = 0;
      ty += scale * dty;
      i += scale * stepy;
    } else {
      previous_axis = 1;
      tx += scale * dtx;
      j += scale * stepx;
    }
  }
}
