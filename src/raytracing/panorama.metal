#include <metal_stdlib>

// Metal Shading Language provides GPU-specific types and functions in this
// namespace, including `uint`, `device`, and the `kernel` entry-point keyword.
using namespace metal;

/// Scalar-only terrain-tracing ABI mirrored by raytrace_gpu.h.
///
/// Projected coordinates have already been rebased around the observer.
struct RaytraceParameters {
  float cell_size;
  float observer_elevation;
  float curvature_coefficient;
  float global_maximum_elevation;
  uint num_levels;
  uint num_azimuth;
  uint num_polar;
  float max_distance;
};

/// One observer-relative origin for a resident atlas slot, mirrored
/// by `ResidentTile` in resident_tile_cache.h. All other parameters are dispatch-wide.
struct ResidentTile {
  float tile_x_min;
  float tile_y_min;
  long row;
  long column;
};

/// One open-addressed lookup entry mapping a global tile key to an atlas slot.
/// This must remain identical to `ResidentTileHashEntry` in resident_tile_cache.h.
struct ResidentTileHashEntry {
  long row;
  long column;
  uint slot;
  uint occupied;
};

/// One unresolved azimuth-column segment in the GPU-owned work frontier.
/// This must remain identical to `TileWorkItem` in raytrace_gpu.h.
struct TileWorkItem {
  uint slot;
  uint azimuth;
  uint first_polar;
  uint start_level;
  float entry_distance;
};

/// One continuation whose successor terrain tile is not resident yet.
///
/// The CPU resolves its tile key and retries it after the tile loader assigns
/// that terrain to an atlas slot. This mirrors `DeferredTileWork` on the host.
struct DeferredTileWork {
  uint azimuth;
  uint first_polar;
  float entry_distance;
};

/// Fixed record offsets for a uint16 terrain atlas, mirrored by the host's
/// `QuantizedTerrainLayout`.
struct QuantizedTerrainLayout {
  uint record_stride;
  uint vertex_offset;
  uint elevation_base_offset;
};

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
    uint2 output_index [[thread_position_in_grid]]
) {
  if (output_index.x >= vertex_count || output_index.y >= tile_count) {
    return;
  }

  device const uchar *record = source_records + output_index.y * source_record_stride;
  device const int *base = reinterpret_cast<device const int *>(record + elevation_base_offset);
  device const ushort *vertices = reinterpret_cast<device const ushort *>(record + vertex_offset);
  const uint slot = destination_slots[output_index.y];
  destination[slot * vertex_count + output_index.x] =
      (float(*base) + float(vertices[output_index.x])) * 0.1F;
}

/// Build one square maximum-mipmap level for a batch of atlas slots.
///
/// For level 1, `source` addresses the vertex grid, `source_step` is one, and
/// every output is the maximum of four adjacent vertices. Later levels bind
/// the preceding mip level, set `source_step` to two, and reduce each disjoint
/// 2 by 2 child block. The Z grid coordinate selects an entry in `slots`, so
/// all newly loaded tiles share one dispatch per level. Separate dispatches
/// provide the global barrier required before a newly written level becomes
/// the next dispatch's source.
template <typename Sample>
inline void build_maximum_mipmap_level_impl(
    device const Sample *source,
    device Sample *destination,
    uint source_side,
    uint source_step,
    device const uint *slots,
    uint source_tile_stride,
    uint destination_tile_stride,
    uint tile_count,
    uint3 output_index
) {
  const uint output_side = source_step == 1U ? source_side - 1U : source_side / 2U;
  if (output_index.x >= output_side || output_index.y >= output_side ||
      output_index.z >= tile_count) {
    return;
  }

  const uint slot = slots[output_index.z];
  source += slot * source_tile_stride;
  destination += slot * destination_tile_stride;
  const uint source_x = output_index.x * source_step;
  const uint source_y = output_index.y * source_step;
  const uint lower_left = source_y * source_side + source_x;
  destination[output_index.y * output_side + output_index.x] =
      max(max(source[lower_left], source[lower_left + 1U]),
          max(source[lower_left + source_side], source[lower_left + source_side + 1U]));
}

kernel void build_maximum_mipmap_level(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &source_side [[buffer(2)]],
    constant uint &source_step [[buffer(3)]],
    device const uint *slots [[buffer(4)]],
    constant uint &source_tile_stride [[buffer(5)]],
    constant uint &destination_tile_stride [[buffer(6)]],
    constant uint &tile_count [[buffer(7)]],
    uint3 output_index [[thread_position_in_grid]]
) {
  build_maximum_mipmap_level_impl(
      source,
      destination,
      source_side,
      source_step,
      slots,
      source_tile_stride,
      destination_tile_stride,
      tile_count,
      output_index
  );
}

kernel void build_quantized_maximum_mipmap_level(
    device const ushort *source [[buffer(0)]],
    device ushort *destination [[buffer(1)]],
    constant uint &source_side [[buffer(2)]],
    constant uint &source_step [[buffer(3)]],
    device const uint *slots [[buffer(4)]],
    constant uint &source_tile_stride [[buffer(5)]],
    constant uint &destination_tile_stride [[buffer(6)]],
    constant uint &tile_count [[buffer(7)]],
    uint3 output_index [[thread_position_in_grid]]
) {
  build_maximum_mipmap_level_impl(
      source,
      destination,
      source_side,
      source_step,
      slots,
      source_tile_stride,
      destination_tile_stride,
      tile_count,
      output_index
  );
}

/// Mix one unsigned 64-bit value for the resident tile lookup table.
inline ulong mix_tile_hash(ulong value) {
  value ^= value >> 30UL;
  value *= 0xbf58476d1ce4e5b9UL;
  value ^= value >> 27UL;
  value *= 0x94d049bb133111ebUL;
  value ^= value >> 31UL;
  return value;
}

/// Return the initial bucket for one global row/column tile key.
inline uint tile_key_hash(long row, long column, uint mask) {
  const ulong mixed_row = mix_tile_hash(ulong(row));
  const ulong mixed_column = mix_tile_hash(ulong(column));
  return uint(mix_tile_hash(mixed_row ^ (mixed_column + 0x9e3779b97f4a7c15UL))) & mask;
}

/// Return a resident atlas slot for a key, or `hash_capacity` when absent.
inline uint lookup_resident_tile(
    device const ResidentTileHashEntry *entries,
    uint hash_capacity,
    long row,
    long column
) {
  const uint mask = hash_capacity - 1U;
  uint index = tile_key_hash(row, column, mask);
  for (uint probe = 0U; probe < hash_capacity; probe++) {
    const ResidentTileHashEntry entry = entries[index];
    if (entry.occupied == 0U) {
      return hash_capacity;
    }
    if (entry.row == row && entry.column == column) {
      return entry.slot;
    }
    index = (index + 1U) & mask;
  }
  return hash_capacity;
}

/// Result of an exact bilinear terrain-patch intersection test.
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

/// Decode one stored terrain sample. Float32 specialization is an identity.
inline float sample_elevation(float sample, int) { return sample; }

inline float sample_elevation(ushort sample, int base_decimeters) {
  return (float(base_decimeters) + float(sample)) * 0.1F;
}

/// Return ray elevation relative to the curved terrain datum at horizontal distance t.
inline float curved_ray_elevation(float origin, float slope, float curvature, float t) {
  return fma(curvature, t * t, fma(slope, t, origin));
}

/// Return the minimum curved ray elevation over one closed distance interval.
inline float minimum_curved_ray_elevation(
    float origin,
    float slope,
    float curvature,
    float t_entry,
    float t_exit
) {
  const float minimum_distance =
      curvature > 0.0F ? clamp(-slope / (2.0F * curvature), t_entry, t_exit)
                       : (slope > 0.0F ? t_entry : t_exit);
  return curved_ray_elevation(origin, slope, curvature, minimum_distance);
}

/// Return whether a rising ray is now permanently above all catalogued terrain.
inline bool above_global_terrain(
    float origin,
    float slope,
    float curvature,
    float distance,
    float global_maximum
) {
  constexpr float kElevationCullingMargin = 1.0F;
  return isfinite(global_maximum) && slope + 2.0F * curvature * distance >= 0.0F &&
         curved_ray_elevation(origin, slope, curvature, distance) >
             global_maximum + kElevationCullingMargin;
}

/// Solve the exact intersection with one bilinear level-0 terrain patch. The
/// local parameter begins at t_entry to avoid cancellation on distant cells.
template <typename Sample>
inline Collision bilinear_collision(
    device const Sample *vertices,
    uint vertex_count,
    int base_decimeters,
    float cell_x,
    float cell_y,
    float delta,
    uint i,
    uint j,
    float3 ray_origin,
    float3 ray_direction,
    float curvature,
    float t_entry,
    float t_exit
) {
  constexpr float kPolynomialEpsilon = 1e-12F;
  const uint lower_left = i * vertex_count + j;
  const float z00 = sample_elevation(vertices[lower_left], base_decimeters);
  const float z01 = sample_elevation(vertices[lower_left + 1U], base_decimeters);
  const float z10 = sample_elevation(vertices[lower_left + vertex_count], base_decimeters);
  const float z11 =
      sample_elevation(vertices[lower_left + vertex_count + 1U], base_decimeters);
  const float interval_length = t_exit - t_entry;

  const float sx0 = (ray_origin.x + t_entry * ray_direction.x - cell_x) / delta;
  const float sy0 = (ray_origin.y + t_entry * ray_direction.y - cell_y) / delta;
  const float sx1 = ray_direction.x / delta;
  const float sy1 = ray_direction.y / delta;
  const float dzdx = z01 - z00;
  const float dzdy = z10 - z00;
  const float twist = z11 - z10 - z01 + z00;

  const float a = curvature - twist * sx1 * sy1;
  const float b = ray_direction.z + 2.0F * curvature * t_entry - dzdx * sx1 - dzdy * sy1 -
                  twist * (sx0 * sy1 + sx1 * sy0);
  const float c =
      curved_ray_elevation(ray_origin.z, ray_direction.z, curvature, t_entry) - z00 -
      dzdx * sx0 - dzdy * sy0 - twist * sx0 * sy0;

  const float cell_speed = max(max(fabs(sx1), fabs(sy1)), 1e-12F);
  const float direction_error = FLT_EPSILON * max(fabs(t_entry), fabs(t_exit)) * cell_speed;
  const float coordinate_tolerance = max(5e-5F, max(128.0F * direction_error, 0.05F / delta));
  const float t_tolerance = coordinate_tolerance / cell_speed;

  // A near-corner DDA transition can leave the root in an adjacent cell. If
  // the ray is below all four vertices by this interval's end, recover the
  // intersection on the near boundary as in the Python reference.
  const auto conservative_boundary_hit = [&]() -> Collision {
    if (curved_ray_elevation(ray_origin.z, ray_direction.z, curvature, t_exit) <=
        min(min(z00, z01), min(z10, z11))) {
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

  // Form the roots without subtracting nearly equal values. Curvature makes
  // even a flat terrain patch quadratic, and the naïve formula loses useful
  // distance precision for steep rays whose near root is much smaller than
  // their far root.
  const float root = sqrt(discriminant);
  const float root_product = -0.5F * (b + copysign(root, b));
  float local_t0 = 0.0F;
  float local_t1 = 0.0F;
  if (root_product == 0.0F) {
    local_t0 = -b / (2.0F * a);
    local_t1 = local_t0;
  } else {
    local_t0 = root_product / a;
    local_t1 = c / root_product;
  }
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

/// Return the level-1 cell count implied by a complete power-of-two mipmap.
///
/// Level 1 is the N×N field and every later level halves that side, so the
/// final 1×1 level makes `num_levels` equal to log2(N) + 1.
inline uint mipmap_finest_side(uint num_levels) { return 1U << (num_levels - 1U); }

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

/// Return the first tile boundary strictly after a segment's exact entry.
inline float tile_exit_distance(
    float tile_x_min,
    float tile_y_min,
    float cell_size,
    uint cell_count,
    float2 direction,
    float entry_distance
) {
  float tile_exit = INFINITY;
  const float x_max = tile_x_min + float(cell_count) * cell_size;
  const float y_max = tile_y_min + float(cell_count) * cell_size;
  if (direction.x > 0.0F) {
    const float candidate = x_max / direction.x;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  } else if (direction.x < 0.0F) {
    const float candidate = tile_x_min / direction.x;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  }
  if (direction.y > 0.0F) {
    const float candidate = y_max / direction.y;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  } else if (direction.y < 0.0F) {
    const float candidate = tile_y_min / direction.y;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  }
  return tile_exit;
}

/// Shared traversal specialized at compile time for Float32 or uint16 terrain.
template <typename Sample>
inline void trace_tile_frontier_impl(
    device const Sample *mipmap,
    device const Sample *vertices,
    int base_decimeters,
    device const float2 *azimuth_directions,
    device const float *polar_slopes,
    TileWorkItem input,
    device const ResidentTile *tiles,
    RaytraceParameters params,
    device float *distances,
    device float *elevations,
    device atomic_uint *first_unresolved,
    uint work_index,
    uint polar_index
) {
  // Neighboring lanes differ in polar index and share the DDA path represented
  // by one azimuth-column work item.
  const ResidentTile resident_tile = tiles[input.slot];
  const float tile_x_min = resident_tile.tile_x_min;
  const float tile_y_min = resident_tile.tile_y_min;
  if (polar_index >= params.num_polar || input.azimuth >= params.num_azimuth) {
    return;
  }
  const uint azimuth_index = input.azimuth;
  const uint output_index = polar_index * params.num_azimuth + azimuth_index;

  // Rays in a given column that have already intersected (and are therefore below the first index
  // that needs tracing because of the 2.5D nature of the heightfield).
  if (polar_index < input.first_polar) {
    return;
  }

  // All resident slots share dimensions.
  const uint num_cell = mipmap_finest_side(params.num_levels);

  // Get ray parameters. Horizontal directions use the compass convention:
  // x is eastward, y is northward, and `dz` is the vertical slope.
  const float2 direction = azimuth_directions[azimuth_index];
  const int stepx = int(direction.x > 0.0F) - int(direction.x < 0.0F);
  const int stepy = int(direction.y > 0.0F) - int(direction.y < 0.0F);
  const float delta = params.cell_size;
  const float dtx = stepx == 0 ? INFINITY : fabs(delta / direction.x);
  const float dty = stepy == 0 ? INFINITY : fabs(delta / direction.y);
  const float dz = polar_slopes[polar_index];
  const float3 ray_origin = {0.0F, 0.0F, params.observer_elevation};
  const float3 ray_direction = {direction.x, direction.y, dz};
  const int n = int(num_cell);

  // The observer tile begins at level 1. An incoming tile begins at its
  // maximum level so the maximum pyramid can skip empty terrain immediately.
  // TODO: check whether it would be better to start at the coarsest level no matter what since we
  // are no longer using the CPU-based continuation trick
  uint level = input.start_level;
  uint scale = 1 << (level - 1);
  // The preceding level areas are N², N²/4, ..., so their geometric-series
  // sum is 4 * (N² - side²) / 3. Use 64-bit intermediates: each level's
  // offset remains a uint under the host's atlas-size limits, but N² should
  // not overflow while the formula is being evaluated.
  const ulong full_area = ulong(num_cell) * ulong(num_cell);
  const ulong current_side = ulong(mipmap_level_side(num_cell, level));
  const ulong current_area = current_side * current_side;
  uint offset = uint((4UL * (full_area - current_area)) / 3UL);

  // The exact near boundary of the active DDA segment. It remains separate
  // from points nudged only to assign deterministic cell ownership.
  float t_start = input.entry_distance;

  // The local observer is exactly at (0, 0), which may lie on one or both
  // shared cell boundaries. Nudge only the coordinate used for ownership so a
  // south/west ray starts in its forward cell; all DDA distances still use the
  // exact observer position and therefore retain t = 0 as their geometry.
  const float boundary_nudge = 1e-3F * delta;
  const float x_entry = t_start * direction.x;
  const float y_entry = t_start * direction.y;
  const float x_classify = stepx == 0 ? x_entry : x_entry + copysign(boundary_nudge, direction.x);
  const float y_classify = stepy == 0 ? y_entry : y_entry + copysign(boundary_nudge, direction.y);
  int i = clamp(int(floor((y_classify - tile_y_min) / delta)), 0, n - 1);
  int j = clamp(int(floor((x_classify - tile_x_min) / delta)), 0, n - 1);

  // align to the correct level boundary
  i = (i / scale) * scale;
  j = (j / scale) * scale;

  // Cell-traversal distances: `tx` and `ty` are the distances to the next
  // vertical and horizontal cell boundary respectively.
  float ty = stepy == 0 ? INFINITY : stepy * (tile_y_min / delta + i + (stepy > 0) * scale) * dty;
  float tx = stepx == 0 ? INFINITY : stepx * (tile_x_min / delta + j + (stepx > 0) * scale) * dtx;
  // A coarse incoming segment can begin inside the other axis's aligned
  // block. Reposition both so neither moves behind the true hand-off.
  if (stepy != 0) {
    ty = next_boundary_after(ty, t_start, scale * dty);
  }
  if (stepx != 0) {
    tx = next_boundary_after(tx, t_start, scale * dtx);
  }

  const uint vertex_count = num_cell + 1U;
  const float tile_exit =
      tile_exit_distance(tile_x_min, tile_y_min, params.cell_size, num_cell, direction, t_start);
  const float segment_limit = min(tile_exit, params.max_distance);

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
            ray_origin.z,
            dz,
            params.curvature_coefficient,
            interval_start,
            params.global_maximum_elevation
        )) {
      return;
    }
    const float z = minimum_curved_ray_elevation(
        ray_origin.z,
        dz,
        params.curvature_coefficient,
        interval_start,
        interval_end
    );

    // find the index of the relevant cell (correct level and location) inside the flattened mipmap
    const uint level_side = mipmap_level_side(num_cell, level);
    const uint cell_index =
        offset + (uint(i) >> (level - 1U)) * level_side + (uint(j) >> (level - 1U));

    // Collision check
    if (z <= sample_elevation(mipmap[cell_index], base_decimeters)) {
      if (level == 1) {
        // Finest level collision check. Restrict the bilinear root search to this cell's
        // actual DDA interval, including its near boundary.
        const Collision collision = bilinear_collision(
            vertices,
            vertex_count,
            base_decimeters,
            tile_x_min + float(j) * delta,
            tile_y_min + float(i) * delta,
            delta,
            uint(i),
            uint(j),
            ray_origin,
            ray_direction,
            params.curvature_coefficient,
            interval_start,
            interval_end
        );

        // Exit only after the exact patch test confirms a hit.
        if (collision.hit) {
          distances[output_index] = collision.distance;
          elevations[output_index] = curved_ray_elevation(
              ray_origin.z,
              dz,
              params.curvature_coefficient,
              collision.distance
          );
          return;
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
        i = clamp(int(floor((cell_y - tile_y_min) / delta)), 0, n - 1);
        j = clamp(int(floor((cell_x - tile_x_min) / delta)), 0, n - 1);

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
        const uint child_side = mipmap_level_side(num_cell, level);
        offset -= child_side * child_side;
        ty = stepy == 0 ? INFINITY : stepy * (tile_y_min / delta + i + (stepy > 0) * scale) * dty;
        tx = stepx == 0 ? INFINITY : stepx * (tile_x_min / delta + j + (stepx > 0) * scale) * dtx;

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
        atomic_fetch_min_explicit(&first_unresolved[work_index], polar_index, memory_order_relaxed);
      }
      return;
    }

    // Go up to a coarser level whenever possible
    if (level < params.num_levels) {
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
      ty += scale * dty;
      i += scale * stepy;
    } else {
      tx += scale * dtx;
      j += scale * stepx;
    }
  }
  // The DDA normally returns through the segment-limit check above. Retain a
  // defensive continuation for an unexpected fine-cell exit discrepancy.
  if (tile_exit <= params.max_distance) {
    atomic_fetch_min_explicit(&first_unresolved[work_index], polar_index, memory_order_relaxed);
  }
}

/// Trace one independent polar ray through Float32 resident terrain.
kernel void trace_tile_frontier(
    device const float *mipmap_atlas [[buffer(0)]],
    device const float *vertex_atlas [[buffer(1)]],
    device const float2 *azimuth_directions [[buffer(2)]],
    device const float *polar_slopes [[buffer(3)]],
    device const TileWorkItem *work_items [[buffer(4)]],
    device const ResidentTile *tiles [[buffer(5)]],
    constant RaytraceParameters &shared_parameters [[buffer(6)]],
    constant uint &mipmap_value_count [[buffer(7)]],
    device float *distances [[buffer(8)]],
    device float *elevations [[buffer(9)]],
    device atomic_uint *first_unresolved [[buffer(10)]],
    uint2 ray_index [[thread_position_in_grid]]
) {
  const TileWorkItem input = work_items[ray_index.y];
  const uint num_cell = mipmap_finest_side(shared_parameters.num_levels);
  const uint vertex_value_count = (num_cell + 1U) * (num_cell + 1U);
  trace_tile_frontier_impl(
      mipmap_atlas + input.slot * mipmap_value_count,
      vertex_atlas + input.slot * vertex_value_count,
      0,
      azimuth_directions,
      polar_slopes,
      input,
      tiles,
      shared_parameters,
      distances,
      elevations,
      first_unresolved,
      ray_index.y,
      ray_index.x
  );
}

/// Trace one independent polar ray while decoding uint16 elevations only when
/// the traversal reads a mipmap maximum or exact bilinear-patch vertex.
kernel void trace_tile_frontier_quantized(
    device const ushort *mipmap_atlas [[buffer(0)]],
    device const uchar *vertex_records [[buffer(1)]],
    device const float2 *azimuth_directions [[buffer(2)]],
    device const float *polar_slopes [[buffer(3)]],
    device const TileWorkItem *work_items [[buffer(4)]],
    device const ResidentTile *tiles [[buffer(5)]],
    constant RaytraceParameters &shared_parameters [[buffer(6)]],
    constant uint &mipmap_value_count [[buffer(7)]],
    device float *distances [[buffer(8)]],
    device float *elevations [[buffer(9)]],
    device atomic_uint *first_unresolved [[buffer(10)]],
    constant QuantizedTerrainLayout &layout [[buffer(11)]],
    uint2 ray_index [[thread_position_in_grid]]
) {
  const TileWorkItem input = work_items[ray_index.y];
  device const uchar *record = vertex_records + input.slot * layout.record_stride;
  device const int *base =
      reinterpret_cast<device const int *>(record + layout.elevation_base_offset);
  device const ushort *vertices =
      reinterpret_cast<device const ushort *>(record + layout.vertex_offset);
  trace_tile_frontier_impl(
      mipmap_atlas + input.slot * mipmap_value_count,
      vertices,
      *base,
      azimuth_directions,
      polar_slopes,
      input,
      tiles,
      shared_parameters,
      distances,
      elevations,
      first_unresolved,
      ray_index.y,
      ray_index.x
  );
}

/// Turn each column's atomic unresolved-polar result into its successor work
/// item. A hash table maps the outgoing global tile key to a resident slot;
/// a nonresident successor is returned to the host without changing its exact
/// hand-off distance.
kernel void emit_tile_frontier(
    device const TileWorkItem *active_items [[buffer(0)]],
    device const ResidentTile *tiles [[buffer(1)]],
    device const float2 *azimuth_directions [[buffer(2)]],
    device const atomic_uint *first_unresolved [[buffer(3)]],
    constant RaytraceParameters &shared_parameters [[buffer(4)]],
    device const ResidentTileHashEntry *resident_hash [[buffer(5)]],
    constant uint &hash_capacity [[buffer(6)]],
    constant uint &next_capacity [[buffer(7)]],
    device TileWorkItem *next_items [[buffer(8)]],
    device atomic_uint *next_count [[buffer(9)]],
    device DeferredTileWork *deferred_items [[buffer(10)]],
    device atomic_uint *deferred_count [[buffer(11)]],
    uint work_index [[thread_position_in_grid]]
) {
  const TileWorkItem active = active_items[work_index];
  const ResidentTile current_tile = tiles[active.slot];
  const RaytraceParameters params = shared_parameters;
  const float tile_x_min = current_tile.tile_x_min;
  const float tile_y_min = current_tile.tile_y_min;
  const uint first_polar =
      atomic_load_explicit(&first_unresolved[work_index], memory_order_relaxed);
  if (first_polar >= params.num_polar) {
    return;
  }

  const float2 direction = azimuth_directions[active.azimuth];
  const float entry_distance = tile_exit_distance(
      tile_x_min,
      tile_y_min,
      params.cell_size,
      mipmap_finest_side(params.num_levels),
      direction,
      active.entry_distance
  );
  if (entry_distance >= params.max_distance) {
    return;
  }

  // The outward nudge selects the neighbour at an edge/corner, while the
  // original exit distance remains the geometric start of its segment.
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

  // The nudge makes these comparisons deterministic at shared edges and
  // corners. A row increases southward while a column increases eastward.
  const float tile_width = float(mipmap_finest_side(params.num_levels)) * params.cell_size;
  const float tile_x_max = tile_x_min + tile_width;
  const float tile_y_max = tile_y_min + tile_width;
  const long row_offset = y < tile_y_min ? 1L : y >= tile_y_max ? -1L : 0L;
  const long column_offset = x < tile_x_min ? -1L : x >= tile_x_max ? 1L : 0L;
  const long successor_row = current_tile.row + row_offset;
  const long successor_column = current_tile.column + column_offset;
  const uint successor_slot =
      lookup_resident_tile(resident_hash, hash_capacity, successor_row, successor_column);
  // The source directory may contain this successor even though a background
  // worker has not loaded it into the atlas yet. Preserve its exact hand-off
  // for host-side retry rather than incorrectly treating it as open sky.
  if (successor_slot == hash_capacity) {
    const uint deferred_index = atomic_fetch_add_explicit(deferred_count, 1U, memory_order_relaxed);
    if (deferred_index < next_capacity) {
      deferred_items[deferred_index] = {active.azimuth, first_polar, entry_distance};
    }
    return;
  }

  const uint output_index = atomic_fetch_add_explicit(next_count, 1U, memory_order_relaxed);
  if (output_index >= next_capacity) {
    // The host allocates one entry per azimuth because a column has only one
    // unresolved suffix. Dropping an unexpected excess is safer than
    // corrupting the adjacent buffer; the host detects the counter overflow.
    return;
  }
  next_items[output_index] = {
      successor_slot,
      active.azimuth,
      first_polar,
      params.num_levels,
      entry_distance,
  };
}
