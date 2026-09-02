#include <metal_stdlib>

// Metal Shading Language provides GPU-specific types and functions in this
// namespace, including `uint`, `device`, and the `kernel` entry-point keyword.
using namespace metal;

/// Pipeline specializations selected once per render. False options let Metal
/// remove the associated collision arithmetic and output writes entirely.
constant bool compute_surface_gradients [[function_constant(0)]];
constant bool store_collision_elevations [[function_constant(1)]];

/// Scalar-only terrain-tracing ABI mirrored by raytrace_gpu.h.
///
/// Projected coordinates have already been rebased around the observer.
struct RaytraceParameters {
  /// LOD-1 vertex spacing in projected metres.
  float cell_size;
  /// Camera elevation in the terrain vertical datum.
  float observer_elevation;
  /// Effective-Earth lift per squared horizontal metre.
  float curvature_coefficient;
  /// Catalogue-wide conservative upper bound, or infinity when unavailable.
  float global_maximum_elevation;
  /// Number of levels in the reference tile's maximum hierarchy.
  uint num_levels;
  /// Number of output rays and collision records.
  uint ray_count;
  /// Maximum permitted horizontal traversal distance.
  float max_distance;
};

/// One normalized horizontal direction and vertical slope per output pixel.
/// This must remain identical to `RayDirection` in ray_projection.h.
struct RayDirection {
  /// Unit horizontal component toward projected east.
  float x;
  /// Unit horizontal component toward projected north.
  float y;
  /// Reciprocal east component used by DDA boundary calculations.
  float inverse_x;
  /// Reciprocal north component used by DDA boundary calculations.
  float inverse_y;
  /// Vertical change per metre of horizontal travel.
  float slope;
};

/// One observer-relative origin for a resident atlas slot, mirrored
/// by `ResidentTile` in tile_manager_gpu.h. All other parameters are dispatch-wide.
struct ResidentTile {
  /// Observer-relative easting of the south-west tile corner.
  float tile_x_min;
  /// Observer-relative northing of the south-west tile corner.
  float tile_y_min;
  /// Conservative complete-tile elevation bound.
  float maximum_elevation;
  /// One-based surface LOD stored at the beginning of this atlas slot.
  uint lod;
  /// Global north-to-south grid row used for successor lookup.
  long row;
  /// Global west-to-east grid column used for successor lookup.
  long column;
};

/// One immutable catalogue key, manifest maximum, and host source index.
struct CatalogueTileHashEntry {
  /// Global north-to-south grid row.
  long row;
  /// Global west-to-east grid column.
  long column;
  /// Manifest upper bound used to skip this source without loading it.
  float maximum_elevation;
  /// Stable host catalogue index returned in deferred work.
  uint source_index;
};

/// One unresolved ray segment in the GPU-owned work frontier.
/// This must remain identical to `RayWorkItem` in raytrace_gpu.h.
struct RayWorkItem {
  /// Resident atlas slot traversed by this invocation.
  uint slot;
  /// Output ray whose distance and optional products are updated.
  uint ray_index;
  /// One-based maximum-mipmap level at which traversal begins.
  uint start_level;
  /// Exact horizontal distance to this tile segment's near boundary.
  float entry_distance;
};

/// One continuation whose successor terrain tile is not resident yet.
///
/// The GPU resolves its catalogue source; the CPU retries the ray after the
/// loader assigns that source to an atlas slot. This mirrors the host type.
struct DeferredRayWork {
  /// Primary or shadow output ray being continued.
  uint ray_index;
  /// Catalogue source required for its next non-skippable segment.
  uint source_index;
  /// Exact horizontal distance from this ray's origin to the source boundary.
  float entry_distance;
};

/// Per-shadow-ray geometry. Origins are observer-relative horizontally, but
/// carry their absolute projected elevation so curvature starts at the
/// camera collision rather than at the observer.
struct ShadowRay {
  /// Observer-relative X/Y, absolute elevation Z; W is unused padding.
  float4 origin;
};

/// Scalar shadow-tracing ABI mirrored by terrain_shadow_gpu.h.
struct ShadowTraceParameters {
  /// Terrain dimensions, curvature, bounds, and ray capacity.
  RaytraceParameters trace;
  /// Common direction from every collision toward the directional sun.
  RayDirection direction;
  /// Observer-relative western edge of catalogue column zero.
  float grid_x_min;
  /// Observer-relative northern edge of catalogue row zero.
  float grid_y_max;
  /// Common physical source-tile width.
  float tile_width;
  /// Power-of-two size of the open-addressed catalogue table.
  uint catalogue_hash_capacity;
};

/// Fixed record offsets for a uint16 terrain atlas, mirrored by the host's
/// `QuantizedTerrainLayout`.
struct QuantizedTerrainLayout {
  /// Byte distance between packed atlas records.
  uint record_stride;
  /// Byte offset from a record to its uint16 vertex array.
  uint vertex_offset;
  /// Byte offset from a record to its signed decimetre elevation base.
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

/// Build four adjacent level-1 cells and their level-2 parent from one 3×3
/// vertex patch.
///
/// Computing these outputs independently would issue sixteen source reads.
/// Grouping them in one thread requires only nine reads, keeps the values in
/// registers, and removes the separate level-2 reduction dispatch.
template <typename Sample>
inline void build_initial_maximum_mipmap_levels_impl(
    device const Sample *source,
    device Sample *destination,
    uint cell_count,
    device const uint *slots,
    uint source_tile_stride,
    uint destination_tile_stride,
    uint tile_count,
    uint3 output_index
) {
  if (output_index.z >= tile_count) {
    return;
  }

  const uint slot = slots[output_index.z];
  source += slot * source_tile_stride;
  destination += slot * destination_tile_stride;
  const uint source_side = cell_count + 1U;
  const uint source_x = 2U * output_index.x;
  const uint source_y = 2U * output_index.y;
  const uint row_0 = source_y * source_side + source_x;
  const uint row_1 = row_0 + source_side;
  const uint row_2 = row_1 + source_side;
  const Sample value_00 = source[row_0];
  const Sample value_01 = source[row_0 + 1U];
  const Sample value_02 = source[row_0 + 2U];
  const Sample value_10 = source[row_1];
  const Sample value_11 = source[row_1 + 1U];
  const Sample value_12 = source[row_1 + 2U];
  const Sample value_20 = source[row_2];
  const Sample value_21 = source[row_2 + 1U];
  const Sample value_22 = source[row_2 + 2U];

  const Sample maximum_00 = max(max(value_00, value_01), max(value_10, value_11));
  const Sample maximum_01 = max(max(value_01, value_02), max(value_11, value_12));
  const Sample maximum_10 = max(max(value_10, value_11), max(value_20, value_21));
  const Sample maximum_11 = max(max(value_11, value_12), max(value_21, value_22));

  const uint level_1_x = 2U * output_index.x;
  const uint level_1_y = 2U * output_index.y;
  const uint level_1_offset = level_1_y * cell_count + level_1_x;
  destination[level_1_offset] = maximum_00;
  destination[level_1_offset + 1U] = maximum_01;
  destination[level_1_offset + cell_count] = maximum_10;
  destination[level_1_offset + cell_count + 1U] = maximum_11;

  const uint level_2_side = cell_count / 2U;
  const uint level_2_offset = cell_count * cell_count;
  destination[level_2_offset + output_index.y * level_2_side + output_index.x] =
      max(max(maximum_00, maximum_01), max(maximum_10, maximum_11));
}

kernel void build_initial_maximum_mipmap_levels(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &cell_count [[buffer(2)]],
    device const uint *slots [[buffer(4)]],
    constant uint &source_tile_stride [[buffer(5)]],
    constant uint &destination_tile_stride [[buffer(6)]],
    constant uint &tile_count [[buffer(7)]],
    uint3 output_index [[thread_position_in_grid]]
) {
  build_initial_maximum_mipmap_levels_impl(
      source,
      destination,
      cell_count,
      slots,
      source_tile_stride,
      destination_tile_stride,
      tile_count,
      output_index
  );
}

kernel void build_quantized_initial_maximum_mipmap_levels(
    device const ushort *source [[buffer(0)]],
    device ushort *destination [[buffer(1)]],
    constant uint &cell_count [[buffer(2)]],
    device const uint *slots [[buffer(4)]],
    constant uint &source_tile_stride [[buffer(5)]],
    constant uint &destination_tile_stride [[buffer(6)]],
    constant uint &tile_count [[buffer(7)]],
    uint3 output_index [[thread_position_in_grid]]
) {
  build_initial_maximum_mipmap_levels_impl(
      source,
      destination,
      cell_count,
      slots,
      source_tile_stride,
      destination_tile_stride,
      tile_count,
      output_index
  );
}

/// Reduce one square maximum-mipmap level for a batch of atlas slots.
///
/// The Z grid coordinate selects an entry in `slots`, so all newly loaded
/// tiles share one dispatch per level. Ending each encoder supplies the global
/// barrier before a newly written level becomes the next dispatch's source.
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

/// Mix one unsigned 64-bit value for the catalogue tile lookup table.
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

/// Return catalogue metadata for a key, or null when terrain coverage ends.
inline device const CatalogueTileHashEntry *lookup_catalogue_tile(
    device const CatalogueTileHashEntry *entries,
    uint capacity,
    long row,
    long column
) {
  const uint mask = capacity - 1U;
  uint index = tile_key_hash(row, column, mask);
  for (uint probe = 0U; probe < capacity; probe++) {
    device const CatalogueTileHashEntry *entry = entries + index;
    if (entry->source_index == 0xffffffffU) {
      return nullptr;
    }
    if (entry->row == row && entry->column == column) {
      return entry;
    }
    index = (index + 1U) & mask;
  }
  return nullptr;
}

/// Result of an exact bilinear terrain-patch intersection test.
struct Collision {
  /// True only when an accepted root lies in this cell's DDA interval.
  bool hit;
  /// Horizontal distance from the ray origin to that root.
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
    float stationary_distance,
    float t_entry,
    float t_exit
) {
  const float minimum_distance = clamp(stationary_distance, t_entry, t_exit);
  return curved_ray_elevation(origin, slope, curvature, minimum_distance);
}

/// Return whether a rising ray is now permanently above all catalogued terrain.
inline bool above_global_terrain(
    float origin,
    float slope,
    float curvature,
    float stationary_distance,
    float distance,
    float global_maximum
) {
  constexpr float kElevationCullingMargin = 1.0F;
  return isfinite(global_maximum) && distance >= stationary_distance &&
         curved_ray_elevation(origin, slope, curvature, distance) >
             global_maximum + kElevationCullingMargin;
}

/// Recover a near-boundary hit when DDA rounding assigns the root to an
/// adjacent cell even though the ray ends below this patch's lowest vertex.
inline Collision conservative_boundary_collision(
    float observer_elevation,
    float slope,
    float curvature,
    float t_entry,
    float t_exit,
    float minimum_vertex
) {
  if (curved_ray_elevation(observer_elevation, slope, curvature, t_exit) <= minimum_vertex) {
    return {true, t_entry};
  }
  return {false, 0.0F};
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
    float inverse_delta,
    uint i,
    uint j,
    float observer_elevation,
    float2 direction,
    float slope,
    float curvature,
    float t_entry,
    float t_exit
) {
  constexpr float kPolynomialEpsilon = 1e-12F;
  const uint lower_left = i * vertex_count + j;
  const float interval_length = t_exit - t_entry;

  const float sx0 = (t_entry * direction.x - cell_x) * inverse_delta;
  const float sy0 = (t_entry * direction.y - cell_y) * inverse_delta;
  const float sx1 = direction.x * inverse_delta;
  const float sy1 = direction.y * inverse_delta;

  // Keep decoded samples and surface gradients out of the root solver's live
  // set. Only the polynomial and the conservative fallback bound are needed
  // after this scope.
  float a;
  float b;
  float c;
  float minimum_vertex;
  {
    const float z00 = sample_elevation(vertices[lower_left], base_decimeters);
    const float z01 = sample_elevation(vertices[lower_left + 1U], base_decimeters);
    const float z10 = sample_elevation(vertices[lower_left + vertex_count], base_decimeters);
    const float z11 = sample_elevation(vertices[lower_left + vertex_count + 1U], base_decimeters);
    const float dzdx = z01 - z00;
    const float dzdy = z10 - z00;
    const float twist = z11 - z10 - z01 + z00;

    a = curvature - twist * sx1 * sy1;
    b = slope + 2.0F * curvature * t_entry - dzdx * sx1 - dzdy * sy1 -
        twist * (sx0 * sy1 + sx1 * sy0);
    c = curved_ray_elevation(observer_elevation, slope, curvature, t_entry) - z00 - dzdx * sx0 -
        dzdy * sy0 - twist * sx0 * sy0;
    minimum_vertex = min(min(z00, z01), min(z10, z11));
  }

  const float cell_speed = max(max(fabs(sx1), fabs(sy1)), 1e-12F);
  const float direction_error = FLT_EPSILON * max(fabs(t_entry), fabs(t_exit)) * cell_speed;
  const float coordinate_tolerance =
      max(5e-5F, max(128.0F * direction_error, 0.05F * inverse_delta));
  const float t_tolerance = coordinate_tolerance / cell_speed;

  // A near-corner DDA transition can leave the root in an adjacent cell. If
  // the ray is below all four vertices by this interval's end, recover the
  // intersection on the near boundary as in the Python reference.
  float local_t = 0.0F;
  if (fabs(a) < kPolynomialEpsilon) {
    if (fabs(b) < kPolynomialEpsilon) {
      return conservative_boundary_collision(
          observer_elevation,
          slope,
          curvature,
          t_entry,
          t_exit,
          minimum_vertex
      );
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
    return conservative_boundary_collision(
        observer_elevation,
        slope,
        curvature,
        t_entry,
        t_exit,
        minimum_vertex
    );
  }

  float discriminant = b * b - 4.0F * a * c;
  if (discriminant < 0.0F) {
    if (discriminant > -kPolynomialEpsilon) {
      discriminant = 0.0F;
    } else {
      return conservative_boundary_collision(
          observer_elevation,
          slope,
          curvature,
          t_entry,
          t_exit,
          minimum_vertex
      );
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
  return conservative_boundary_collision(
      observer_elevation,
      slope,
      curvature,
      t_entry,
      t_exit,
      minimum_vertex
  );
}

/// Evaluate the analytical east/north derivatives of a bilinear terrain patch
/// at a confirmed collision. Reloading the four samples here keeps them and
/// the derivative temporaries out of the root solver's peak live register set.
/// The two float16 gradients are sufficient to reconstruct the upward normal
/// as normalize(float3(-dz/deast, -dz/dnorth, 1)).
template <typename Sample>
inline uint packed_surface_gradients(
    device const Sample *vertices,
    uint vertex_count,
    int base_decimeters,
    float cell_x,
    float cell_y,
    float inverse_delta,
    uint i,
    uint j,
    float2 direction,
    float distance
) {
  const uint lower_left = i * vertex_count + j;
  const float z00 = sample_elevation(vertices[lower_left], base_decimeters);
  const float z01 = sample_elevation(vertices[lower_left + 1U], base_decimeters);
  const float z10 = sample_elevation(vertices[lower_left + vertex_count], base_decimeters);
  const float z11 = sample_elevation(vertices[lower_left + vertex_count + 1U], base_decimeters);
  const float twist = z11 - z10 - z01 + z00;
  const float sx = clamp((distance * direction.x - cell_x) * inverse_delta, 0.0F, 1.0F);
  const float sy = clamp((distance * direction.y - cell_y) * inverse_delta, 0.0F, 1.0F);
  constexpr float kMaximumHalf = 65504.0F;
  const float east_gradient =
      clamp((z01 - z00 + twist * sy) * inverse_delta, -kMaximumHalf, kMaximumHalf);
  const float north_gradient =
      clamp((z10 - z00 + twist * sx) * inverse_delta, -kMaximumHalf, kMaximumHalf);
  return as_type<uint>(half2(east_gradient, north_gradient));
}

/// Return whether stepping from this index crosses a parent-block boundary.
/// Only such a crossing permits traversal to move to the next coarser level.
inline bool at_level_boundary(int index, int direction, uint scale) {
  const uint block_mask = 2U * scale - 1U;
  const uint boundary_remainder = direction > 0 ? block_mask : 0U;
  return direction != 0 && (uint(index) & block_mask) == boundary_remainder;
}

/// Return the opposite-axis DDA adjustment when moving to the next coarser level.
inline float offset_jump(int index, int direction, uint scale, float dt) {
  if (direction == 0) {
    return 0.0F;
  }
  const bool upper_sibling = (uint(index) & scale) != 0U;
  const bool crosses_sibling = upper_sibling != (direction > 0);
  return float(crosses_sibling) * float(scale) * dt;
}

/// Align a nonnegative level-1 cell index to its containing power-of-two block.
inline int align_to_level(int index, uint scale) { return int(uint(index) & ~(scale - 1U)); }

/// Return the level-1 cell count implied by a complete power-of-two mipmap.
///
/// Level 1 is the N×N field and every later level halves that side, so the
/// final 1×1 level makes `num_levels` equal to log2(N) + 1.
inline uint mipmap_finest_side(uint num_levels) { return 1U << (num_levels - 1U); }

/// Return the row-major side length of a one-indexed mipmap level.
inline uint mipmap_level_side(uint cell_count, uint level) { return cell_count >> (level - 1U); }

/// Return the number of values in a flattened complete maximum-mipmap.
inline uint mipmap_value_count(uint cell_count) {
  uint count = 0U;
  for (uint side = cell_count; side != 0U; side >>= 1U) {
    count += side * side;
  }
  return count;
}

/// Derive the geometry of a resident tile from the dispatch's LOD-1 reference
/// values. Every higher LOD doubles cell spacing but keeps the tile extent.
inline bool resident_lod_geometry(
    ResidentTile tile,
    RaytraceParameters params,
    thread uint &cell_count,
    thread uint &num_levels,
    thread float &cell_size,
    thread uint &lod_mipmap_value_count
) {
  if (tile.lod == 0U || tile.lod > params.num_levels) {
    return false;
  }
  const uint lod_shift = tile.lod - 1U;
  num_levels = params.num_levels - lod_shift;
  cell_count = mipmap_finest_side(num_levels);
  cell_size = params.cell_size * float(1U << lod_shift);
  lod_mipmap_value_count = mipmap_value_count(cell_count);
  return true;
}

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
    float4 direction,
    float entry_distance
) {
  float tile_exit = INFINITY;
  const float x_max = tile_x_min + float(cell_count) * cell_size;
  const float y_max = tile_y_min + float(cell_count) * cell_size;
  if (direction.x > 0.0F) {
    const float candidate = x_max * direction.z;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  } else if (direction.x < 0.0F) {
    const float candidate = tile_x_min * direction.z;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  }
  if (direction.y > 0.0F) {
    const float candidate = y_max * direction.w;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  } else if (direction.y < 0.0F) {
    const float candidate = tile_y_min * direction.w;
    if (candidate > entry_distance) {
      tile_exit = min(tile_exit, candidate);
    }
  }
  return tile_exit;
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

    // Collision check
    if (z <= sample_elevation(mipmap[cell_index], base_decimeters)) {
      if (level == 1) {
        // Finest level collision check. Restrict the bilinear root search to this cell's
        // actual DDA interval, including its near boundary.
        const Collision collision = bilinear_collision(
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

        // Exit only after the exact patch test confirms a hit.
        if (collision.hit) {
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
            surface_gradients[output_index] = packed_surface_gradients(
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
    if (level < params.num_levels) {
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
    uint ray_index [[thread_position_in_grid]]
) {
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

  const long column = long(floor((x - params.grid_x_min) / params.tile_width));
  const long row = long(floor((params.grid_y_max - y) / params.tile_width));
  device const CatalogueTileHashEntry *source =
      lookup_catalogue_tile(catalogue_hash, params.catalogue_hash_capacity, row, column);
  if (source == nullptr) {
    return;
  }
  const uint output = atomic_fetch_add_explicit(deferred_count, 1U, memory_order_relaxed);
  if (output < params.trace.ray_count) {
    deferred_items[output] = {ray_index, source->source_index, 0.0F};
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
    if (source == nullptr) {
      break;
    }
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
    if (source == nullptr) {
      break;
    }
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
