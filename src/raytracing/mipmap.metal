#include <metal_stdlib>

// Metal Shading Language provides GPU-specific types and functions in this
// namespace, including `uint`, `device`, and the `kernel` entry-point keyword.
using namespace metal;

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
