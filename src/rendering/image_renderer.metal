#include <metal_stdlib>

using namespace metal;

/// Atomic ordered-float endpoints produced by the diagnostic range reduction.
struct FiniteRange {
  atomic_uint minimum;
  atomic_uint maximum;
};

/// Five-stop approximations of the CLI's built-in perceptual colourmaps.
constant float3 preset_colourmaps[6][5] = {
    {
        float3(68.0F, 1.0F, 84.0F),
        float3(59.0F, 82.0F, 139.0F),
        float3(33.0F, 145.0F, 140.0F),
        float3(94.0F, 201.0F, 98.0F),
        float3(253.0F, 231.0F, 37.0F),
    },
    {
        float3(13.0F, 8.0F, 135.0F),
        float3(126.0F, 3.0F, 168.0F),
        float3(204.0F, 71.0F, 120.0F),
        float3(248.0F, 149.0F, 64.0F),
        float3(240.0F, 249.0F, 33.0F),
    },
    {
        float3(0.0F, 0.0F, 4.0F),
        float3(87.0F, 15.0F, 109.0F),
        float3(187.0F, 55.0F, 84.0F),
        float3(249.0F, 142.0F, 9.0F),
        float3(252.0F, 255.0F, 164.0F),
    },
    {
        float3(0.0F, 0.0F, 4.0F),
        float3(81.0F, 18.0F, 124.0F),
        float3(183.0F, 55.0F, 121.0F),
        float3(252.0F, 137.0F, 97.0F),
        float3(252.0F, 253.0F, 191.0F),
    },
    {
        float3(0.0F, 32.0F, 77.0F),
        float3(65.0F, 77.0F, 108.0F),
        float3(124.0F, 124.0F, 120.0F),
        float3(190.0F, 175.0F, 110.0F),
        float3(255.0F, 233.0F, 69.0F),
    },
    {
        float3(48.0F, 18.0F, 59.0F),
        float3(40.0F, 188.0F, 235.0F),
        float3(164.0F, 252.0F, 60.0F),
        float3(251.0F, 126.0F, 33.0F),
        float3(122.0F, 4.0F, 3.0F),
    },
};

/// Map float32 bits to unsigned integers whose ordering matches the floats.
inline uint ordered_float(float value) {
  const uint bits = as_type<uint>(value);
  return bits ^ ((bits & 0x80000000U) != 0U ? 0xffffffffU : 0x80000000U);
}

/// Reverse `ordered_float` after the atomic reduction completes.
inline float decode_ordered_float(uint value) {
  const uint bits = (value & 0x80000000U) != 0U ? value ^ 0x80000000U : ~value;
  return as_type<float>(bits);
}

/// Reduce finite scalar values to one min/max pair without host-side scanning.
///
/// The host always dispatches complete 256-thread groups, allowing this fixed
/// threadgroup reduction even when the final group extends beyond `count`.
/// Synthetic colouring enables `collisions_only` so zero-filled miss pixels do
/// not distort the distance or elevation range.
kernel void reduce_finite_range(
    device const float *values [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    device FiniteRange *range [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &collisions_only [[buffer(4)]],
    uint index [[thread_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]]
) {
  threadgroup uint local_minima[256];
  threadgroup uint local_maxima[256];
  const bool valid = index < count && isfinite(values[index]) &&
                     (collisions_only == 0U ||
                      (distances[index] > 0.0F && isfinite(distances[index])));
  const uint ordered = valid ? ordered_float(values[index]) : 0U;
  local_minima[local_index] = valid ? ordered : 0xffffffffU;
  local_maxima[local_index] = ordered;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint offset = 128U; offset != 0U; offset >>= 1U) {
    if (local_index < offset) {
      local_minima[local_index] = min(local_minima[local_index],
                                      local_minima[local_index + offset]);
      local_maxima[local_index] = max(local_maxima[local_index],
                                      local_maxima[local_index + offset]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0U && local_minima[0] != 0xffffffffU) {
    atomic_fetch_min_explicit(&range->minimum, local_minima[0], memory_order_relaxed);
    atomic_fetch_max_explicit(&range->maximum, local_maxima[0], memory_order_relaxed);
  }
}

/// Interpolate one of the five-stop preset colourmap approximations.
inline float3 preset_colourmap(float normalised_value, uint colourmap) {
  constexpr float positions[] = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
  const float value = clamp(normalised_value, 0.0F, 1.0F);
  const uint map_index = min(colourmap, 5U);
  uint upper = 1U;
  while (upper < 4U && value > positions[upper]) {
    upper++;
  }
  const uint lower = upper - 1U;
  const float fraction =
      (value - positions[lower]) / (positions[upper] - positions[lower]);
  const float3 interpolated =
      preset_colourmaps[map_index][lower] +
      fraction * (preset_colourmaps[map_index][upper] - preset_colourmaps[map_index][lower]);
  return floor(clamp(interpolated, 0.0F, 255.0F) + 0.5F) / 255.0F;
}

/// Normalise a finite scalar using a completed atomic range reduction.
inline float normalised_value(float value, device const FiniteRange *range) {
  const float minimum = decode_ordered_float(
      atomic_load_explicit(&range->minimum, memory_order_relaxed)
  );
  const float maximum = decode_ordered_float(
      atomic_load_explicit(&range->maximum, memory_order_relaxed)
  );
  return maximum == minimum ? 0.5F : (value - minimum) / (maximum - minimum);
}

/// Convert a finite scalar field to the diagnostic viridis image.
kernel void present_scalar_viridis(
    device const float *values [[buffer(0)]],
    device const FiniteRange *range [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 position [[thread_position_in_grid]]
) {
  if (position.x >= output.get_width() || position.y >= output.get_height()) {
    return;
  }
  const uint index = position.y * output.get_width() + position.x;
  const float value = values[index];
  const uint ordered_minimum = atomic_load_explicit(&range->minimum, memory_order_relaxed);
  if (!isfinite(value) || ordered_minimum == 0xffffffffU) {
    output.write(float4(0.0F, 0.0F, 0.0F, 1.0F), position);
    return;
  }
  output.write(float4(preset_colourmap(normalised_value(value, range), 0U), 1.0F), position);
}

/// Reconstruct an upward unit normal from the trace kernel's packed half2.
inline float3 surface_normal(uint packed_gradients) {
  const float2 gradients = float2(as_type<half2>(packed_gradients));
  const float inverse_length =
      1.0F / sqrt(gradients.x * gradients.x + gradients.y * gradients.y + 1.0F);
  return float3(-gradients.x * inverse_length, -gradients.y * inverse_length, inverse_length);
}

/// Make diagnostic RGB bytes stable across UNorm texture storage and readback.
inline float3 quantize_unorm8(float3 value) {
  return floor(clamp(value, 0.0F, 1.0F) * 255.0F + 0.5F) / 255.0F;
}

/// Convert packed terrain gradients to a conventional RGB normal map.
kernel void present_surface_normals(
    device const uint *packed_gradients [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 position [[thread_position_in_grid]]
) {
  if (position.x >= output.get_width() || position.y >= output.get_height()) {
    return;
  }
  const uint index = position.y * output.get_width() + position.x;
  const float distance = distances[index];
  const float3 color = distance > 0.0F && isfinite(distance)
                           ? quantize_unorm8(
                                 0.5F * surface_normal(packed_gradients[index]) + 0.5F
                             )
                           : float3(0.0F);
  output.write(float4(color, 1.0F), position);
}

/// Convert an sRGB colour to the linear-light domain used for illumination.
inline float3 srgb_to_linear(float3 value) {
  return select(
      value / 12.92F,
      pow((value + 0.055F) / 1.055F, float3(2.4F)),
      value > 0.04045F
  );
}

/// Convert a shaded linear-light colour back to sRGB for the output texture.
inline float3 linear_to_srgb(float3 value) {
  return select(
      12.92F * value,
      1.055F * pow(value, float3(1.0F / 2.4F)) - 0.055F,
      value > 0.0031308F
  );
}

/// Render white Lambertian terrain under one directional sun and ambient term.
kernel void present_synthetic_terrain(
    device const uint *packed_gradients [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    constant float4 &sun_and_ambient [[buffer(2)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 position [[thread_position_in_grid]]
) {
  if (position.x >= output.get_width() || position.y >= output.get_height()) {
    return;
  }
  const uint index = position.y * output.get_width() + position.x;
  const float distance = distances[index];
  if (!(distance > 0.0F) || !isfinite(distance)) {
    output.write(float4(0.0F, 0.0F, 0.0F, 1.0F), position);
    return;
  }

  const float diffuse = max(0.0F, dot(surface_normal(packed_gradients[index]),
                                      sun_and_ambient.xyz));
  const float linear = sun_and_ambient.w + (1.0F - sun_and_ambient.w) * diffuse;
  const float srgb = linear <= 0.0031308F
                         ? 12.92F * linear
                         : 1.055F * pow(linear, 1.0F / 2.4F) - 0.055F;
  output.write(float4(srgb, srgb, srgb, 1.0F), position);
}

/// Render colourmapped Lambertian terrain under sun and ambient light.
///
/// Keeping this separate from the white kernel prevents the palette and sRGB
/// arithmetic from increasing register pressure in the default render path.
kernel void present_colourmapped_synthetic_terrain(
    device const uint *packed_gradients [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    constant float4 &sun_and_ambient [[buffer(2)]],
    device const float *colour_values [[buffer(3)]],
    device const FiniteRange *range [[buffer(4)]],
    constant uint &colourmap [[buffer(5)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 position [[thread_position_in_grid]]
) {
  if (position.x >= output.get_width() || position.y >= output.get_height()) {
    return;
  }
  const uint index = position.y * output.get_width() + position.x;
  const float distance = distances[index];
  if (!(distance > 0.0F) || !isfinite(distance)) {
    output.write(float4(0.0F, 0.0F, 0.0F, 1.0F), position);
    return;
  }

  const float diffuse = max(0.0F, dot(surface_normal(packed_gradients[index]),
                                      sun_and_ambient.xyz));
  const float illumination = sun_and_ambient.w + (1.0F - sun_and_ambient.w) * diffuse;

  const float3 base_srgb = preset_colourmap(
      normalised_value(colour_values[index], range),
      colourmap
  );
  output.write(float4(linear_to_srgb(srgb_to_linear(base_srgb) * illumination), 1.0F), position);
}

/// Pack the reusable RGBA presentation texture for ImageIO's faster RGB path.
///
/// Interactive rendering consumes the texture directly and never dispatches
/// this kernel; it belongs solely to host readback products such as CLI PNGs.
kernel void pack_presented_rgb(
    texture2d<float, access::read> input [[texture(0)]],
    device packed_uchar3 *output [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]
) {
  if (position.x >= input.get_width() || position.y >= input.get_height()) {
    return;
  }
  const uint index = position.y * input.get_width() + position.x;
  output[index] = packed_uchar3(
      uchar3(floor(clamp(input.read(position).rgb, 0.0F, 1.0F) * 255.0F + 0.5F))
  );
}
