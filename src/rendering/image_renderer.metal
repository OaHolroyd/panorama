#include <metal_stdlib>

using namespace metal;

/// Per-pixel terrain ray ABI shared with `RayDirection` in ray_projection.h.
/// The visibility overlay only consumes the normalized horizontal direction,
/// but retaining the complete layout lets it read the tracer's buffer directly.
struct VisibilityRayDirection {
  float x;
  float y;
  float inverse_x;
  float inverse_y;
  float slope;
};

/// Affine projected-terrain-to-Metal-clip transform for the small fixed-centre
/// minimap. The host derives the two basis vectors through the terrain CRS and
/// MapKit, preserving local grid convergence relative to geographic north.
struct VisibilityMapParameters {
  float centre_x;
  float centre_y;
  float east_x_per_metre;
  float east_y_per_metre;
  float north_x_per_metre;
  float north_y_per_metre;
  float point_size;
  uint ray_count;
};

struct VisibilityPointVertex {
  float4 position [[position]];
  float point_size [[point_size]];
};

/// Snapshot one immutable observer-relative east/north point for each completed
/// collision. The viewer publishes this buffer with the matching frame so a
/// subsequent trace can safely reuse its ray and distance storage.
kernel void visibility_collision_points(
    device const VisibilityRayDirection *rays [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    device float2 *points [[buffer(2)]],
    constant uint &ray_count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
  if (index >= ray_count) {
    return;
  }
  const float distance = distances[index];
  points[index] = distance > 0.0F && isfinite(distance)
                      ? distance * float2(rays[index].x, rays[index].y)
                      : float2(INFINITY);
}

/// Project every completed terrain collision directly into the minimap.
/// Invalid/no-hit rays are moved outside the clip volume and therefore emit no
/// fragment. Point coverage is deliberately fixed for this first implementation.
vertex VisibilityPointVertex visibility_point_vertex(
    device const float2 *points [[buffer(0)]],
    constant VisibilityMapParameters &map [[buffer(1)]],
    uint index [[vertex_id]]
) {
  VisibilityPointVertex output;
  output.point_size = map.point_size;
  if (index >= map.ray_count) {
    output.position = float4(2.0F, 2.0F, 0.0F, 1.0F);
    return output;
  }

  const float2 point = points[index];
  if (!all(isfinite(point))) {
    output.position = float4(2.0F, 2.0F, 0.0F, 1.0F);
    return output;
  }

  output.position = float4(
      map.centre_x + point.x * map.east_x_per_metre + point.y * map.north_x_per_metre,
      map.centre_y + point.x * map.east_y_per_metre + point.y * map.north_y_per_metre,
      0.0F,
      1.0F
  );
  return output;
}

/// Emit a constant translucent system-blue-like highlight. RGB is
/// premultiplied because Core Animation composites the transparent Metal layer.
fragment float4 visibility_point_fragment() {
  constexpr float alpha = 0.34F;
  constexpr float3 colour = float3(0.0F, 0.48F, 1.0F);
  return float4(alpha * colour, alpha);
}

/// Five-stop approximations of the CLI's built-in colourmaps.
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
  const float fraction = (value - positions[lower]) / (positions[upper] - positions[lower]);
  const float3 interpolated =
      preset_colourmaps[map_index][lower] +
      fraction * (preset_colourmaps[map_index][upper] - preset_colourmaps[map_index][lower]);
  return floor(clamp(interpolated, 0.0F, 255.0F) + 0.5F) / 255.0F;
}

/// Normalise a scalar over a caller-selected fixed interval.
inline float normalised_value(float value, float2 range) {
  return (value - range.x) / (range.y - range.x);
}

/// Convert a finite scalar field to the diagnostic viridis image.
kernel void present_scalar_viridis(
    device const float *values [[buffer(0)]],
    constant float2 &range [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 position [[thread_position_in_grid]]
) {
  if (position.x >= output.get_width() || position.y >= output.get_height()) {
    return;
  }
  const uint index = position.y * output.get_width() + position.x;
  const float value = values[index];
  if (!isfinite(value)) {
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
                           ? quantize_unorm8(0.5F * surface_normal(packed_gradients[index]) + 0.5F)
                           : float3(0.0F);
  output.write(float4(color, 1.0F), position);
}

/// Convert an sRGB colour to the linear-light domain used for illumination.
inline float3 srgb_to_linear(float3 value) {
  return select(value / 12.92F, pow((value + 0.055F) / 1.055F, float3(2.4F)), value > 0.04045F);
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
    constant uint &use_surface_normals [[buffer(3)]],
    constant float &diffusivity [[buffer(4)]],
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
  if (use_surface_normals == 0U) {
    output.write(float4(1.0F), position);
    return;
  }

  const float diffuse =
      max(0.0F, dot(surface_normal(packed_gradients[index]), sun_and_ambient.xyz));
  const float linear = sun_and_ambient.w + diffusivity * (1.0F - sun_and_ambient.w) * diffuse;
  const float srgb =
      linear <= 0.0031308F ? 12.92F * linear : 1.055F * pow(linear, 1.0F / 2.4F) - 0.055F;
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
    constant float2 &range [[buffer(4)]],
    constant uint &colourmap [[buffer(5)]],
    constant uint &use_surface_normals [[buffer(6)]],
    constant float &diffusivity [[buffer(7)]],
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

  const float3 base_srgb =
      preset_colourmap(normalised_value(colour_values[index], range), colourmap);
  if (use_surface_normals == 0U) {
    output.write(float4(base_srgb, 1.0F), position);
    return;
  }
  const float diffuse =
      max(0.0F, dot(surface_normal(packed_gradients[index]), sun_and_ambient.xyz));
  const float illumination = sun_and_ambient.w + diffusivity * (1.0F - sun_and_ambient.w) * diffuse;
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
  output[index] =
      packed_uchar3(uchar3(floor(clamp(input.read(position).rgb, 0.0F, 1.0F) * 255.0F + 0.5F)));
}
