#include <metal_stdlib>

using namespace metal;

/// Per-pixel terrain ray ABI shared with `RayDirection` in ray_projection.h.
/// Keeping the complete layout lets the minimap and outline passes consume the
/// tracer's buffer directly without a repacking pass.
struct PresentationRayDirection {
  float x;
  float y;
  float inverse_x;
  float inverse_y;
  float slope;
};

/// Affine projected-terrain-to-Metal-clip transform for the minimap. The host
/// derives the two basis vectors through the terrain CRS and
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
    device const PresentationRayDirection *rays [[buffer(0)]],
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
/// fragment. Fixed two-pixel coverage keeps cost and opacity predictable.
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

/// Exact 96-colour distance palette extracted from the indexed panorama GIFs
/// published at https://viewfinderpanoramas.org/panoramas.html. It progresses
/// from dark green at the viewpoint to white at the selected maximum distance.
constant uchar3 viewfinder_colourmap[96] = {
    uchar3(0x00, 0x78, 0x00), uchar3(0x00, 0x81, 0x00), uchar3(0x00, 0x8a, 0x00),
    uchar3(0x00, 0x93, 0x00), uchar3(0x00, 0x9c, 0x00), uchar3(0x00, 0xa5, 0x00),
    uchar3(0x00, 0xaf, 0x00), uchar3(0x00, 0xb9, 0x00), uchar3(0x00, 0xc3, 0x00),
    uchar3(0x00, 0xd1, 0x00), uchar3(0x00, 0xdf, 0x00), uchar3(0x00, 0xed, 0x00),
    uchar3(0x00, 0xf6, 0x00), uchar3(0x00, 0xff, 0x00), uchar3(0x14, 0xff, 0x00),
    uchar3(0x28, 0xff, 0x00), uchar3(0x3c, 0xff, 0x00), uchar3(0x50, 0xff, 0x00),
    uchar3(0x60, 0xff, 0x00), uchar3(0x71, 0xff, 0x00), uchar3(0x82, 0xff, 0x00),
    uchar3(0x92, 0xff, 0x00), uchar3(0x9b, 0xff, 0x00), uchar3(0xa4, 0xff, 0x00),
    uchar3(0xad, 0xff, 0x00), uchar3(0xb6, 0xff, 0x00), uchar3(0xbf, 0xff, 0x00),
    uchar3(0xc8, 0xff, 0x00), uchar3(0xd1, 0xff, 0x00), uchar3(0xda, 0xff, 0x00),
    uchar3(0xe3, 0xff, 0x00), uchar3(0xec, 0xff, 0x00), uchar3(0xf5, 0xff, 0x00),
    uchar3(0xff, 0xff, 0x00), uchar3(0xff, 0xf9, 0x00), uchar3(0xff, 0xf6, 0x00),
    uchar3(0xff, 0xf3, 0x00), uchar3(0xff, 0xec, 0x00), uchar3(0xff, 0xe8, 0x00),
    uchar3(0xff, 0xe6, 0x00), uchar3(0xff, 0xe4, 0x00), uchar3(0xff, 0xe3, 0x00),
    uchar3(0xff, 0xe0, 0x00), uchar3(0xff, 0xdb, 0x00), uchar3(0xff, 0xd5, 0x00),
    uchar3(0xff, 0xcf, 0x00), uchar3(0xff, 0xc9, 0x00), uchar3(0xff, 0xc0, 0x00),
    uchar3(0xff, 0xba, 0x00), uchar3(0xff, 0xb4, 0x00), uchar3(0xff, 0xae, 0x00),
    uchar3(0xff, 0xa8, 0x00), uchar3(0xff, 0xa2, 0x00), uchar3(0xff, 0x9b, 0x00),
    uchar3(0xff, 0x91, 0x00), uchar3(0xff, 0x8a, 0x00), uchar3(0xff, 0x85, 0x00),
    uchar3(0xff, 0x7d, 0x00), uchar3(0xff, 0x76, 0x00), uchar3(0xff, 0x6e, 0x00),
    uchar3(0xff, 0x64, 0x00), uchar3(0xff, 0x5a, 0x00), uchar3(0xff, 0x50, 0x00),
    uchar3(0xff, 0x46, 0x00), uchar3(0xff, 0x3c, 0x00), uchar3(0xff, 0x32, 0x00),
    uchar3(0xff, 0x00, 0x00), uchar3(0xff, 0x00, 0x48), uchar3(0xff, 0x00, 0x60),
    uchar3(0xff, 0x00, 0x84), uchar3(0xff, 0x00, 0x90), uchar3(0xff, 0x00, 0x9d),
    uchar3(0xff, 0x00, 0xab), uchar3(0xff, 0x00, 0xb9), uchar3(0xff, 0x00, 0xc7),
    uchar3(0xff, 0x00, 0xd5), uchar3(0xff, 0x00, 0xe1), uchar3(0xff, 0x00, 0xea),
    uchar3(0xff, 0x00, 0xff), uchar3(0xff, 0x3c, 0xff), uchar3(0xff, 0x5a, 0xff),
    uchar3(0xff, 0x78, 0xff), uchar3(0xff, 0x8c, 0xff), uchar3(0xff, 0xa0, 0xff),
    uchar3(0xff, 0xb4, 0xff), uchar3(0xff, 0xb9, 0xff), uchar3(0xff, 0xc0, 0xff),
    uchar3(0xff, 0xc7, 0xff), uchar3(0xff, 0xce, 0xff), uchar3(0xff, 0xd5, 0xff),
    uchar3(0xff, 0xdc, 0xff), uchar3(0xff, 0xe3, 0xff), uchar3(0xff, 0xea, 0xff),
    uchar3(0xff, 0xf1, 0xff), uchar3(0xff, 0xf8, 0xff), uchar3(0xff, 0xff, 0xff),
};

/// Interpolate one of the five-stop preset colourmap approximations.
inline float3 preset_colourmap(float normalised_value, uint colourmap) {
  constexpr float positions[] = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
  const float value = clamp(normalised_value, 0.0F, 1.0F);
  if (colourmap == 6U) {
    const uint index = min(uint(floor(value * 95.0F + 0.5F)), 95U);
    return float3(viewfinder_colourmap[index]) / 255.0F;
  }
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

/// Redistribute a normalised interval without changing its endpoints.
inline float scaled_normalised_value(float value, float2 range, uint scale) {
  const float linear = clamp(normalised_value(value, range), 0.0F, 1.0F);
  switch (scale) {
  case 1U:
    // A log1p curve remains defined when the range starts at zero and gives
    // roughly two decades of extra resolution to nearby/lower values.
    return log2(1.0F + 255.0F * linear) * 0.125F;
  case 2U:
    return sqrt(linear);
  case 3U:
    return linear * linear;
  default:
    return linear;
  }
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

/// Reconstruct a terrain collision in the observer's local east/north/up
/// frame. Tracing stores horizontal distance; its ray slope and effective-
/// Earth curvature recover the vertical component without consulting DEM
/// normals.
inline float3 collision_point(PresentationRayDirection ray, float distance, float curvature) {
  return float3(
      distance * ray.x,
      distance * ray.y,
      distance * ray.slope + curvature * distance * distance
  );
}

/// Measure how much farther apart two collision points are than adjacent rays
/// at their nearer slant range would normally be. A continuous front-facing
/// surface is close to one; separated surfaces and grazing ridges are larger.
inline float surface_separation(
    device const PresentationRayDirection *rays,
    device const float *distances,
    uint index,
    uint neighbour_index,
    float curvature
) {
  const float distance = distances[index];
  const float neighbour_distance = distances[neighbour_index];
  if (!(neighbour_distance > distance) || !isfinite(neighbour_distance)) {
    return 0.0F;
  }

  const PresentationRayDirection ray = rays[index];
  const PresentationRayDirection neighbour_ray = rays[neighbour_index];
  const float3 ray_vector = float3(ray.x, ray.y, ray.slope);
  const float3 neighbour_ray_vector = float3(neighbour_ray.x, neighbour_ray.y, neighbour_ray.slope);
  const float ray_length = length(ray_vector);
  const float neighbour_ray_length = length(neighbour_ray_vector);
  const float angular_chord =
      length(ray_vector / ray_length - neighbour_ray_vector / neighbour_ray_length);
  const float nearer_slant_range =
      min(distance * ray_length, neighbour_distance * neighbour_ray_length);
  const float expected_spacing = max(nearer_slant_range * angular_chord, 1.0F);
  const float actual_spacing = length(
      collision_point(neighbour_ray, neighbour_distance, curvature) -
      collision_point(ray, distance, curvature)
  );
  return actual_spacing / expected_spacing;
}

/// Generate a reusable one-byte outline mask from geometric separation at
/// one-, two-, and four-pixel baselines. Fine-scale evidence localises the
/// result while agreement at a wider scale rejects isolated depth noise.
kernel void compute_feature_outlines(
    device const PresentationRayDirection *rays [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    constant float &detail [[buffer(2)]],
    constant float &curvature [[buffer(3)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 position [[thread_position_in_grid]]
) {
  const uint width = output.get_width();
  const uint height = output.get_height();
  if (position.x >= width || position.y >= height) {
    return;
  }
  const uint index = position.y * width + position.x;
  const float distance = distances[index];
  if (!(distance > 0.0F) || !isfinite(distance)) {
    output.write(float4(0.0F), position);
    return;
  }

  const float sensitivity = clamp(detail, 0.0F, 1.0F);
  // Bias most of the control toward major separations. The upper end still
  // exposes fine grazing ridges without making the default image excessively
  // dense.
  const float separation_threshold = mix(24.0F, 6.0F, sensitivity);
  // A fixed physical gap occupies a smaller fraction of the expected spacing
  // at wider baselines, hence the falling thresholds. Fine evidence remains
  // mandatory below so coarse tests validate an edge without broadening it.
  const float fine_threshold = max(1.35F, 0.45F * separation_threshold);
  const float medium_threshold = max(1.5F, 0.75F * separation_threshold);
  const float coarse_threshold = max(1.25F, 0.55F * separation_threshold);
  constexpr int2 directions[] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
  constexpr int radii[] = {1, 2, 4};

  bool outlined = false;
  for (uint direction_index = 0U; direction_index < 4U && !outlined; direction_index++) {
    float evidence[3] = {0.0F, 0.0F, 0.0F};
    bool immediate_sky = false;
    for (uint scale = 0U; scale < 3U; scale++) {
      const int2 neighbour = int2(position) + radii[scale] * directions[direction_index];
      if (neighbour.x < 0 || neighbour.y < 0 || neighbour.x >= int(width) ||
          neighbour.y >= int(height)) {
        continue;
      }
      const uint neighbour_index = uint(neighbour.y) * width + uint(neighbour.x);
      const float neighbour_distance = distances[neighbour_index];
      if (!(neighbour_distance > 0.0F) || !isfinite(neighbour_distance)) {
        immediate_sky = immediate_sky || scale == 0U;
        continue;
      }
      evidence[scale] = surface_separation(rays, distances, index, neighbour_index, curvature);
    }

    const bool fine = evidence[0] > fine_threshold;
    const bool medium = evidence[1] > medium_threshold;
    const bool coarse = evidence[2] > coarse_threshold;
    outlined = immediate_sky || (fine && (medium || coarse));
  }
  output.write(float4(outlined ? 1.0F : 0.0F), position);
}

/// Approximate atmospheric light with a zenith lobe and four diagonal lobes.
/// Sunward diagonal weights add aspect detail; normalisation keeps horizontal
/// terrain at the configured sky strength.
inline float sky_lobe_exposure(float3 normal, float3 sun, float detail) {
  constexpr float diagonal_z = 0.70710678F;
  constexpr float diagonal_xy = 0.5F;
  const float2 sun_horizontal =
      length_squared(sun.xy) > 1e-8F ? normalize(sun.xy) : float2(0.0F, 1.0F);
  const float3 directions[4] = {
      float3(diagonal_xy, diagonal_xy, diagonal_z),
      float3(-diagonal_xy, diagonal_xy, diagonal_z),
      float3(-diagonal_xy, -diagonal_xy, diagonal_z),
      float3(diagonal_xy, -diagonal_xy, diagonal_z),
  };
  float irradiance = 1.5F * max(normal.z, 0.0F);
  float horizontal_irradiance = 1.5F;
  for (uint index = 0U; index < 4U; index++) {
    const float weight =
        1.0F + 0.75F * max(dot(normalize(directions[index].xy), sun_horizontal), 0.0F);
    irradiance += weight * max(dot(normal, directions[index]), 0.0F);
    horizontal_irradiance += weight * diagonal_z;
  }
  const float normalised = clamp(irradiance / horizontal_irradiance, 0.0F, 1.0F);
  // Preserve scattered fill on surfaces facing away from every sampled lobe.
  const float detailed = mix(0.35F, 1.0F, normalised);
  return mix(1.0F, detailed, detail);
}

/// Render white Lambertian terrain under one directional sun and skylight.
kernel void present_synthetic_terrain(
    device const uint *packed_gradients [[buffer(0)]],
    device const float *distances [[buffer(1)]],
    constant float4 &sun_and_ambient [[buffer(2)]],
    constant uint &use_surface_normals [[buffer(3)]],
    constant float &diffusivity [[buffer(4)]],
    constant uint &feature_outlines [[buffer(5)]],
    device const uchar *shadow_visibility [[buffer(6)]],
    constant uint &use_shadows [[buffer(7)]],
    constant float &ambient_detail [[buffer(8)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float, access::read> feature_outline_mask [[texture(1)]],
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
  if (feature_outlines != 0U && feature_outline_mask.read(position).r >= 0.5F) {
    output.write(float4(0.0F, 0.0F, 0.0F, 1.0F), position);
    return;
  }
  if (use_surface_normals == 0U) {
    output.write(float4(1.0F), position);
    return;
  }

  const float3 normal = surface_normal(packed_gradients[index]);
  const float visible = use_shadows == 0U ? 1.0F : float(shadow_visibility[index] != 0U);
  const float diffuse = visible * max(0.0F, dot(normal, sun_and_ambient.xyz));
  const float sky =
      sun_and_ambient.w * sky_lobe_exposure(normal, sun_and_ambient.xyz, ambient_detail);
  const float linear = sky + diffusivity * (1.0F - sun_and_ambient.w) * diffuse;
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
    constant uint &colour_scale [[buffer(6)]],
    constant uint &use_surface_normals [[buffer(7)]],
    constant float &diffusivity [[buffer(8)]],
    constant uint &feature_outlines [[buffer(9)]],
    device const uchar *shadow_visibility [[buffer(10)]],
    constant uint &use_shadows [[buffer(11)]],
    constant float &ambient_detail [[buffer(12)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float, access::read> feature_outline_mask [[texture(1)]],
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
  if (feature_outlines != 0U && feature_outline_mask.read(position).r >= 0.5F) {
    output.write(float4(0.0F, 0.0F, 0.0F, 1.0F), position);
    return;
  }

  const float3 base_srgb = preset_colourmap(
      scaled_normalised_value(colour_values[index], range, colour_scale),
      colourmap
  );
  if (use_surface_normals == 0U) {
    output.write(float4(base_srgb, 1.0F), position);
    return;
  }
  const float3 normal = surface_normal(packed_gradients[index]);
  const float visible = use_shadows == 0U ? 1.0F : float(shadow_visibility[index] != 0U);
  const float diffuse = visible * max(0.0F, dot(normal, sun_and_ambient.xyz));
  const float sky =
      sun_and_ambient.w * sky_lobe_exposure(normal, sun_and_ambient.xyz, ambient_detail);
  const float illumination = sky + diffusivity * (1.0F - sun_and_ambient.w) * diffuse;
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
