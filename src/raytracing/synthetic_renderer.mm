#include "synthetic_renderer.h"

#include "png_writer.h"
#include "surface_normal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace panorama {
namespace {

/// Unit light direction in the same east/north/up basis as `SurfaceNormal`.
struct SunDirection {
  float east;
  float north;
  float up;
};

[[nodiscard]] float clamp_unit(float value) { return std::clamp(value, 0.0F, 1.0F); }

/// Encode one linear-light channel with the standard sRGB transfer function.
[[nodiscard]] uint8_t linear_channel(float value) {
  const float linear = std::max(value, 0.0F);
  const float srgb = linear <= 0.0031308F
                         ? 12.92F * linear
                         : 1.055F * std::pow(linear, 1.0F / 2.4F) - 0.055F;
  return static_cast<uint8_t>(255.0F * clamp_unit(srgb) + 0.5F);
}

/// Convert compass bearing and elevation into an east/north/up unit vector.
[[nodiscard]] SunDirection sun_direction(const SyntheticRenderOptions &options) {
  const float azimuth = static_cast<float>(options.sun_azimuth);
  const float elevation = static_cast<float>(options.sun_elevation);
  const float horizontal = std::cos(elevation);
  return {
      std::sin(azimuth) * horizontal,
      std::cos(azimuth) * horizontal,
      std::sin(elevation),
  };
}

/// Return linear grayscale intensity for a white Lambertian surface.
[[nodiscard]] float shade_terrain(
    SurfaceNormal normal,
    SunDirection sun,
    float ambient_light
) {
  const float diffuse = std::max(
      0.0F, normal.east * sun.east + normal.north * sun.north + normal.up * sun.up
  );
  return ambient_light + (1.0F - ambient_light) * diffuse;
}

/// Validate dimensions before allocating the row-major output image.
[[nodiscard]] size_t checked_pixel_count(ImageSize image) {
  if (image.width == 0U || image.height == 0U) {
    throw std::invalid_argument("Synthetic image dimensions must be positive");
  }
  const size_t count = static_cast<size_t>(image.width) * image.height;
  if (count / image.width != image.height) {
    throw std::overflow_error("Synthetic image dimensions overflow host storage");
  }
  return count;
}

} // namespace

void write_synthetic_terrain_png(
    const std::filesystem::path &path,
    std::span<const uint32_t> packed_gradients,
    std::span<const float> distances,
    ImageSize image,
    const SyntheticRenderOptions &options
) {
  const size_t pixel_count = checked_pixel_count(image);
  if (packed_gradients.size() != pixel_count || distances.size() != pixel_count) {
    throw std::invalid_argument("Synthetic image inputs do not match its declared dimensions");
  }
  if (!std::isfinite(options.sun_azimuth) || !std::isfinite(options.sun_elevation) ||
      !std::isfinite(options.ambient_light) || options.ambient_light < 0.0F ||
      options.ambient_light > 1.0F) {
    throw std::invalid_argument("Synthetic render options are invalid");
  }

  const SunDirection sun = sun_direction(options);
  std::vector<Rgb> pixels(pixel_count);
  for (size_t index = 0; index < pixel_count; index++) {
    if (!(distances[index] > 0.0F) || !std::isfinite(distances[index])) {
      pixels[index] = {0U, 0U, 0U};
      continue;
    }

    const SurfaceNormal normal = decode_surface_normal(packed_gradients[index]);
    const uint8_t intensity = linear_channel(shade_terrain(normal, sun, options.ambient_light));
    pixels[index] = {intensity, intensity, intensity};
  }
  write_rgb_png(path, pixels, image.width, image.height);
}

} // namespace panorama
