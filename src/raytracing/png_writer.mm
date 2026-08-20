#include "png_writer.h"

#import <Foundation/Foundation.h>

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
namespace {

/// One normalised position and its sRGB colour in a piecewise-linear palette.
struct ColorStop {
  float position;
  Rgb color;
};

/// Clamp a scalar to the range accepted by every public colormap.
[[nodiscard]] float clamp_unit(float value) { return std::clamp(value, 0.0F, 1.0F); }

/// Round a float colour channel to the nearest valid 8-bit value.
[[nodiscard]] uint8_t to_byte(float value) {
  return static_cast<uint8_t>(std::clamp(value, 0.0F, 255.0F) + 0.5F);
}

/// Expand one IEEE 754 binary16 bit pattern without relying on a host-specific
/// half type. Normal trace output is finite, but preserving infinities and NaNs
/// makes corrupt or future diagnostic data fail visibly.
[[nodiscard]] float decode_binary16(uint16_t bits) {
  const bool negative = (bits & 0x8000U) != 0U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t significand = bits & 0x03ffU;
  float value = 0.0F;
  if (exponent == 0U) {
    value = std::ldexp(static_cast<float>(significand), -24);
  } else if (exponent == 0x1fU) {
    value = significand == 0U ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
  } else {
    value =
        std::ldexp(static_cast<float>(0x0400U + significand), static_cast<int>(exponent) - 25);
  }
  return negative ? -value : value;
}

/// Linearly interpolate an sRGB colour from a sorted set of colour stops.
template <size_t Count>
[[nodiscard]] Rgb
interpolate_colormap(const std::array<ColorStop, Count> &stops, float normalised_value) {
  const float value = clamp_unit(normalised_value);
  for (size_t index = 1; index < Count; index++) {
    if (value <= stops[index].position) {
      const ColorStop &lower = stops[index - 1];
      const ColorStop &upper = stops[index];
      const float fraction = (value - lower.position) / (upper.position - lower.position);
      return {
          to_byte(
              static_cast<float>(lower.color.red) +
              fraction * (static_cast<float>(upper.color.red) - static_cast<float>(lower.color.red))
          ),
          to_byte(
              static_cast<float>(lower.color.green) +
              fraction *
                  (static_cast<float>(upper.color.green) - static_cast<float>(lower.color.green))
          ),
          to_byte(
              static_cast<float>(lower.color.blue) +
              fraction *
                  (static_cast<float>(upper.color.blue) - static_cast<float>(lower.color.blue))
          )};
    }
  }
  return stops.back().color;
}

/// Validate dimensions and calculate the exact number of input pixels.
[[nodiscard]] size_t checked_pixel_count(uint32_t width, uint32_t height) {
  if (width == 0U || height == 0U) {
    throw std::invalid_argument("PNG dimensions must both be positive");
  }
  const size_t pixel_count = static_cast<size_t>(width) * height;
  if (pixel_count / width != height || pixel_count > std::numeric_limits<size_t>::max() / 4U) {
    throw std::invalid_argument("PNG dimensions are too large");
  }
  return pixel_count;
}

/// Find the finite value range used to normalise a diagnostic image.
[[nodiscard]] std::pair<float, float> finite_range(std::span<const float> values) {
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  for (float value : values) {
    if (std::isfinite(value)) {
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
  }
  if (!std::isfinite(minimum)) {
    throw std::invalid_argument("PNG input contains no finite values");
  }
  return {minimum, maximum};
}

/// Encode an already-colormapped, tightly packed RGB buffer with ImageIO.
void encode_png(
    const std::filesystem::path &path,
    const std::vector<uint8_t> &rgb,
    uint32_t width,
    uint32_t height
) {
  @autoreleasepool {
    NSString *path_string = [NSString stringWithUTF8String:path.string().c_str()];
    if (path_string == nil) {
      throw std::invalid_argument("PNG path is not valid UTF-8");
    }
    NSURL *url = [NSURL fileURLWithPath:path_string];
    NSData *data = [NSData dataWithBytes:rgb.data() length:rgb.size()];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGColorSpaceRef color_space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    const CGBitmapInfo bitmap_info = static_cast<CGBitmapInfo>(kCGBitmapByteOrderDefault) |
                                     static_cast<CGBitmapInfo>(kCGImageAlphaNone);
    CGImageRef image = CGImageCreate(
        width,
        height,
        8,
        24,
        static_cast<size_t>(width) * 3U,
        color_space,
        bitmap_info,
        provider,
        nullptr,
        false,
        kCGRenderingIntentDefault
    );
    CGImageDestinationRef destination =
        CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);

    if (provider == nullptr || color_space == nullptr || image == nullptr ||
        destination == nullptr) {
      if (destination != nullptr) {
        CFRelease(destination);
      }
      if (image != nullptr) {
        CGImageRelease(image);
      }
      if (color_space != nullptr) {
        CGColorSpaceRelease(color_space);
      }
      if (provider != nullptr) {
        CGDataProviderRelease(provider);
      }
      throw std::runtime_error("Could not create PNG encoder resources");
    }

    CGImageDestinationAddImage(destination, image, nullptr);
    const bool wrote_file = CGImageDestinationFinalize(destination);
    CFRelease(destination);
    CGImageRelease(image);
    CGColorSpaceRelease(color_space);
    CGDataProviderRelease(provider);
    if (!wrote_file) {
      throw std::runtime_error("Could not write PNG " + path.string());
    }
  }
}

} // namespace

Rgb colormaps::grayscale(float normalised_value) {
  const uint8_t value = to_byte(clamp_unit(normalised_value) * 255.0F);
  return {value, value, value};
}

Rgb colormaps::viridis(float normalised_value) {
  constexpr std::array<ColorStop, 5> kStops = {{{0.0F, {68, 1, 84}},
                                                {0.25F, {59, 82, 139}},
                                                {0.5F, {33, 145, 140}},
                                                {0.75F, {94, 201, 98}},
                                                {1.0F, {253, 231, 37}}}};
  return interpolate_colormap(kStops, normalised_value);
}

Rgb colormaps::jet(float normalised_value) {
  constexpr std::array<ColorStop, 5> kStops = {{{0.0F, {0, 0, 128}},
                                                {0.25F, {0, 0, 255}},
                                                {0.5F, {0, 255, 255}},
                                                {0.75F, {255, 255, 0}},
                                                {1.0F, {128, 0, 0}}}};
  return interpolate_colormap(kStops, normalised_value);
}

void write_colormapped_png(
    const std::filesystem::path &path,
    std::span<const float> values,
    uint32_t width,
    uint32_t height,
    Colormap colormap
) {
  const size_t pixel_count = checked_pixel_count(width, height);
  if (values.size() != pixel_count) {
    throw std::invalid_argument("PNG input length does not match its declared dimensions");
  }
  if (colormap == nullptr) {
    throw std::invalid_argument("PNG colormap must not be null");
  }

  const auto [minimum, maximum] = finite_range(values);
  const float range = maximum - minimum;
  std::vector<uint8_t> rgb(pixel_count * 3U);
  for (size_t index = 0; index < pixel_count; index++) {
    const float value = values[index];
    // Black makes absent rays or other non-finite diagnostic values obvious
    // without preventing the finite part of an image from being inspected.
    const Rgb color = std::isfinite(value)
                          ? colormap(range == 0.0F ? 0.5F : (value - minimum) / range)
                          : Rgb{0, 0, 0};
    rgb[index * 3U] = color.red;
    rgb[index * 3U + 1U] = color.green;
    rgb[index * 3U + 2U] = color.blue;
  }
  encode_png(path, rgb, width, height);
}

void write_surface_normals_png(
    const std::filesystem::path &path,
    std::span<const uint32_t> packed_gradients,
    std::span<const float> distances,
    uint32_t width,
    uint32_t height
) {
  const size_t pixel_count = checked_pixel_count(width, height);
  if (packed_gradients.size() != pixel_count || distances.size() != pixel_count) {
    throw std::invalid_argument("Normal-map input length does not match its declared dimensions");
  }

  std::vector<uint8_t> rgb(pixel_count * 3U);
  for (size_t index = 0; index < pixel_count; index++) {
    if (!(distances[index] > 0.0F) || !std::isfinite(distances[index])) {
      continue;
    }
    const uint32_t packed = packed_gradients[index];
    const float east_gradient = decode_binary16(static_cast<uint16_t>(packed));
    const float north_gradient = decode_binary16(static_cast<uint16_t>(packed >> 16U));
    const float inverse_length =
        1.0F / std::sqrt(east_gradient * east_gradient + north_gradient * north_gradient + 1.0F);
    const float normal_east = -east_gradient * inverse_length;
    const float normal_north = -north_gradient * inverse_length;
    const float normal_up = inverse_length;
    rgb[index * 3U] = to_byte((0.5F * normal_east + 0.5F) * 255.0F);
    rgb[index * 3U + 1U] = to_byte((0.5F * normal_north + 0.5F) * 255.0F);
    rgb[index * 3U + 2U] = to_byte((0.5F * normal_up + 0.5F) * 255.0F);
  }
  encode_png(path, rgb, width, height);
}

} // namespace panorama
