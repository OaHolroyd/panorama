#include "png_writer.h"

#import <Foundation/Foundation.h>

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {

void write_rgb_png(
    const std::filesystem::path &path,
    std::span<const uint8_t> bytes,
    uint32_t width,
    uint32_t height,
    size_t bytes_per_row
) {
  if (width == 0U || height == 0U) {
    throw std::invalid_argument("PNG dimensions must both be positive");
  }
  const size_t active_row_bytes = static_cast<size_t>(width) * 3U;
  if (bytes_per_row < active_row_bytes ||
      (height > 1U && bytes_per_row > (std::numeric_limits<size_t>::max() - active_row_bytes) /
                                          (static_cast<size_t>(height) - 1U))) {
    throw std::invalid_argument("PNG row stride is invalid");
  }
  const size_t required_bytes =
      bytes_per_row * (static_cast<size_t>(height) - 1U) + active_row_bytes;
  if (bytes.size() < required_bytes) {
    throw std::invalid_argument("PNG input is shorter than its dimensions and row stride");
  }

  @autoreleasepool {
    NSString *path_string = [NSString stringWithUTF8String:path.string().c_str()];
    if (path_string == nil) {
      throw std::invalid_argument("PNG path is not valid UTF-8");
    }
    NSURL *url = [NSURL fileURLWithPath:path_string];
    NSData *data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    if (provider == nullptr) {
      throw std::runtime_error("Could not create PNG data provider");
    }
    CGColorSpaceRef color_space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (color_space == nullptr) {
      CGDataProviderRelease(provider);
      throw std::runtime_error("Could not create PNG colour space");
    }
    const CGBitmapInfo bitmap_info = static_cast<CGBitmapInfo>(kCGBitmapByteOrderDefault) |
                                     static_cast<CGBitmapInfo>(kCGImageAlphaNone);
    CGImageRef image = CGImageCreate(
        width,
        height,
        8,
        24,
        bytes_per_row,
        color_space,
        bitmap_info,
        provider,
        nullptr,
        false,
        kCGRenderingIntentDefault
    );
    if (image == nullptr) {
      CGColorSpaceRelease(color_space);
      CGDataProviderRelease(provider);
      throw std::runtime_error("Could not create PNG image");
    }
    CGImageDestinationRef destination =
        CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
    if (destination == nullptr) {
      CGImageRelease(image);
      CGColorSpaceRelease(color_space);
      CGDataProviderRelease(provider);
      throw std::runtime_error("Could not create PNG destination " + path.string());
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

} // namespace panorama
