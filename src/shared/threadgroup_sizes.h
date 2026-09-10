#pragma once

// One source of truth for compute-dispatch shapes shared by Objective-C++
// encoders and Metal kernels. Keep each width a multiple of the SIMD width.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#import <Metal/Metal.h>
#include <cstdint>
#endif

namespace panorama::threadgroups {

enum : unsigned {
  linear_width = 256U,
  linear_height = 1U,
  linear_depth = 1U,
  bvh_width = 32U,
  bvh_height = 16U,
  bvh_depth = 1U,
  spatial_width = 32U,
  spatial_height = 32U,
  spatial_depth = 1U,
};

#ifndef __METAL_VERSION__
inline constexpr MTLSize linear = {linear_width, linear_height, linear_depth};
inline constexpr MTLSize bvh = {bvh_width, bvh_height, bvh_depth};
inline constexpr MTLSize spatial = {spatial_width, spatial_height, spatial_depth};

/// Some specialized pipelines expose fewer threads than the default linear
/// policy. Preserve the one-dimensional shape while respecting that limit.
[[nodiscard]] inline constexpr MTLSize bounded_linear(NSUInteger maximum_threads) {
  return {
      maximum_threads < linear.width ? maximum_threads : linear.width,
      linear.height,
      linear.depth,
  };
}

/// Ray-intersection specializations may lower Metal's pipeline-specific
/// maximum. Retain neighboring columns and reduce only the configured height.
[[nodiscard]] inline constexpr MTLSize bounded_bvh(NSUInteger maximum_threads) {
  const NSUInteger supported_height = maximum_threads / bvh.width;
  return {
      bvh.width,
      supported_height < bvh.height ? supported_height : bvh.height,
      bvh.depth,
  };
}
#endif

} // namespace panorama::threadgroups
