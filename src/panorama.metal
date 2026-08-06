#include <metal_stdlib>

// Metal Shading Language provides GPU-specific types and functions in this
// namespace, including `uint`, `device`, and the `kernel` entry-point keyword.
using namespace metal;

// A `kernel` function is an entry point that the CPU can dispatch on the GPU.
// The `[[buffer(n)]]` attributes are the ABI between this file and main.mm:
//
//   buffer(0): mutable, row-major float32 field
//   buffer(1): field width, copied by the host
//   buffer(2): field height, copied by the host
//
// `thread_position_in_grid` gives every invocation its 2D position. The host
// dispatches one useful thread for each field element, then calls this same
// kernel repeatedly to demonstrate successive compute passes.
kernel void multiply_2d_by_two(device float *field [[buffer(0)]],
                               constant uint &width [[buffer(1)]],
                               constant uint &height [[buffer(2)]],
                               uint2 position [[thread_position_in_grid]]) {
  if (position.x >= width || position.y >= height) {
    return;
  }

  // Metal buffers are one-dimensional. Flatten the 2D coordinate using the
  // same row-major formula used by the host, then update this one element.
  const uint index = position.y * width + position.x;
  field[index] *= 2.0F;
}
