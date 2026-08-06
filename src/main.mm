// Foundation provides Objective-C runtime utilities such as NSString and
// NSError. Metal exposes the GPU device, buffers, and compute API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

// The Makefile supplies the matching debug/release path. Keeping a default
// makes this source usable by an IDE or another build system too.
#ifndef PANORAMA_METALLIB_PATH
#define PANORAMA_METALLIB_PATH "obj/release/panorama.metallib"
#endif

// A .metallib is the compiled Metal shader library loaded at runtime.
constexpr const char *kMetallibPath = PANORAMA_METALLIB_PATH;

// The example is a fixed-size 2D field. It lives in a flat row-major buffer:
// element (x, y) has index y * kFieldWidth + x.
constexpr uint32_t kFieldWidth = 8;
constexpr uint32_t kFieldHeight = 4;

// Parse the optional number of kernel dispatches. The kernel receives scalar
// dimensions as Metal `uint` values, which are unsigned 32-bit integers.
bool parse_iterations(const char *text, uint32_t *iterations) {
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *iterations = static_cast<uint32_t>(parsed);
  return true;
}

// Convert a useful Objective-C error description into command-line output.
void print_error(NSString *context, NSError *error) {
  std::fprintf(stderr, "%s: %s\n", context.UTF8String,
               error.localizedDescription.UTF8String);
}

} // namespace

int main(int argc, const char *argv[]) {
  // Metal objects use Objective-C reference counting. This pool releases
  // temporary Foundation objects before the command-line program exits.
  @autoreleasepool {
    // With no argument, dispatch the multiplication kernel three times.
    uint32_t iterations = 3;
    if (argc == 2 && !parse_iterations(argv[1], &iterations)) {
      std::fprintf(stderr, "usage: %s [positive-iteration-count]\n", argv[0]);
      return EXIT_FAILURE;
    }
    if (argc > 2) {
      std::fprintf(stderr, "usage: %s [positive-iteration-count]\n", argv[0]);
      return EXIT_FAILURE;
    }

    const size_t fieldSize = static_cast<size_t>(kFieldWidth) * kFieldHeight;
    std::vector<float> values(fieldSize);
    std::vector<float> expected(fieldSize);
    for (uint32_t y = 0; y < kFieldHeight; ++y) {
      for (uint32_t x = 0; x < kFieldWidth; ++x) {
        const size_t index = static_cast<size_t>(y) * kFieldWidth + x;
        values[index] = static_cast<float>(index + 1);
        expected[index] = values[index];
      }
    }
    // Calculate the expected result on the CPU with the same float32 operation
    // used by the kernel, solely to validate the GPU demonstration.
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      for (float &value : expected) {
        value *= 2.0F;
      }
    }

    // A device represents the system's default Metal-capable GPU.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "No Metal device is available.\n");
      return EXIT_FAILURE;
    }

    // Load the precompiled shader library, then find the named kernel in it
    // and compile that function into a device-specific pipeline.
    NSError *error = nil;
    NSURL *libraryUrl =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:libraryUrl error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      return EXIT_FAILURE;
    }

    id<MTLFunction> function =
        [library newFunctionWithName:@"multiply_2d_by_two"];
    if (function == nil) {
      std::fprintf(stderr, "Kernel multiply_2d_by_two is missing from %s.\n",
                   kMetallibPath);
      return EXIT_FAILURE;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      print_error(@"Could not create the compute pipeline", error);
      return EXIT_FAILURE;
    }

    // Buffer 0 holds the mutable flat 2D field. Scalar buffer arguments 1 and
    // 2 hold its width and height. These indices must match panorama.metal.
    // Shared storage lets this command-line program read the final field after
    // the command buffer has completed.
    id<MTLBuffer> field =
        [device newBufferWithBytes:values.data()
                            length:values.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
    if (field == nil) {
      std::fprintf(stderr, "Could not allocate the Metal field buffer.\n");
      return EXIT_FAILURE;
    }

    // GPU work is recorded into a command buffer, then submitted through a
    // command queue. The encoder records this compute pass.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:field offset:0 atIndex:0];
    [encoder setBytes:&kFieldWidth length:sizeof(kFieldWidth) atIndex:1];
    [encoder setBytes:&kFieldHeight length:sizeof(kFieldHeight) atIndex:2];

    // Each dispatch is a 2D grid with one useful GPU thread per field element.
    // Repeating the dispatch encodes several calls to the same kernel: every
    // pass reads the values written by the preceding pass and doubles them.
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      [encoder dispatchThreads:MTLSizeMake(kFieldWidth, kFieldHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(kFieldWidth, kFieldHeight, 1)];
    }
    [encoder endEncoding];
    [command commit];
    // Waiting is appropriate for this command-line smoke test. A renderer
    // should normally submit work asynchronously and overlap CPU/GPU work.
    [command waitUntilCompleted];

    if (command.status == MTLCommandBufferStatusError) {
      print_error(@"The Metal command failed", command.error);
      return EXIT_FAILURE;
    }

    // The GPU has finished, so the shared field buffer can now be read.
    const auto *result = static_cast<const float *>(field.contents);
    bool matches = true;
    std::printf("%u passes over a %u x %u field:\n", iterations, kFieldWidth,
                kFieldHeight);
    for (uint32_t y = 0; y < kFieldHeight; ++y) {
      for (uint32_t x = 0; x < kFieldWidth; ++x) {
        const size_t index = static_cast<size_t>(y) * kFieldWidth + x;
        std::printf("%8.1f", result[index]);
        matches = matches && result[index] == expected[index];
      }
      std::printf("\n");
    }
    return matches ? EXIT_SUCCESS : EXIT_FAILURE;
  }
}
