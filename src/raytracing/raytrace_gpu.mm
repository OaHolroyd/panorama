#include "raytrace_gpu.h"

#import <Foundation/Foundation.h>

#include "timer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

#ifndef PANORAMA_METALLIB_PATH
#define PANORAMA_METALLIB_PATH "obj/release/panorama.metallib"
#endif

constexpr const char *kMetallibPath = PANORAMA_METALLIB_PATH;

static_assert(sizeof(HorizontalDirection) == 4U * sizeof(float));
static_assert(sizeof(RaytraceParameters) == 8U * sizeof(uint32_t));
static_assert(sizeof(TileWorkItem) == 5U * sizeof(uint32_t));
static_assert(sizeof(DeferredTileWork) == 3U * sizeof(uint32_t));

/// Print a Foundation error in the command-line form used by host tools.
void print_error(NSString *context, NSError *error) {
  std::fprintf(stderr, "%s: %s\n", context.UTF8String, error.localizedDescription.UTF8String);
}

/// Check that a byte count fits Metal's NSUInteger buffer-length argument.
[[nodiscard]] NSUInteger checked_buffer_length(size_t count, size_t size, const char *name) {
  if (count > std::numeric_limits<size_t>::max() / size) {
    throw std::overflow_error(std::string(name) + " buffer is too large");
  }
  return static_cast<NSUInteger>(count * size);
}

/// Allocate a shared Metal buffer, optionally copying immutable input bytes.
[[nodiscard]] id<MTLBuffer>
make_buffer(id<MTLDevice> device, const void *data, NSUInteger length, const char *name) {
  id<MTLBuffer> buffer = data == nullptr ? [device newBufferWithLength:length
                                                               options:MTLResourceStorageModeShared]
                                         : [device newBufferWithBytes:data
                                                               length:length
                                                              options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(std::string("Could not allocate ") + name + " Metal buffer");
  }
  return buffer;
}

/// Clear a newly allocated shared output/counter buffer before first use.
void clear_buffer(id<MTLBuffer> buffer, const char *name) {
  void *contents = buffer.contents;
  if (contents == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name + " Metal buffer");
  }
  std::memset(contents, 0, buffer.length);
}

} // namespace

/// Mutable Objective-C++ state hidden from the scheduling implementation.
struct GpuRaytraceResources::State {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> trace_pipeline;
  id<MTLComputePipelineState> emit_pipeline;
  id<MTLBuffer> azimuths;
  id<MTLBuffer> slopes;
  id<MTLBuffer> distance_output;
  id<MTLBuffer> elevation_output;
  id<MTLBuffer> work_a;
  id<MTLBuffer> work_b;
  id<MTLBuffer> active;
  id<MTLBuffer> next;
  id<MTLBuffer> unresolved;
  id<MTLBuffer> next_count;
  id<MTLBuffer> deferred_items;
  id<MTLBuffer> deferred_count;
  uint32_t frontier_capacity;
  bool trace_quantized;
  bool capture_active = false;
};

GpuRaytraceResources::GpuRaytraceResources(
    std::span<const HorizontalDirection> directions,
    std::span<const float> slopes,
    size_t ray_count,
    uint32_t frontier_capacity,
    bool trace_quantized
) {
  if (directions.empty() || slopes.empty() || frontier_capacity == 0U) {
    throw std::invalid_argument("GPU raytrace resources require nonempty rays and frontier");
  }
  auto state = std::make_unique<State>();
  state->frontier_capacity = frontier_capacity;
  state->trace_quantized = trace_quantized;
  state->device = MTLCreateSystemDefaultDevice();
  if (state->device == nil) {
    throw std::runtime_error("No Metal device is available");
  }
  state->queue = [state->device newCommandQueue];
  if (state->queue == nil) {
    throw std::runtime_error("Could not create Metal command queue");
  }

  NSError *error = nil;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
  id<MTLLibrary> library = [state->device newLibraryWithURL:url error:&error];
  if (library == nil) {
    print_error(@"Could not load the Metal library", error);
    throw std::runtime_error("Could not load Metal library");
  }
  NSString *trace_name =
      trace_quantized ? @"trace_tile_frontier_quantized" : @"trace_tile_frontier";
  id<MTLFunction> trace = [library newFunctionWithName:trace_name];
  id<MTLFunction> emit = [library newFunctionWithName:@"emit_tile_frontier"];
  if (trace == nil || emit == nil) {
    throw std::runtime_error("GPU-frontier Metal kernels are missing");
  }
  state->trace_pipeline = [state->device newComputePipelineStateWithFunction:trace error:&error];
  if (state->trace_pipeline == nil) {
    print_error(@"Could not create trace pipeline", error);
    throw std::runtime_error("Could not create trace pipeline");
  }
  state->emit_pipeline = [state->device newComputePipelineStateWithFunction:emit error:&error];
  if (state->emit_pipeline == nil) {
    print_error(@"Could not create continuation pipeline", error);
    throw std::runtime_error("Could not create continuation pipeline");
  }

  // Static angular inputs and output images persist for every frontier pass.
  state->azimuths = make_buffer(
      state->device,
      directions.data(),
      checked_buffer_length(directions.size(), sizeof(HorizontalDirection), "azimuth directions"),
      "azimuth directions"
  );
  state->slopes = make_buffer(
      state->device,
      slopes.data(),
      checked_buffer_length(slopes.size(), sizeof(float), "polar slopes"),
      "polar slopes"
  );
  state->distance_output = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(ray_count, sizeof(float), "distance output"),
      "distance output"
  );
  state->elevation_output = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(ray_count, sizeof(float), "elevation output"),
      "elevation output"
  );
  clear_buffer(state->distance_output, "distance output");
  clear_buffer(state->elevation_output, "elevation output");

  // The two frontier buffers are swapped after each completed trace/emit pass.
  state->work_a = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(frontier_capacity, sizeof(TileWorkItem), "active frontier"),
      "active frontier"
  );
  state->work_b = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(frontier_capacity, sizeof(TileWorkItem), "next frontier"),
      "next frontier"
  );
  state->unresolved = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(frontier_capacity, sizeof(uint32_t), "unresolved polar indices"),
      "unresolved polar indices"
  );
  state->next_count = make_buffer(state->device, nullptr, sizeof(uint32_t), "next frontier count");
  state->deferred_items = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(frontier_capacity, sizeof(DeferredTileWork), "deferred frontier"),
      "deferred frontier"
  );
  state->deferred_count =
      make_buffer(state->device, nullptr, sizeof(uint32_t), "deferred frontier count");
  state->active = state->work_a;
  state->next = state->work_b;
  state_ = std::move(state);
}

GpuRaytraceResources::~GpuRaytraceResources() { stop_capture(); }

void GpuRaytraceResources::initialise_frontier(uint32_t num_azimuth) {
  State &state = *state_;
  if (num_azimuth > state.frontier_capacity) {
    throw std::invalid_argument("Initial GPU frontier exceeds its capacity");
  }
  auto *items = static_cast<TileWorkItem *>(state.active.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map active frontier buffer");
  }
  for (uint32_t azimuth = 0U; azimuth < num_azimuth; azimuth++) {
    items[azimuth] = {0U, azimuth, 0U, 1U, 0.0F};
  }
}

id<MTLDevice> GpuRaytraceResources::device() const { return state_->device; }

id<MTLBuffer> GpuRaytraceResources::active_frontier() const { return state_->active; }

void GpuRaytraceResources::swap_frontiers() {
  id<MTLBuffer> temporary = state_->active;
  state_->active = state_->next;
  state_->next = temporary;
}

GpuFrontierPassResult GpuRaytraceResources::trace_frontier(
    const ResidentTileCacheBindings &cache,
    const RaytraceParameters &parameters,
    uint32_t mipmap_value_count,
    uint32_t active_count,
    Timer &timer
) {
  State &state = *state_;
  if (active_count > state.frontier_capacity) {
    throw std::invalid_argument("GPU frontier exceeds its capacity");
  }
  auto *first_unresolved = static_cast<uint32_t *>(state.unresolved.contents);
  auto *next_total = static_cast<uint32_t *>(state.next_count.contents);
  auto *deferred_total = static_cast<uint32_t *>(state.deferred_count.contents);
  const auto *active_items = static_cast<const TileWorkItem *>(state.active.contents);
  if (first_unresolved == nullptr || next_total == nullptr || deferred_total == nullptr ||
      active_items == nullptr) {
    throw std::runtime_error("Could not map GPU frontier buffers");
  }
  uint32_t polar_offset = parameters.num_polar;
  for (uint32_t index = 0U; index < active_count; index++) {
    polar_offset = std::min(polar_offset, active_items[index].first_polar);
  }
  // Every item has already resolved the polar prefix below `first_polar`.
  // Trim the prefix common to this pass while retaining one rectangular
  // dispatch, which gives the GPU substantially better occupancy than many
  // exact but narrow per-suffix dispatches.
  const uint32_t dispatched_polar_count = parameters.num_polar - polar_offset;
  std::fill_n(first_unresolved, active_count, parameters.num_polar);
  *next_total = 0U;
  *deferred_total = 0U;

  timer.start_wall("GPU command encoding");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create GPU frontier command");
  }
  command.label = @"GPU tile frontier";
  encoder.label = @"trace_tile_frontier";
  [encoder setComputePipelineState:state.trace_pipeline];
  [encoder setBuffer:cache.mipmap_atlas offset:0 atIndex:0];
  [encoder setBuffer:cache.vertex_atlas offset:0 atIndex:1];
  [encoder setBuffer:state.azimuths offset:0 atIndex:2];
  [encoder setBuffer:state.slopes offset:0 atIndex:3];
  [encoder setBuffer:state.active offset:0 atIndex:4];
  [encoder setBuffer:cache.metadata offset:0 atIndex:5];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
  [encoder setBytes:&mipmap_value_count length:sizeof(mipmap_value_count) atIndex:7];
  [encoder setBuffer:state.distance_output offset:0 atIndex:8];
  [encoder setBuffer:state.elevation_output offset:0 atIndex:9];
  [encoder setBuffer:state.unresolved offset:0 atIndex:10];
  if (state.trace_quantized) {
    [encoder setBytes:&cache.quantized_layout length:sizeof(cache.quantized_layout) atIndex:11];
  }
  [encoder setBytes:&polar_offset length:sizeof(polar_offset) atIndex:12];
  [encoder dispatchThreads:MTLSizeMake(dispatched_polar_count, active_count, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
  [encoder endEncoding];

  encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Could not create successor frontier encoder");
  }
  encoder.label = @"emit_tile_frontier";
  [encoder setComputePipelineState:state.emit_pipeline];
  [encoder setBuffer:state.active offset:0 atIndex:0];
  [encoder setBuffer:cache.metadata offset:0 atIndex:1];
  [encoder setBuffer:state.azimuths offset:0 atIndex:2];
  [encoder setBuffer:state.unresolved offset:0 atIndex:3];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
  [encoder setBuffer:cache.hash offset:0 atIndex:5];
  [encoder setBytes:&cache.hash_slot_count length:sizeof(cache.hash_slot_count) atIndex:6];
  [encoder setBytes:&state.frontier_capacity length:sizeof(state.frontier_capacity) atIndex:7];
  [encoder setBuffer:state.next offset:0 atIndex:8];
  [encoder setBuffer:state.next_count offset:0 atIndex:9];
  [encoder setBuffer:state.deferred_items offset:0 atIndex:10];
  [encoder setBuffer:state.deferred_count offset:0 atIndex:11];
  [encoder dispatchThreads:MTLSizeMake(active_count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [encoder endEncoding];
  timer.stop("GPU command encoding");

  timer.start_wall("GPU command wait");
  [command commit];
  [command waitUntilCompleted];
  timer.stop("GPU command wait");
  if (command.status == MTLCommandBufferStatusError) {
    print_error(@"GPU frontier Metal command failed", command.error);
    throw std::runtime_error("GPU frontier Metal command failed");
  }
  const double device_milliseconds = 1'000.0 * (command.GPUEndTime - command.GPUStartTime);
  return {*next_total, *deferred_total, device_milliseconds};
}

std::span<const DeferredTileWork> GpuRaytraceResources::deferred_work(uint32_t count) const {
  if (count > state_->frontier_capacity) {
    throw std::invalid_argument("Deferred GPU frontier exceeds its capacity");
  }
  const auto *items = static_cast<const DeferredTileWork *>(state_->deferred_items.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map deferred frontier buffer");
  }
  return {items, count};
}

id<MTLBuffer> GpuRaytraceResources::distances() const { return state_->distance_output; }
id<MTLBuffer> GpuRaytraceResources::elevations() const { return state_->elevation_output; }

void GpuRaytraceResources::start_capture_if_requested() {
  const char *value = std::getenv("MTL_CAPTURE_ENABLED");
  if (value == nullptr || std::strcmp(value, "1") != 0) {
    return;
  }
  NSString *path = [[NSFileManager.defaultManager currentDirectoryPath]
      stringByAppendingPathComponent:@"panorama.gputrace"];
  if ([NSFileManager.defaultManager fileExistsAtPath:path]) {
    throw std::runtime_error("Refusing to overwrite panorama.gputrace; move or remove it first");
  }
  MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
  descriptor.captureObject = state_->queue;
  descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
  descriptor.outputURL = [NSURL fileURLWithPath:path];
  NSError *error = nil;
  if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor
                                                                      error:&error]) {
    print_error(@"Could not start the Metal GPU capture", error);
    throw std::runtime_error("Could not start the Metal GPU capture");
  }
  state_->capture_active = true;
  std::printf("Capturing GPU work to %s\n", path.UTF8String);
}

void GpuRaytraceResources::stop_capture() {
  if (state_ != nullptr && state_->capture_active) {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
    state_->capture_active = false;
  }
}

} // namespace panorama
