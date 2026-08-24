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

static_assert(sizeof(RayDirection) == 5U * sizeof(float));
static_assert(sizeof(RaytraceParameters) == 7U * sizeof(uint32_t));
static_assert(sizeof(RayWorkItem) == 4U * sizeof(uint32_t));
static_assert(sizeof(DeferredRayWork) == 3U * sizeof(uint32_t));

struct CatalogueTileHashEntry {
  int64_t row;
  int64_t column;
  float maximum_elevation;
  uint32_t source_index;
};

static_assert(sizeof(CatalogueTileHashEntry) == 3U * sizeof(uint64_t));

[[nodiscard]] uint64_t mix_tile_hash(uint64_t value) {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return value;
}

[[nodiscard]] uint32_t catalogue_hash_capacity(size_t source_count) {
  uint64_t capacity = 1U;
  while (capacity < 2U * source_count) {
    capacity <<= 1U;
  }
  if (capacity > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("Terrain catalogue hash exceeds Metal uint range");
  }
  return static_cast<uint32_t>(capacity);
}

[[nodiscard]] uint32_t tile_hash(TileKey key, uint32_t mask) {
  const uint64_t row = mix_tile_hash(static_cast<uint64_t>(key.row));
  const uint64_t column = mix_tile_hash(static_cast<uint64_t>(key.column));
  return static_cast<uint32_t>(
      mix_tile_hash(row ^ (column + 0x9e3779b97f4a7c15ULL)) & mask
  );
}

/// Print a Foundation error in the command-line form used by host tools.
void print_error(NSString *context, NSError *error) {
  const char *detail = error == nil ? "unknown error" : error.localizedDescription.UTF8String;
  std::fprintf(stderr, "%s: %s\n", context.UTF8String, detail);
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
  id<MTLLibrary> library;
  id<MTLComputePipelineState> trace_pipeline;
  id<MTLComputePipelineState> emit_pipeline;
  id<MTLBuffer> rays;
  id<MTLBuffer> distance_output;
  id<MTLBuffer> elevation_output;
  id<MTLBuffer> surface_gradient_output;
  id<MTLBuffer> active;
  id<MTLBuffer> continuations;
  id<MTLBuffer> deferred_items;
  id<MTLBuffer> deferred_count;
  id<MTLBuffer> catalogue_hash;
  id<MTLBuffer> local_skip_count;
  id<MTLBuffer> global_skip_count;
  uint32_t frontier_capacity;
  uint32_t catalogue_hash_capacity;
  bool trace_quantized;
  GpuTraceOutputRequirements outputs;
  bool capture_active = false;
};

GpuRaytraceResources::GpuRaytraceResources(
    std::span<const RayDirection> rays,
    std::span<const TerrainSource> sources,
    bool trace_quantized,
    GpuTraceOutputRequirements outputs
) {
  if (rays.empty() || rays.size() > std::numeric_limits<uint32_t>::max() || sources.empty() ||
      sources.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("GPU raytrace resources require a valid nonempty ray field");
  }
  auto state = std::make_unique<State>();
  state->frontier_capacity = static_cast<uint32_t>(rays.size());
  state->catalogue_hash_capacity = catalogue_hash_capacity(sources.size());
  state->trace_quantized = trace_quantized;
  state->outputs = outputs;
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
  state->library = [state->device newLibraryWithURL:url error:&error];
  if (state->library == nil) {
    print_error(@"Could not load the Metal library", error);
    throw std::runtime_error("Could not load Metal library");
  }
  NSString *trace_name =
      trace_quantized ? @"trace_tile_frontier_quantized" : @"trace_tile_frontier";
  MTLFunctionConstantValues *trace_constants = [[MTLFunctionConstantValues alloc] init];
  [trace_constants setConstantValue:&outputs.surface_gradients type:MTLDataTypeBool atIndex:0];
  [trace_constants setConstantValue:&outputs.elevations type:MTLDataTypeBool atIndex:1];
  id<MTLFunction> trace =
      [state->library newFunctionWithName:trace_name
                           constantValues:trace_constants
                                    error:&error];
  id<MTLFunction> emit = [state->library newFunctionWithName:@"emit_tile_frontier"];
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

  // Static per-pixel directions and scientific output buffers persist for
  // every frontier pass.
  state->rays = make_buffer(
      state->device,
      rays.data(),
      checked_buffer_length(rays.size(), sizeof(RayDirection), "ray directions"),
      "ray directions"
  );
  state->distance_output = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(rays.size(), sizeof(float), "distance output"),
      "distance output"
  );
  // Function-constant specialization removes disabled output work. One-word
  // placeholders preserve the common kernel ABI without full per-ray buffers.
  state->elevation_output = make_buffer(
      state->device,
      nullptr,
      outputs.elevations
          ? checked_buffer_length(rays.size(), sizeof(float), "elevation output")
          : sizeof(float),
      "elevation output"
  );
  state->surface_gradient_output = make_buffer(
      state->device,
      nullptr,
      outputs.surface_gradients
          ? checked_buffer_length(rays.size(), sizeof(uint32_t), "surface gradient output")
          : sizeof(uint32_t),
      "surface gradient output"
  );
  clear_buffer(state->distance_output, "distance output");
  if (outputs.elevations) {
    clear_buffer(state->elevation_output, "elevation output");
  }
  // Every positive distance is written beside its gradient, and consumers use
  // distance as the validity mask, so clearing this potentially large buffer
  // would add setup bandwidth without defining any observable output.

  // A command completes before the host replaces this frontier with the next
  // source-bucketed batch, so one shared work buffer is sufficient.
  state->active = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(rays.size(), sizeof(RayWorkItem), "active frontier"),
      "active frontier"
  );
  state->continuations = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(rays.size(), sizeof(float), "ray continuations"),
      "ray continuations"
  );
  state->deferred_items = make_buffer(
      state->device,
      nullptr,
      checked_buffer_length(rays.size(), sizeof(DeferredRayWork), "deferred frontier"),
      "deferred frontier"
  );
  state->deferred_count =
      make_buffer(state->device, nullptr, sizeof(uint32_t), "deferred frontier count");
  std::vector<CatalogueTileHashEntry> catalogue_entries(
      state->catalogue_hash_capacity,
      {0, 0, std::numeric_limits<float>::infinity(), std::numeric_limits<uint32_t>::max()}
  );
  const uint32_t catalogue_mask = state->catalogue_hash_capacity - 1U;
  for (uint32_t source_index = 0U; source_index < static_cast<uint32_t>(sources.size());
       source_index++) {
    const TerrainSource &source = sources[source_index];
    uint32_t index = tile_hash(source.key, catalogue_mask);
    while (catalogue_entries[index].source_index != std::numeric_limits<uint32_t>::max()) {
      index = (index + 1U) & catalogue_mask;
    }
    catalogue_entries[index] = {
        source.key.row,
        source.key.column,
        source.maximum_elevation.value_or(std::numeric_limits<float>::infinity()),
        source_index,
    };
  }
  state->catalogue_hash = make_buffer(
      state->device,
      catalogue_entries.data(),
      checked_buffer_length(
          catalogue_entries.size(), sizeof(CatalogueTileHashEntry), "catalogue hash"
      ),
      "catalogue hash"
  );
  state->local_skip_count =
      make_buffer(state->device, nullptr, sizeof(uint32_t), "local skip count");
  state->global_skip_count =
      make_buffer(state->device, nullptr, sizeof(uint32_t), "global skip count");
  state_ = std::move(state);
}

GpuRaytraceResources::~GpuRaytraceResources() { stop_capture(); }

void GpuRaytraceResources::initialise_frontier() {
  State &state = *state_;
  auto *items = static_cast<RayWorkItem *>(state.active.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map active frontier buffer");
  }
  for (uint32_t ray = 0U; ray < state.frontier_capacity; ray++) {
    items[ray] = {0U, ray, 1U, 0.0F};
  }
}

id<MTLDevice> GpuRaytraceResources::device() const { return state_->device; }

id<MTLCommandQueue> GpuRaytraceResources::command_queue() const { return state_->queue; }

id<MTLLibrary> GpuRaytraceResources::library() const { return state_->library; }

id<MTLBuffer> GpuRaytraceResources::active_frontier() const { return state_->active; }

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
  auto *deferred_total = static_cast<uint32_t *>(state.deferred_count.contents);
  auto *local_skips = static_cast<uint32_t *>(state.local_skip_count.contents);
  auto *global_skips = static_cast<uint32_t *>(state.global_skip_count.contents);
  if (deferred_total == nullptr || local_skips == nullptr || global_skips == nullptr ||
      state.continuations.contents == nullptr) {
    throw std::runtime_error("Could not map GPU frontier buffers");
  }
  *deferred_total = 0U;
  *local_skips = 0U;
  *global_skips = 0U;

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
  [encoder setBuffer:state.rays offset:0 atIndex:2];
  [encoder setBuffer:state.active offset:0 atIndex:3];
  [encoder setBuffer:cache.metadata offset:0 atIndex:4];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
  [encoder setBytes:&mipmap_value_count length:sizeof(mipmap_value_count) atIndex:6];
  [encoder setBuffer:state.distance_output offset:0 atIndex:7];
  [encoder setBuffer:state.elevation_output offset:0 atIndex:8];
  [encoder setBuffer:state.continuations offset:0 atIndex:9];
  if (state.trace_quantized) {
    [encoder setBytes:&cache.quantized_layout length:sizeof(cache.quantized_layout) atIndex:10];
  }
  [encoder setBuffer:state.surface_gradient_output offset:0 atIndex:11];
  [encoder dispatchThreads:MTLSizeMake(active_count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];

  encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Could not create successor frontier encoder");
  }
  encoder.label = @"emit_tile_frontier";
  [encoder setComputePipelineState:state.emit_pipeline];
  [encoder setBuffer:state.active offset:0 atIndex:0];
  [encoder setBuffer:cache.metadata offset:0 atIndex:1];
  [encoder setBuffer:state.rays offset:0 atIndex:2];
  [encoder setBuffer:state.continuations offset:0 atIndex:3];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
  [encoder setBytes:&state.frontier_capacity length:sizeof(state.frontier_capacity) atIndex:5];
  [encoder setBuffer:state.deferred_items offset:0 atIndex:6];
  [encoder setBuffer:state.deferred_count offset:0 atIndex:7];
  [encoder setBuffer:state.catalogue_hash offset:0 atIndex:8];
  [encoder setBytes:&state.catalogue_hash_capacity
              length:sizeof(state.catalogue_hash_capacity)
             atIndex:9];
  [encoder setBuffer:state.local_skip_count offset:0 atIndex:10];
  [encoder setBuffer:state.global_skip_count offset:0 atIndex:11];
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
  return {*deferred_total, *local_skips, *global_skips, device_milliseconds};
}

std::span<const DeferredRayWork> GpuRaytraceResources::deferred_work(uint32_t count) const {
  if (count > state_->frontier_capacity) {
    throw std::invalid_argument("Deferred GPU frontier exceeds its capacity");
  }
  const auto *items = static_cast<const DeferredRayWork *>(state_->deferred_items.contents);
  if (items == nullptr) {
    throw std::runtime_error("Could not map deferred frontier buffer");
  }
  return {items, count};
}

id<MTLBuffer> GpuRaytraceResources::distances() const { return state_->distance_output; }
id<MTLBuffer> GpuRaytraceResources::elevations() const {
  if (!state_->outputs.elevations) {
    throw std::logic_error("Elevations were not requested");
  }
  return state_->elevation_output;
}

id<MTLBuffer> GpuRaytraceResources::surface_gradients() const {
  if (!state_->outputs.surface_gradients) {
    throw std::logic_error("Surface gradients were not requested");
  }
  return state_->surface_gradient_output;
}

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
