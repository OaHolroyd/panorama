#include "terrain_shadow_gpu.h"

#import <Foundation/Foundation.h>

#include "timer.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

/// Use the same bit layout as primary traversal specialization selection.
[[nodiscard]] constexpr uint32_t trace_variant_index(bool bilinear, bool c1_normals) {
  return (bilinear ? 1U : 0U) | (c1_normals ? 2U : 0U);
}

[[nodiscard]] id<MTLBuffer> make_buffer(id<MTLDevice> device, NSUInteger length, const char *name) {
  id<MTLBuffer> result = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
  if (result == nil) {
    throw std::runtime_error(std::string("Could not allocate ") + name + " Metal buffer");
  }
  return result;
}

void check_command(id<MTLCommandBuffer> command, const char *name) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    throw std::runtime_error(std::string(name) + " Metal command failed");
  }
}

} // namespace

static_assert(sizeof(ShadowTraceParameters) == 16U * sizeof(uint32_t));

struct GpuTerrainShadowResources::State {
  // Pipelines reuse the primary trace's device, command queue, and library.
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  id<MTLComputePipelineState> initialise_pipeline;
  // C1 normals are not consumed by any-hit shadows, but caching the matching
  // four specializations keeps primary and shadow option changes atomic.
  std::array<id<MTLComputePipelineState>, 4U> trace_pipelines = {};
  id<MTLComputePipelineState> trace_pipeline;
  id<MTLComputePipelineState> emit_pipeline;

  // Ray-sized buffers follow primary output indices. `active` and `deferred`
  // use the same ABI as the primary HostFrontier.
  id<MTLBuffer> rays;
  id<MTLBuffer> visibility;
  id<MTLBuffer> active;
  id<MTLBuffer> continuations;
  id<MTLBuffer> deferred;
  id<MTLBuffer> deferred_count;
  uint32_t capacity = 0U;
  bool trace_quantized;

  [[nodiscard]] id<MTLComputePipelineState>
  make_trace_pipeline(bool bilinear, bool c1Normals) const {
    NSError *error = nil;
    NSString *trace_name =
        trace_quantized ? @"trace_shadow_tile_frontier_quantized" : @"trace_shadow_tile_frontier";
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bilinear type:MTLDataTypeBool atIndex:3];
    [constants setConstantValue:&c1Normals type:MTLDataTypeBool atIndex:4];
    id<MTLFunction> function = [library newFunctionWithName:trace_name
                                             constantValues:constants
                                                      error:&error];
    if (function == nil) {
      throw std::runtime_error("Could not specialize shadow trace kernel");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    if (pipeline == nil) {
      throw std::runtime_error("Could not create shadow trace pipeline");
    }
    return pipeline;
  }

  void select_trace_pipeline(bool bilinear, bool c1Normals) {
    trace_pipeline = trace_pipelines[trace_variant_index(bilinear, c1Normals)];
    if (trace_pipeline == nil) {
      throw std::logic_error("Requested shadow trace pipeline specialization is unavailable");
    }
  }
};

GpuTerrainShadowResources::GpuTerrainShadowResources(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLLibrary> library,
    bool trace_quantized,
    bool bilinear_collisions,
    bool c1_normals
) {
  if (device == nil || queue == nil || library == nil) {
    throw std::invalid_argument("Shadow resources require an existing Metal trace context");
  }
  auto state = std::make_unique<State>();
  state->device = device;
  state->queue = queue;
  state->library = library;
  state->trace_quantized = trace_quantized;
  NSError *error = nil;
  id<MTLFunction> initialise = [library newFunctionWithName:@"initialise_shadow_rays"];
  id<MTLFunction> emit = [library newFunctionWithName:@"emit_shadow_tile_frontier"];
  if (initialise == nil || emit == nil) {
    throw std::runtime_error("Shadow Metal kernels are missing");
  }
  state->initialise_pipeline = [device newComputePipelineStateWithFunction:initialise error:&error];
  state->emit_pipeline = [device newComputePipelineStateWithFunction:emit error:&error];
  if (state->initialise_pipeline == nil || state->emit_pipeline == nil) {
    throw std::runtime_error("Could not create shadow Metal pipelines");
  }
  for (uint32_t index = 0U; index < state->trace_pipelines.size(); index++) {
    state->trace_pipelines[index] =
        state->make_trace_pipeline((index & 1U) != 0U, (index & 2U) != 0U);
  }
  state->select_trace_pipeline(bilinear_collisions, c1_normals);
  state->deferred_count = make_buffer(device, sizeof(uint32_t), "shadow deferred count");
  state_ = std::move(state);
}

GpuTerrainShadowResources::~GpuTerrainShadowResources() = default;

void GpuTerrainShadowResources::resize(uint32_t ray_count) {
  State &state = *state_;
  if (ray_count == 0U) {
    throw std::invalid_argument("Shadow ray count must be positive");
  }
  if (state.capacity == ray_count) {
    return;
  }
  state.rays = make_buffer(state.device, sizeof(float) * 4U * ray_count, "shadow rays");
  state.visibility = make_buffer(state.device, sizeof(uint8_t) * ray_count, "shadow visibility");
  state.active = make_buffer(state.device, sizeof(RayWorkItem) * ray_count, "shadow frontier");
  state.continuations =
      make_buffer(state.device, sizeof(float) * ray_count, "shadow continuations");
  state.deferred =
      make_buffer(state.device, sizeof(DeferredRayWork) * ray_count, "shadow deferred work");
  state.capacity = ray_count;
}

void GpuTerrainShadowResources::set_collision_options(bool bilinear_collisions, bool c1_normals) {
  // The render thread synchronously completes every shadow pass before this
  // method can run, so choosing a cached pipeline is safe without a fence.
  state_->select_trace_pipeline(bilinear_collisions, c1_normals);
}

std::span<const DeferredRayWork> GpuTerrainShadowResources::initialise(
    id<MTLBuffer> camera_rays,
    id<MTLBuffer> distances,
    id<MTLBuffer> elevations,
    id<MTLBuffer> surface_gradients,
    id<MTLBuffer> catalogue_hash,
    const ShadowTraceParameters &parameters,
    Timer &timer
) {
  State &state = *state_;
  resize(parameters.trace.ray_count);
  auto *count = static_cast<uint32_t *>(state.deferred_count.contents);
  if (count == nullptr) {
    throw std::runtime_error("Could not map shadow deferred count");
  }
  *count = 0U;
  timer.start_wall("GPU shadow encoding");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create shadow initialisation command");
  }
  // Reconstruct eligible collision origins and emit their starting catalogue
  // sources. Ineligible pixels retain a visible byte and require no frontier.
  [encoder setComputePipelineState:state.initialise_pipeline];
  [encoder setBuffer:camera_rays offset:0 atIndex:0];
  [encoder setBuffer:distances offset:0 atIndex:1];
  [encoder setBuffer:elevations offset:0 atIndex:2];
  [encoder setBuffer:surface_gradients offset:0 atIndex:3];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
  [encoder setBuffer:state.rays offset:0 atIndex:5];
  [encoder setBuffer:state.visibility offset:0 atIndex:6];
  [encoder setBuffer:state.deferred offset:0 atIndex:7];
  [encoder setBuffer:state.deferred_count offset:0 atIndex:8];
  [encoder setBuffer:catalogue_hash offset:0 atIndex:9];
  [encoder dispatchThreads:MTLSizeMake(state.capacity, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  timer.stop("GPU shadow encoding");
  timer.start_wall("GPU shadow wait");
  check_command(command, "Shadow initialisation");
  timer.stop("GPU shadow wait");
  return deferred_work(*count);
}

GpuFrontierPassResult GpuTerrainShadowResources::trace_frontier(
    const TileManagerBindings &cache,
    id<MTLBuffer> catalogue_hash,
    const ShadowTraceParameters &parameters,
    uint32_t mipmap_value_count,
    uint32_t active_count,
    Timer &timer
) {
  State &state = *state_;
  auto *count = static_cast<uint32_t *>(state.deferred_count.contents);
  if (count == nullptr || active_count > state.capacity) {
    throw std::runtime_error("Invalid shadow frontier");
  }
  *count = 0U;
  timer.start_wall("GPU shadow encoding");
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) {
    throw std::runtime_error("Could not create shadow frontier command");
  }
  // Any-hit traversal shares TileManager's resident buffers with the primary
  // trace, but clears visibility instead of writing collision products.
  [encoder setComputePipelineState:state.trace_pipeline];
  [encoder setBuffer:cache.mipmap_atlas offset:0 atIndex:0];
  [encoder setBuffer:cache.vertex_atlas offset:0 atIndex:1];
  [encoder setBuffer:state.active offset:0 atIndex:2];
  [encoder setBuffer:cache.metadata offset:0 atIndex:3];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
  [encoder setBytes:&mipmap_value_count length:sizeof(mipmap_value_count) atIndex:5];
  [encoder setBuffer:state.rays offset:0 atIndex:6];
  [encoder setBuffer:state.visibility offset:0 atIndex:7];
  [encoder setBuffer:state.continuations offset:0 atIndex:8];
  if (state.trace_quantized) {
    [encoder setBytes:&cache.quantized_layout length:sizeof(cache.quantized_layout) atIndex:9];
  }
  [encoder dispatchThreads:MTLSizeMake(active_count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];

  encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Could not create shadow continuation command");
  }
  // Continue clear sun rays from their individual collision origins. As in
  // the primary path, the host receives source indices rather than slots.
  [encoder setComputePipelineState:state.emit_pipeline];
  [encoder setBuffer:state.active offset:0 atIndex:0];
  [encoder setBuffer:cache.metadata offset:0 atIndex:1];
  [encoder setBuffer:state.continuations offset:0 atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder setBytes:&state.capacity length:sizeof(state.capacity) atIndex:4];
  [encoder setBuffer:state.deferred offset:0 atIndex:5];
  [encoder setBuffer:state.deferred_count offset:0 atIndex:6];
  [encoder setBuffer:catalogue_hash offset:0 atIndex:7];
  [encoder setBuffer:state.rays offset:0 atIndex:8];
  [encoder dispatchThreads:MTLSizeMake(active_count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [encoder endEncoding];
  timer.stop("GPU shadow encoding");
  timer.start_wall("GPU shadow wait");
  check_command(command, "Shadow frontier");
  timer.stop("GPU shadow wait");
  const double milliseconds = 1'000.0 * (command.GPUEndTime - command.GPUStartTime);
  return {*count, 0U, 0U, milliseconds};
}

std::span<const DeferredRayWork> GpuTerrainShadowResources::deferred_work(uint32_t count) const {
  if (count > state_->capacity || state_->deferred.contents == nullptr) {
    throw std::runtime_error("Invalid shadow deferred frontier");
  }
  return {static_cast<const DeferredRayWork *>(state_->deferred.contents), count};
}

id<MTLBuffer> GpuTerrainShadowResources::active_frontier() const { return state_->active; }

id<MTLBuffer> GpuTerrainShadowResources::visibility() const { return state_->visibility; }

void GpuTerrainShadowResources::fill_visibility(uint8_t value) {
  if (state_->visibility.contents == nullptr) {
    throw std::runtime_error("Could not map shadow visibility");
  }
  std::memset(state_->visibility.contents, value, state_->capacity);
}

} // namespace panorama
