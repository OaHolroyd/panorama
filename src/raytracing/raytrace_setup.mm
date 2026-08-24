#include "raytrace_setup.h"

#include "host_frontier.h"
#include "loaded_tile.h"
#include "metal_tile.h"
#include "png_writer.h"
#include "raytrace_gpu.h"
#include "resident_tile_cache.h"
#include "terrain_catalogue.h"
#include "tile_preparer.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama {
namespace {

/// Validate trace settings which do not belong to the output projection.
void validate_configuration(const RaytraceConfig &config) {
  if (config.tile_cache_size_bytes == 0U) {
    throw std::invalid_argument("Tile-cache byte budget must be positive");
  }
  if (config.tile_dir.empty()) {
    throw std::invalid_argument("Terrain tile directory must not be empty");
  }
  if (!std::isfinite(config.observer.easting) || !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) || !std::isfinite(config.max_distance) ||
      config.max_distance <= 0.0F) {
    throw std::invalid_argument("Raytrace configuration must be finite");
  }
}

/// Return a checked byte count for one host-side terrain payload allocation.
[[nodiscard]] size_t checked_byte_count(size_t count, size_t size, const char *name) {
  if (count > std::numeric_limits<size_t>::max() / size) {
    throw std::overflow_error(std::string(name) + " payload is too large");
  }
  return count * size;
}

/// Build the scalar tracing ABI shared by compatible resident terrain tiles.
[[nodiscard]] RaytraceParameters make_raytrace_parameters(
    const LoadedTile &tile,
    const RaytraceConfig &config,
    const TerrainCatalogue &catalogue,
    uint32_t ray_count
) {
  const double curvature_lift =
      kCurvatureCoefficient * static_cast<double>(config.max_distance) * config.max_distance;
  if (tile.delta > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation > static_cast<double>(std::numeric_limits<float>::max()) ||
      curvature_lift > static_cast<double>(std::numeric_limits<float>::max()) ||
      tile.num_levels == 0U || tile.num_levels >= 32U ||
      tile.size != (1U << (tile.num_levels - 1U))) {
    throw std::overflow_error("Raytrace geometry has an invalid float32 mipmap layout");
  }
  return {
      static_cast<float>(tile.delta),
      static_cast<float>(config.observer.elevation),
      static_cast<float>(kCurvatureCoefficient),
      catalogue.maximum_elevation().value_or(std::numeric_limits<float>::infinity()),
      tile.num_levels,
      ray_count,
      config.max_distance,
  };
}

/// Validate the projection-independent ray buffer and its Metal indexing.
[[nodiscard]] uint32_t validate_ray_field(const RayField &field) {
  const uint64_t ray_count = static_cast<uint64_t>(field.image.width) * field.image.height;
  if (ray_count == 0U || ray_count > std::numeric_limits<uint32_t>::max() ||
      field.rays.size() != ray_count) {
    throw std::invalid_argument("Ray field has invalid dimensions or storage");
  }
  for (const RayDirection &ray : field.rays) {
    const float horizontal_length = std::hypot(ray.x, ray.y);
    if (!std::isfinite(ray.x) || !std::isfinite(ray.y) || !std::isfinite(ray.slope) ||
        !std::isfinite(horizontal_length) || std::abs(horizontal_length - 1.0F) > 1e-4F ||
        (ray.x != 0.0F && !std::isfinite(ray.inverse_x)) ||
        (ray.y != 0.0F && !std::isfinite(ray.inverse_y))) {
      throw std::invalid_argument("Ray field contains an invalid projected direction");
    }
  }
  return static_cast<uint32_t>(ray_count);
}

/// Return a typed, non-owning view of a completed shared Metal buffer.
template <typename Value>
[[nodiscard]] std::span<const Value>
view_buffer(id<MTLBuffer> buffer, size_t count, const char *name) {
  const auto *contents = static_cast<const Value *>(buffer.contents);
  if (contents == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name + " Metal buffer");
  }
  return {contents, count};
}

/// Validate output choices before starting an otherwise expensive trace.
void validate_output_configuration(
    const RaytraceConfig &config,
    const RaytraceOutputConfig &outputs
) {
  if (!outputs.write_diagnostics && !outputs.write_synthetic) {
    throw std::invalid_argument("At least one raytrace output must be enabled");
  }
  if (outputs.write_synthetic && !config.compute_normals) {
    throw std::invalid_argument("Synthetic output requires collision-normal computation");
  }
}

/// Convert and encode one image while aggregating the two costs separately.
template <typename PixelGenerator>
void make_and_write_png(
    Timer &timer,
    const std::filesystem::path &path,
    ImageSize image,
    PixelGenerator &&generate_pixels
) {
  timer.start_wall("Pixel conversion");
  const std::vector<Rgb> pixels = generate_pixels();
  timer.stop("Pixel conversion");

  timer.start_wall("PNG encoding");
  write_rgb_png(path, pixels, image.width, image.height);
  timer.stop("PNG encoding");
}

/// Map completed shared buffers and write the independently selected PNGs.
///
/// This stage has its own top-level timer so terrain-tracing measurements stay
/// comparable when callers select different output products.
void write_trace_outputs(
    const RaytraceConfig &config,
    const RaytraceOutputConfig &outputs,
    const RayField &field,
    const GpuRaytraceResources &gpu,
    uint32_t ray_count
) {
  Timer timer("PNG generation");
  const std::span<const float> distances =
      view_buffer<float>(gpu.distances(), ray_count, "distance output");

  std::span<const uint32_t> surface_gradients;
  if (config.compute_normals) {
    surface_gradients =
        view_buffer<uint32_t>(gpu.surface_gradients(), ray_count, "surface gradient output");
  }

  if (outputs.write_diagnostics) {
    make_and_write_png(
        timer,
        "distances.png",
        field.image,
        [&] {
          return make_colormapped_pixels(
              distances, field.image.width, field.image.height, colormaps::viridis
          );
        }
    );
    if (config.compute_elevations) {
      make_and_write_png(
          timer,
          "elevations.png",
          field.image,
          [&] {
            return make_colormapped_pixels(
                view_buffer<float>(gpu.elevations(), ray_count, "elevation output"),
                field.image.width,
                field.image.height,
                colormaps::viridis
            );
          }
      );
    }
    if (config.compute_normals) {
      make_and_write_png(
          timer,
          "normals.png",
          field.image,
          [&] {
            return make_surface_normal_pixels(
                surface_gradients, distances, field.image.width, field.image.height
            );
          }
      );
    }
  }

  if (outputs.write_synthetic) {
    make_and_write_png(
        timer,
        "synthetic.png",
        field.image,
        [&] {
          return render_synthetic_terrain(
              surface_gradients, distances, field.image, outputs.synthetic_options
          );
        }
    );
  }
  timer.print();
}

} // namespace

/// Trace a fixed observer's per-pixel ray field through a set of terrain tiles
/// using a GPU-owned frontier and a host-managed resident-tile cache.
///
/// Each work item represents one unresolved ray segment in one resident tile.
/// A frontier pass either resolves that ray or emits its exact continuation.
/// The GPU walks across catalogue tiles rejected by their manifest maxima and
/// resolves the next required source; the host groups those continuations for
/// asynchronous loading and later activation.
///
/// Background workers load, validate, and mipmap GeoTIFF sources or open custom
/// tile handles. The main thread installs prepared sources into fixed-stride
/// terrain and maximum-mipmap atlases between GPU command buffers. When the
/// atlas is full, it evicts the least-recently-used slot not referenced by the
/// imminent frontier.
/// The loop ends when every ray has intersected terrain, reached the range
/// limit, risen above the catalogue, or left available terrain coverage.
void raytrace_tiled_heightmap(
    const RaytraceConfig &config,
    const RayField &field,
    const RaytraceOutputConfig &outputs
) {
  validate_configuration(config);
  validate_output_configuration(config, outputs);
  const uint32_t ray_count = validate_ray_field(field);
  // Start a composite timer.
  Timer timer("Total elapsed");
  timer.start_wall("Initial setup");

  // Discover the finite source set once. The catalogue provides the observer
  // source at index zero and an immutable key-to-source lookup thereafter.
  const TerrainCatalogue catalogue = TerrainCatalogue::discover(
      config.tile_dir,
      config.observer,
      config.max_distance,
      config.max_tile_count
  );
  const TileGrid &grid = catalogue.grid();
  const TileKey origin_key = catalogue.origin().key;

  // The observer tile establishes common data dimensions and the projected
  // coordinate system required by every fixed-stride atlas slot.
  timer.start_work("Tile load");
  LoadedTile origin = LoadedTile::load(catalogue.origin().path);
  timer.stop("Tile load");

  // GeoTIFF preparation builds its mipmap on the CPU. Custom files store no
  // mipmap hierarchy, so the cache generates it in the atlas on the GPU.
  const bool custom_origin = is_metal_tile_path(catalogue.origin().path);
  if (!custom_origin) {
    timer.start_work("Mipmap generation");
    origin.compute_mipmap();
    timer.stop("Mipmap generation");
  }

  validate_terrain_tile_position(origin, origin_key, grid);

  bool trace_quantized = false;
  QuantizedMetalTileRecordLayout quantized_layout = {};
  if (config.retain_quantized) {
    if (!custom_origin) {
      throw std::invalid_argument("--retain-quantized requires uint16 custom terrain tiles");
    }
    const MetalTileHeader header = read_metal_tile_header(catalogue.origin().path);
    if (header.sample_type != MetalTileSampleType::Uint16Decimeters) {
      throw std::invalid_argument("--retain-quantized requires uint16 custom terrain tiles");
    }
    quantized_layout = quantized_metal_tile_record_layout(header);
    trace_quantized = true;
  }

  const std::vector<TerrainSource> &paths = catalogue.sources();
  if (trace_quantized && std::any_of(paths.begin(), paths.end(), [](const TerrainSource &source) {
        return !is_metal_tile_path(source.path);
      })) {
    throw std::invalid_argument("--retain-quantized requires a custom-only terrain directory");
  }

  // Every rechunked tile must have the origin tile's dimensions, allowing
  // constant per-slot atlas strides. Derive the number of permitted slots
  // from the byte budget rather than hard-coding a tile count.
  const size_t mip_count = static_cast<size_t>(metal_tile_mipmap_value_count(origin.size));
  const size_t vertex_side = static_cast<size_t>(origin.size) + 1U;
  const size_t vertex_count = vertex_side * vertex_side;
  if (mip_count > std::numeric_limits<uint32_t>::max() ||
      vertex_count > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("Terrain tile arrays exceed Metal uint indexing");
  }
  const size_t tile_bytes =
      trace_quantized ? checked_byte_count(mip_count, sizeof(uint16_t), "terrain mipmap") +
                            quantized_layout.stride
                      : checked_byte_count(mip_count + vertex_count, sizeof(float), "terrain tile");
  const uint64_t slot_capacity = config.tile_cache_size_bytes / tile_bytes;
  if (slot_capacity == 0U) {
    throw std::runtime_error("Tile-cache byte budget cannot hold one terrain tile");
  }
  const uint32_t tile_count = static_cast<uint32_t>(paths.size());
  const uint64_t bounded_slot_count = std::min(slot_capacity, static_cast<uint64_t>(tile_count));
  if (bounded_slot_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::overflow_error("Tile-cache slot count exceeds Metal uint range");
  }
  const uint32_t atlas_slot_count = static_cast<uint32_t>(bounded_slot_count);

  // A ray has at most one unresolved segment globally, so every active and
  // deferred frontier is bounded by the number of output pixels.
  const RaytraceParameters shared_parameters =
      make_raytrace_parameters(origin, config, catalogue, ray_count);

  @autoreleasepool {
    // GPU resources own the device, reusable command/pipeline state, static
    // ray inputs, output images, and reusable frontier storage.
    const GpuTraceOutputRequirements gpu_outputs = {
        config.compute_elevations,
        config.compute_normals,
    };
    GpuRaytraceResources gpu(field.rays, paths, trace_quantized, gpu_outputs);
    gpu.initialise_frontier();

    // The cache shares the GPU resource owner's device. Preparation workers
    // use that device only to open custom-tile file handles in parallel; the
    // cache remains the sole owner of atlas buffers and GPU commands.
    ResidentTileCache
        cache(gpu.device(), paths, origin, origin_key, config, atlas_slot_count, timer);
    AsyncTilePreparer preparer(
        gpu.device(),
        paths,
        origin,
        grid,
        atlas_slot_count,
        config.max_tile_preparation_workers,
        timer
    );
    const ResidentTileCacheBindings cache_bindings = cache.bindings();

    // The observer tile is resident, so start GPU tracing while background
    // workers continue to prepare later source tiles.
    timer.stop("Initial setup");

    // Each frontier pass completes synchronously because the CPU needs its
    // emitted counts and deferred entries before it can schedule the next pass.
    gpu.start_capture_if_requested();
    uint32_t active_count = ray_count;

    // Host-frontier state owns deferred continuations and source lookup.
    HostFrontier frontier(catalogue, field.rays, shared_parameters, cache.slot_capacity());
    const std::vector<uint8_t> no_pinned_slots(cache.slot_capacity(), 0U);

    // Count GPU-resolved continuations independently of host load requests;
    // many rays can share one source request.
    uint64_t deferred_successor_work = 0U;
    uint64_t gpu_locally_skipped_tiles = 0U;
    uint64_t gpu_globally_skipped_tiles = 0U;

    // Wall time includes the CPU's per-pass buffer setup, command encoding,
    // submission, and waits. The device timestamps recorded below separately
    // report the GPU's execution-only work within this same region.
    timer.start_wall("GPU raytrace");
    try {
      // Loading starts only after every permanent buffer and the command queue
      // exists. Any later failure is caught below, which stops and joins these
      // threads before unwinding their owning vector.
      preparer.start();
      // The first observer-tile pass reveals which rays leave that tile.

      // Keep going until all rays have completed
      while (active_count != 0U) {
        // Account separately for CPU frontier bookkeeping before the Metal
        // command is created. This includes LRU use stamps and counter reset.
        timer.start_wall("Frontier bookkeeping");

        // Each ray can have only one unresolved tile segment. Validate that
        // the preceding pass preserved that contract.
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_frontier(gpu.active_frontier(), active_count, "active frontier");
#endif

        // This pass is about to read every referenced slot. Updating the LRU
        // stamp now makes recently traced terrain the last cache victim. The
        // unique slot set was recorded while the frontier was constructed.
        frontier.record_active_slot_use(cache);

        // The GPU resource resets its append counters, encodes the ordered
        // trace/emit kernels, waits for their shared results, and returns the
        // host-visible successor counts.
        timer.stop("Frontier bookkeeping");
        const GpuFrontierPassResult pass = gpu.trace_frontier(
            cache_bindings,
            shared_parameters,
            static_cast<uint32_t>(mip_count),
            active_count,
            timer
        );
        timer.add_work("GPU raytrace", pass.device_milliseconds);
        gpu_locally_skipped_tiles += pass.locally_skipped_tiles;
        gpu_globally_skipped_tiles += pass.globally_skipped_tiles;

        // One active ray emits at most one successor. The kernel guards its
        // writes; this check turns a counter overflow into a useful failure.
        if (pass.deferred_count > ray_count) {
          throw std::runtime_error("GPU frontier exceeds the ray frontier capacity");
        }

        deferred_successor_work += static_cast<uint64_t>(pass.deferred_count);

        // Keep the mapped GPU output alive through atlas publication, then
        // consume it directly while constructing the next active frontier.
        const std::span<const DeferredRayWork> deferred = gpu.deferred_work(pass.deferred_count);
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_deferred_work(deferred);
#endif

        timer.start_wall("Frontier bookkeeping");
        // The completed pass no longer references atlas slots, so any slot may
        // be selected while publishing newly prepared sources.
        frontier.mark_installed(cache.install_prepared(preparer, no_pinned_slots, timer));

        // Activate continuations whose terrain became resident during the
        // synchronous installation above.
        active_count =
            frontier.activate_resident(gpu.active_frontier(), 0U, cache, preparer, deferred);
#if defined(PANORAMA_DEBUG_VALIDATION)
        frontier.validate_deferred_work();
        frontier.validate_frontier(gpu.active_frontier(), active_count, "activated frontier");
#endif
        timer.stop("Frontier bookkeeping");

        // If no resident continuation is ready, wait for a requested tile to
        // complete preparation. There is no GPU work to submit in this case,
        // so an empty pin set makes every currently resident slot evictable.
        while (active_count == 0U && frontier.has_deferred_work()) {
          timer.start_wall("Tile availability wait");
          preparer.wait_for_prepared();
          timer.stop("Tile availability wait");

          timer.start_wall("Frontier bookkeeping");
          frontier.mark_installed(cache.install_prepared(preparer, no_pinned_slots, timer));
          active_count =
              frontier.activate_resident(gpu.active_frontier(), active_count, cache, preparer);
#if defined(PANORAMA_DEBUG_VALIDATION)
          frontier.validate_deferred_work();
          frontier.validate_frontier(gpu.active_frontier(), active_count, "activated frontier");
#endif
          timer.stop("Frontier bookkeeping");
        }
      }
    } catch (...) {
      preparer.stop_and_join();
      gpu.stop_capture();
      throw;
    }
    preparer.stop_and_join();
    gpu.stop_capture();
    timer.stop("GPU raytrace");

    // Report the finite source catalogue separately from the bounded resident
    // cache, so a memory-budget change is visible when rechunk levels vary.
    const TilePreparationStatistics preparation_statistics = preparer.statistics();
    const ResidentTileCacheStatistics cache_statistics = cache.statistics();
    std::printf(
        "Terrain sources: %u (resident slots %u / cache capacity %u, preparation workers %u).\n",
        tile_count,
        cache_statistics.resident_tiles,
        cache_statistics.slot_capacity,
        preparation_statistics.worker_count
    );
    std::printf(
        "  GPU frontier continuations: %llu rays deferred to source buckets.\n",
        static_cast<unsigned long long>(deferred_successor_work)
    );
    const uint64_t locally_skipped_tiles = gpu_locally_skipped_tiles;
    const uint64_t globally_skipped_tiles = gpu_globally_skipped_tiles;
    const uint64_t skipped_tiles = locally_skipped_tiles + globally_skipped_tiles;
    std::printf(
        "  Tile I/O: %llu requests (%llu unique, %llu duplicate); %llu skips "
        "(%llu local, %llu global; %s).\n",
        static_cast<unsigned long long>(preparation_statistics.requests),
        static_cast<unsigned long long>(preparation_statistics.unique_requests),
        static_cast<unsigned long long>(preparation_statistics.duplicate_requests),
        static_cast<unsigned long long>(skipped_tiles),
        static_cast<unsigned long long>(locally_skipped_tiles),
        static_cast<unsigned long long>(globally_skipped_tiles),
        catalogue.maximum_elevation().has_value() ? "GPU cutoff enabled" : "no complete maxima"
    );
    std::printf(
        "  Atlas installations: %llu, copied: %.3f GiB, Metal I/O: %.3f GiB, "
        "evictions: %llu.\n",
        static_cast<unsigned long long>(cache_statistics.installations),
        static_cast<double>(cache_statistics.bytes_copied) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(cache_statistics.bytes_loaded_with_metal_io) /
            (1024.0 * 1024.0 * 1024.0),
        static_cast<unsigned long long>(cache_statistics.evictions)
    );
    timer.print();

    // Encoding remains outside `Total elapsed`; it is an output backend rather
    // than part of terrain traversal and may be omitted by future consumers.
    write_trace_outputs(config, outputs, field, gpu, ray_count);
  }
}

} // namespace panorama
