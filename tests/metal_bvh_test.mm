#include "arguments.h"
#include "metal_tile.h"
#include "terrain_manifest.h"
#include "terrain_trace_session.h"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numbers>
#include <stdexcept>
#include <vector>

using namespace panorama;

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Match the viewer's default camera and output requirements, excluding image
// generation, lighting and display. Run each backend in a separate process.
void benchmark(int argc, const char *argv[]) {
  if (argc < 5 || argc > 10)
    throw std::invalid_argument(
        "usage: metal-bvh-test --benchmark TILE_DIR software|metal-bvh CACHE_MIB "
        "[RANGE [WIDTH HEIGHT [LOD_SCALE [DEBUG_OUTPUTS]]]]"
    );
  RaytraceConfig config{};
  config.tile_dir = argv[2];
  config.observer = {2623452.4, 1100502.2, 3415.0};
  config.raytracer = arguments::parse_raytracer(argv[3]);
  config.bvh_cache_size_bytes = std::stoull(argv[4]) * 1048576ULL;
  config.tile_cache_size_bytes = 128ULL * 1048576ULL;
  config.max_tile_preparation_workers = 8U;
  config.retain_quantized = true;
  config.max_distance = argc > 5 ? std::stof(argv[5]) : 600000.0F;
  config.lod_scale = argc > 8 ? std::stof(argv[8]) : 0.0F;
  const ImageSize image = {
      argc > 6 ? static_cast<uint32_t>(std::stoul(argv[6])) : 1600U,
      argc > 7 ? static_cast<uint32_t>(std::stoul(argv[7])) : 900U,
  };
  const bool debugging = argc <= 9 || std::stoi(argv[9]) != 0;
  const auto intrinsics =
      CameraIntrinsics::from_vertical_field_of_view(image, 70.0 * std::numbers::pi / 180.0);
  auto field = make_camera_ray_field(image, {{0, 0, 0}, intrinsics, NoDistortion{}});
  TerrainTraceSession session(config, field, {true, true, debugging});
  std::printf(
      "Device: %s; range %.0f m, LOD %.2f, debug %d\n",
      session.device().name.UTF8String,
      config.max_distance,
      config.lod_scale,
      debugging
  );
  for (uint32_t frame = 0; frame < 6; ++frame) {
    @autoreleasepool {
      if (frame == 3)
        field = make_camera_ray_field(
            image,
            {{std::numbers::pi / 180.0, 0, 0}, intrinsics, NoDistortion{}}
        );
      if (frame == 5) {
        config.observer.easting += 1.0;
        require(session.relocate_observer(config.observer), "Benchmark relocation failed");
      }
      std::printf(
          "%s: ",
          frame == 0   ? "cold"
          : frame == 3 ? "turn 1 degree"
          : frame == 5 ? "move 1 metre"
                       : "repeat"
      );
      session.trace(field);
      session.print_trace_statistics();
    }
  }
}

void write_fixture(const std::filesystem::path &directory, bool quantized, double spacing) {
  std::filesystem::create_directories(directory);
  constexpr uint32_t cells = 16U;
  constexpr uint32_t levels = 4U;
  std::vector<TerrainManifestEntry> manifest;
  for (int row = -1; row <= 1; ++row) {
    for (int column = -1; column <= 2; ++column) {
      // A missing column followed by terrain tests termination at coverage gaps.
      if (column == 1)
        continue;
      std::vector<MetalTileLod> lods;
      std::vector<std::byte> payload;
      uint64_t offset = kMetalTileLodHeaderSize + levels * sizeof(MetalTileLod);
      for (uint32_t lod = 1U; lod <= levels; ++lod) {
        const uint32_t side = (cells >> (lod - 1U)) + 1U;
        std::vector<float> heights;
        for (uint32_t y = 0; y < side; ++y) {
          for (uint32_t x = 0; x < side; ++x) {
            const double east = column * 16.0 + x * double(1U << (lod - 1U));
            const double north = -(row + 1) * 16.0 + y * double(1U << (lod - 1U));
            heights.push_back(
                float(
                    std::round(
                        10.0 * (1000.0 + 70.0 * std::sin(east * 0.17) * std::cos(north * 0.11))
                    ) *
                    0.1
                )
            );
          }
        }
        const uint64_t bytes = heights.size() * (quantized ? sizeof(uint16_t) : sizeof(float));
        const float maximum = *std::max_element(heights.begin(), heights.end());
        lods.push_back(
            {lod,
             side - 1U,
             uint32_t(std::countr_zero(side - 1U)) + 1U,
             quantized ? 9000 : 0,
             maximum,
             0U,
             offset,
             bytes}
        );
        const size_t start = payload.size();
        payload.resize(start + bytes);
        if (quantized) {
          for (size_t i = 0; i < heights.size(); ++i) {
            const uint16_t value = static_cast<uint16_t>(std::lround(heights[i] * 10.0F) - 9000);
            std::memcpy(payload.data() + start + i * sizeof(value), &value, sizeof(value));
          }
        } else
          std::memcpy(payload.data() + start, heights.data(), bytes);
        offset += bytes;
      }
      const auto &base = lods.front();
      MetalTileHeader header = {kMetalTileLodMagic,
                                kMetalTileLodVersion,
                                kMetalTileLodHeaderSize,
                                MetalTileCompression::None,
                                2056U,
                                cells,
                                5U,
                                base.maximum_elevation,
                                quantized ? MetalTileSampleType::Uint16Decimeters
                                          : MetalTileSampleType::Float32,
                                base.elevation_base_decimeters,
                                0U,
                                row,
                                column,
                                2600000.0 + double(column) * cells * spacing,
                                1200000.0 - double(row + 1) * cells * spacing,
                                spacing,
                                base.vertex_offset,
                                base.vertex_byte_count,
                                levels,
                                uint32_t(sizeof(MetalTileLod)),
                                kMetalTileLodHeaderSize,
                                levels * sizeof(MetalTileLod)};
      const auto path =
          directory / ("test_r" + std::to_string(row) + "_c" + std::to_string(column) + ".ptile");
      write_metal_tile_lods(path, header, lods, payload);
      manifest.push_back({row, column, 1071.0F, 929.0F});
    }
  }
  // Leave float fixtures without a sidecar to retain legacy/no-manifest coverage.
  if (quantized)
    write_terrain_manifest(terrain_manifest_path(directory), manifest);
}

void compare(
    TerrainTraceSession &software,
    TerrainTraceSession &hardware,
    const RayField &field,
    float distance_tolerance = 0.01F,
    const MetalTileHeader *real_grid = nullptr
) {
  software.trace(field);
  hardware.trace(field);
  const auto *expected = static_cast<const float *>(software.distances().contents);
  const auto *actual = static_cast<const float *>(hardware.distances().contents);
  const auto *expected_z = static_cast<const float *>(software.elevations().contents);
  const auto *actual_z = static_cast<const float *>(hardware.elevations().contents);
  const auto *expected_n = static_cast<const uint32_t *>(software.surface_gradients().contents);
  const auto *actual_n = static_cast<const uint32_t *>(hardware.surface_gradients().contents);
  size_t mismatches = 0, hits = 0;
  float maximum_error = 0.0F;
  for (size_t i = 0; i < field.rays.size(); ++i) {
    if (expected[i] > 0)
      ++hits;
    const float delta = std::abs(expected[i] - actual[i]);
    maximum_error = std::max(maximum_error, delta);
    const float elevation_tolerance =
        real_grid == nullptr
            ? 0.02F
            : 0.002F + distance_tolerance * (std::abs(field.rays[i].slope) +
                                             2.0F * float(kCurvatureCoefficient) * expected[i]);
    const bool mismatch =
        (expected[i] > 0) != (actual[i] > 0) || delta > distance_tolerance ||
        (expected[i] > 0 && std::abs(expected_z[i] - actual_z[i]) > elevation_tolerance);
    if (mismatch) {
      if (mismatches < 8)
        std::fprintf(
            stderr,
            "ray %zu: software %.7g BVH %.7g (dz %.7g), direction %.9g %.9g %.9g\n",
            i,
            expected[i],
            actual[i],
            expected_z[i] - actual_z[i],
            field.rays[i].x,
            field.rays[i].y,
            field.rays[i].slope
        );
      ++mismatches;
    }
    if (expected[i] > 0 && actual[i] > 0 && expected_n[i] != actual_n[i]) {
      // The legacy bilinear solver accepts/clamps roots just outside a cell.
      // Only at that documented edge tolerance can patch-local normals differ
      // discontinuously. Never exempt hit masks or distance/elevation errors.
      bool on_patch_edge = false;
      if (real_grid != nullptr) {
        const ObserverLocation observer = software.observer();
        const double edge_tolerance = std::max(0.05, 128.0 * FLT_EPSILON * expected[i]);
        const double x =
            observer.easting + expected[i] * double(field.rays[i].x) - real_grid->lower_left_x;
        const double y =
            observer.northing + expected[i] * double(field.rays[i].y) - real_grid->lower_left_y;
        const double actual_x =
            observer.easting + actual[i] * double(field.rays[i].x) - real_grid->lower_left_x;
        const double actual_y =
            observer.northing + actual[i] * double(field.rays[i].y) - real_grid->lower_left_y;
        on_patch_edge =
            std::abs(std::remainder(x, real_grid->cell_size)) <= edge_tolerance ||
            std::abs(std::remainder(y, real_grid->cell_size)) <= edge_tolerance ||
            std::abs(std::remainder(actual_x, real_grid->cell_size)) <= edge_tolerance ||
            std::abs(std::remainder(actual_y, real_grid->cell_size)) <= edge_tolerance;
      }
      // Half precision gradients may round by one ULP after a submillimetre
      // difference in root location. Compare decoded values, not packed bits.
      for (unsigned component = 0; component < 2; ++component) {
        const uint16_t a = uint16_t(expected_n[i] >> (component * 16U));
        const uint16_t b = uint16_t(actual_n[i] >> (component * 16U));
        const float ga = float(std::bit_cast<_Float16>(a));
        const float gb = float(std::bit_cast<_Float16>(b));
        const float half_tolerance =
            std::max(0.002F, std::max(std::abs(ga), std::abs(gb)) * 0.002F);
        if (!on_patch_edge && std::abs(ga - gb) > half_tolerance) {
          if (mismatches < 30)
            std::fprintf(
                stderr,
                "gradient %zu: %.7g vs %.7g, distances %.9g %.9g; xy %.9g %.9g\n",
                i,
                ga,
                gb,
                expected[i],
                actual[i],
                expected[i] * field.rays[i].x,
                expected[i] * field.rays[i].y
            );
          ++mismatches;
        }
      }
    }
  }
  std::printf(
      "Parity: %zu rays, %zu hits, max distance error %.7g m, %zu mismatches.\n",
      field.rays.size(),
      hits,
      maximum_error,
      mismatches
  );
  require(hits > 0, "Fixture must hit terrain");
  require(mismatches == 0, "BVH/software parity failed");
}

void exercise(const std::filesystem::path &directory, bool retain, double spacing, uint32_t block) {
  RaytraceConfig config{directory,
                        {2600000.0 + 4.5 * spacing, 1200000.0 - 5.5 * spacing, 1120.0},
                        float(48.0 * spacing),
                        0U,
                        16384U,
                        2U,
                        retain,
                        true,
                        false};
  auto field = make_angular_ray_field({129, 65}, {0.0, 2 * std::numbers::pi, -1.3, 0.1});
  // Include exact cardinal/diagonal directions and steep, unnormalised slopes.
  for (size_t i = 0; i < 8; ++i) {
    const float x = i < 4 ? (i == 0   ? 1.0F
                             : i == 1 ? -1.0F
                                      : 0.0F)
                          : (i & 1 ? -1.0F : 1.0F) * std::sqrt(0.5F);
    const float y = i < 4 ? (i == 2   ? 1.0F
                             : i == 3 ? -1.0F
                                      : 0.0F)
                          : (i & 2 ? -1.0F : 1.0F) * std::sqrt(0.5F);
    field.rays[i] = {x, y, x == 0 ? INFINITY : 1.0F / x, y == 0 ? INFINITY : 1.0F / y, -2.0F};
  }
  const GpuTraceOutputRequirements outputs{true, true, true};
  TerrainTraceSession software(config, field, outputs);
  config.raytracer = Raytracer::MetalBvh;
  config.bvh_block_cells = block;
  TerrainTraceSession hardware(config, field, outputs);
  std::printf("Fixture: retain=%d spacing=%g block=%u\n", retain, spacing, block);
  compare(software, hardware, field, spacing > 10 ? 0.25F : 0.01F);
  // Same session, different camera and dimensions: residency must be declared again.
  auto next = make_angular_ray_field({97, 33}, {0.35, 6.6, -1.25, 0.05});
  software.set_collision_options(false, true);
  hardware.set_collision_options(false, true);
  compare(software, hardware, next, spacing > 10 ? 0.25F : 0.01F);
  const uint64_t builds_before_move = hardware.bvh_statistics().builds;
  ObserverLocation moved = config.observer;
  moved.easting -= 8 * spacing;
  moved.northing += 3 * spacing;
  require(
      software.relocate_observer(moved) && hardware.relocate_observer(moved),
      "Relocation failed"
  );
  compare(software, hardware, next, spacing > 10 ? 0.25F : 0.01F);
  const auto builds_after_move = hardware.bvh_statistics().builds;
  // A move may request previously unseen tiles, but already cached tiles must
  // survive. Returning to the previous view requires no detailed rebuilds.
  require(hardware.relocate_observer(config.observer), "Return relocation failed");
  hardware.trace(next);
  require(
      hardware.bvh_statistics().builds == builds_after_move &&
          builds_after_move >= builds_before_move,
      "Affine relocation rebuilt cached terrain BVHs"
  );
  require(hardware.relocate_observer(moved), "Second relocation failed");
  software.set_collision_options(true, true);
  hardware.set_collision_options(true, true);
  software.set_lod_scale(3.0F);
  hardware.set_lod_scale(3.0F);
  compare(software, hardware, next, spacing > 10 ? 0.25F : 0.01F);
  hardware.set_raytracer(Raytracer::Software);
  compare(software, hardware, next, spacing > 10 ? 0.25F : 0.01F);
  hardware.set_raytracer(Raytracer::MetalBvh);
  compare(software, hardware, next, spacing > 10 ? 0.25F : 0.01F);
  software.trace_shadows(2.1, 0.35);
  hardware.trace_shadows(2.1, 0.35);
  require(
      std::memcmp(
          software.shadow_visibility().contents,
          hardware.shadow_visibility().contents,
          next.rays.size()
      ) == 0,
      "Shadow output changed"
  );
}

void check_limits_and_optional_outputs(const std::filesystem::path &directory) {
  auto field = make_angular_ray_field({33, 17}, {0.0, 6.3, -1.3, 0.1});
  RaytraceConfig
      config{directory, {2600045.0, 1199945.0, 1120.0}, 40.0F, 0U, 16384U, 2U, true, true, false};
  config.raytracer = Raytracer::MetalBvh;
  for (bool bilinear : {false, true}) {
    config.bilinear_collisions = bilinear;
    TerrainTraceSession trace(config, field, {false, false, false});
    trace.trace(field);
    const auto *values = static_cast<const float *>(trace.distances().contents);
    for (size_t i = 0; i < field.rays.size(); ++i) {
      require(
          std::isfinite(values[i]) && values[i] >= 0 && values[i] <= config.max_distance,
          "BVH escaped its finite distance interval"
      );
    }
    auto observer = config.observer;
    observer.elevation += 40.0;
    require(trace.relocate_observer(observer), "Height-only relocation failed");
    trace.trace(field);
  }
  config.bvh_block_cells = 0;
  bool rejected = false;
  try {
    TerrainTraceSession invalid(config, field, {false, false, false});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "Zero BVH block size must be rejected");
  rejected = false;
  try {
    (void)arguments::parse_raytracer("unknown");
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "Unknown backend must be rejected");
}

void check_streaming(const std::filesystem::path &directory) {
  const auto field = make_angular_ray_field({129, 65}, {0, 2 * std::numbers::pi, -1.3, 0.1});
  RaytraceConfig
      config{directory, {2600045, 1199945, 1120}, 600000, 0, 16384, 2, true, true, false};
  TerrainTraceSession software(config, field, {true, true, true});
  config.raytracer = Raytracer::MetalBvh;
  // Size selected from the device's real build workspace for the first tile.
  // The first pass cannot reserve room for every tile concurrently.
  auto *geometry = [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
  geometry.boundingBoxCount = 16;
  geometry.boundingBoxStride = 24;
  geometry.opaque = NO;
  geometry.allowDuplicateIntersectionFunctionInvocation = NO;
  auto *descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
  descriptor.geometryDescriptors = @[ geometry ];
  const auto sizes = [software.device() accelerationStructureSizesWithDescriptor:descriptor];
  config.bvh_cache_size_bytes = 17U * 17U * 2U + 16U * (20U + 24U) + 48U + 8U +
                                2U * sizes.accelerationStructureSize + sizes.buildScratchBufferSize;
  TerrainTraceSession hardware(config, field, {true, true, true});
  compare(software, hardware, field);
  compare(software, hardware, field);
  const auto stats = hardware.bvh_statistics();
  const auto catalogue =
      TerrainCatalogue::discover(directory, config.observer, config.max_distance, 0);
  require(stats.builds <= 2 * catalogue.sources().size(), "A tile was rebuilt within a frame");
  std::printf(
      "Streaming: %llu builds, %llu evictions, %llu peak / %llu budget bytes.\n",
      static_cast<unsigned long long>(stats.builds),
      static_cast<unsigned long long>(stats.evictions),
      static_cast<unsigned long long>(stats.peak_bytes),
      static_cast<unsigned long long>(stats.budget_bytes)
  );
  require(stats.evictions > 0, "Streaming fixture did not force eviction");
  require(
      stats.peak_bytes <= stats.budget_bytes && stats.resident_bytes <= stats.budget_bytes,
      "BVH cache exceeded its budget"
  );
}
} // namespace

int main(int argc, const char *argv[]) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    @autoreleasepool {
      require(arguments::parse_raytracer("metal-bvh") == Raytracer::MetalBvh, "Backend parser");
      if (argc >= 2 && std::string_view(argv[1]) == "--benchmark") {
        benchmark(argc, argv);
        return EXIT_SUCCESS;
      }
      const bool edge_cases = argc == 2 && std::string_view(argv[1]) == "--edge-cases";
      const bool streaming = argc == 2 && std::string_view(argv[1]) == "--streaming";
      if (argc >= 2 && !edge_cases && !streaming) {
        RaytraceConfig config{argv[1],
                              {2623452.4, 1100502.2, 3415.0},
                              21000.0F,
                              0U,
                              32U * 1024U * 1024U,
                              4U,
                              true,
                              true,
                              false};
        if (argc >= 3)
          config.max_distance = std::stof(argv[2]);
        if (argc >= 4)
          config.bvh_cache_size_bytes = std::stoull(argv[3]) * 1048576U;
        const auto field =
            make_angular_ray_field({512, 128}, {0, 2 * std::numbers::pi, -0.6, 0.15});
        TerrainTraceSession software(config, field, {true, true, true});
        config.raytracer = Raytracer::MetalBvh;
        TerrainTraceSession hardware(config, field, {true, true, true});
        const auto catalogue =
            TerrainCatalogue::discover(config.tile_dir, config.observer, config.max_distance, 0U);
        const auto header = read_metal_tile_header(catalogue.origin().path);
        compare(software, hardware, field, 0.15F, &header);
        compare(software, hardware, field, 0.15F, &header);
        require(
            hardware.bvh_statistics().peak_bytes <= config.bvh_cache_size_bytes,
            "Real BVH cache exceeded budget"
        );
        const auto *candidates = static_cast<const float *>(hardware.num_steps().contents);
        double total_candidates = 0.0;
        float maximum_candidates = 0.0F;
        for (size_t i = 0; i < field.rays.size(); ++i) {
          total_candidates += candidates[i];
          maximum_candidates = std::max(maximum_candidates, candidates[i]);
        }
        std::printf(
            "BVH candidates: mean %.3f, maximum %.0f per ray.\n",
            total_candidates / double(field.rays.size()),
            maximum_candidates
        );
        software.print_statistics();
        hardware.print_statistics();
      } else {
        char temporary[] = "/tmp/panorama-bvh-test.XXXXXX";
        const char *created = mkdtemp(temporary);
        require(created != nullptr, "Could not create fixture directory");
        const std::filesystem::path root(created);
        std::printf("Test fixtures: %s\n", created);
        write_fixture(root / "float", false, 10.0);
        write_fixture(root / "quantized", true, 10.0);
        write_fixture(root / "distant", true, 1000.0);
        if (streaming) {
          check_streaming(root / "quantized");
        } else if (edge_cases) {
          @autoreleasepool {
            check_limits_and_optional_outputs(root / "quantized");
          }
          @autoreleasepool {
            exercise(root / "quantized", false, 10.0, 4U);
          }
          @autoreleasepool {
            exercise(root / "distant", true, 1000.0, 4U);
          }
        } else {
          for (uint32_t block : {1U, 4U, 7U}) {
            @autoreleasepool {
              exercise(root / "float", false, 10.0, block);
            }
            @autoreleasepool {
              exercise(root / "quantized", true, 10.0, block);
            }
          }
        }
      }
    }
    std::puts("Metal BVH tests passed.");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
