#include "arguments.h"
#include "gpu_terrain_frame.h"
#include "metal_bvh_types.metalh"
#include "metal_tile.h"
#include "metalfx_upscaler.h"
#include "terrain_manifest.h"
#include "terrain_trace_session.h"
#include "terrain_transform.h"
#include "timer.h"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <chrono>
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
size_t pixel_count(ImageSize image) { return size_t(image.width) * image.height; }
RayFieldRequest angular_field(ImageSize image, AngularProjection projection) {
  return {image, projection};
}
RayFieldRequest camera_field(ImageSize image, Projection projection) { return {image, projection}; }
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Match the viewer's default camera and output requirements, excluding image
// generation, lighting and display. Run each backend in a separate process.
void benchmark(int argc, const char *argv[]) {
  if (argc < 5 || argc > 11)
    throw std::invalid_argument(
        "usage: metal-bvh-test --benchmark TILE_DIR software|metal-bvh CACHE_MIB "
        "[RANGE [WIDTH HEIGHT [LOD_SCALE [DEBUG_OUTPUTS [TILE_BVH]]]]]"
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
  config.use_tile_bvh = argc <= 10 || std::stoi(argv[10]) != 0;
  const bool debugging = argc <= 9 || std::stoi(argv[9]) != 0;
  const auto intrinsics =
      CameraIntrinsics::from_vertical_field_of_view(image, 70.0 * std::numbers::pi / 180.0);
  auto field = camera_field(image, CameraProjection{{0, 0, 0}, intrinsics, NoDistortion{}});
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
        field = camera_field(
            image,
            CameraProjection{{std::numbers::pi / 180.0, 0, 0}, intrinsics, NoDistortion{}}
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

// Includes GPU projection/LOD preparation and the complete producer.
void benchmark_camera(int argc, const char *argv[]) {
  if (argc != 4 && argc != 5)
    throw std::invalid_argument(
        "usage: metal-bvh-test --benchmark-camera TILE_DIR gpu|gpu-shadows [FALLBACK_TILE_DIR]"
    );
  const bool shadows = std::string_view(argv[3]) == "gpu-shadows";
  require(shadows || std::string_view(argv[3]) == "gpu", "Choose gpu or gpu-shadows");
  RaytraceConfig config{argv[2],
                        {2623452.4, 1100502.2, 3415},
                        21000,
                        0,
                        128ULL * 1048576,
                        4,
                        true,
                        true,
                        false};
  config.lod_scale = 1.5F;
  config.raytracer = Raytracer::MetalBvh;
  config.bvh_cache_size_bytes = 2048ULL * 1048576;
  if (argc == 5) {
    config.terrain_datasets = {
        TerrainDatasetConfig{argv[2], 0.0},
        TerrainDatasetConfig{argv[4], 0.0},
    };
  }
  if (shadows)
    config.max_distance = 600000;
  const ImageSize image{1600, 900};
  RayFieldRequest camera{
      image,
      CameraProjection{
          {0, 0, 0},
          CameraIntrinsics::from_vertical_field_of_view(image, 70 * std::numbers::pi / 180),
          NoDistortion{}}};
  auto session = std::make_unique<TerrainTraceSession>(
      config,
      camera,
      GpuTraceOutputRequirements{true, true, true}
  );
  GpuImageRenderer renderer(
      session->device(),
      session->command_queue(),
      session->library(),
      image,
      {false, false, false, true, true, true}
  );
  TerrainPresentationSettings settings{};
  settings.appearance.raytraced_shadows = shadows;
  settings.appearance.sun_azimuth = 2.1;
  settings.appearance.sun_elevation = 0.35;
  settings.colour_range = {0, 21000};
  std::vector<double> warm;
  for (uint32_t frame = 0; frame < 18; ++frame) {
    @autoreleasepool {
      // Warm pans, followed by movement and zoom to exercise plan invalidation.
      std::get<CameraProjection>(camera.projection).orientation.heading = 0.001 * frame;
      if (frame == 14) {
        config.observer.easting += 1;
        require(session->relocate_observer(config.observer), "Benchmark relocation failed");
      }
      if (frame == 16)
        std::get<CameraProjection>(camera.projection).intrinsics =
            CameraIntrinsics::from_vertical_field_of_view(image, 1.1);
      const auto started = std::chrono::steady_clock::now();
      const auto before = session->bvh_statistics();
      const auto io_before = session->tile_statistics().bytes_loaded_with_metal_io;
      const auto timing = render_terrain_frame(*session, &camera, renderer, settings);
      const double wall =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
              .count();
      const auto preparation = session->camera_statistics();
      if (shadows) {
        const auto after = session->bvh_statistics();
        std::printf(
            "Shadow frame %u: repair %.3f ms, caster builds=%llu, repair passes=%llu, "
            "capacity fallbacks=%llu, terrain I/O %.3f MiB\n",
            frame,
            timing.shadow_repair_milliseconds,
            static_cast<unsigned long long>(after.shadow_tiles_built - before.shadow_tiles_built),
            static_cast<unsigned long long>(after.shadow_passes - before.shadow_passes),
            static_cast<unsigned long long>(
                after.shadow_cache_fallbacks - before.shadow_cache_fallbacks
            ),
            double(session->tile_statistics().bytes_loaded_with_metal_io - io_before) / 1048576.0
        );
      }
      std::printf(
          "Camera benchmark %s frame=%u: total %.3f ms, "
          "GPU plan wall/device %.3f/%.3f ms, producer %.3f ms, streamed=%d\n",
          shadows ? "gpu-shadows" : "gpu",
          frame,
          wall,
          preparation.preparation_wall_ms,
          preparation.preparation_gpu_ms,
          timing.gpu_milliseconds,
          timing.streamed
      );
      if (frame >= 2 && frame < 14)
        warm.push_back(wall);
    }
  }
  std::sort(warm.begin(), warm.end());
  std::printf(
      "Camera benchmark %s warm pan median %.3f ms, p95 %.3f ms (%zu samples)\n",
      shadows ? "gpu-shadows" : "gpu",
      0.5 * (warm[5] + warm[6]),
      warm.back(),
      warm.size()
  );
}

void write_fixture(
    const std::filesystem::path &directory,
    bool quantized,
    double spacing,
    uint32_t epsg = 2056U,
    double origin_x = 2600000.0,
    double origin_y = 1200000.0,
    double elevation_offset = 0.0
) {
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
                        10.0 * (1000.0 + elevation_offset +
                                70.0 * std::sin(east * 0.17) * std::cos(north * 0.11))
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
                                epsg,
                                cells,
                                5U,
                                base.maximum_elevation,
                                quantized ? MetalTileSampleType::Uint16Decimeters
                                          : MetalTileSampleType::Float32,
                                base.elevation_base_decimeters,
                                0U,
                                row,
                                column,
                                origin_x + double(column) * cells * spacing,
                                origin_y - double(row + 1) * cells * spacing,
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
      manifest.push_back(
          {row, column, float(1071.0 + elevation_offset), float(929.0 + elevation_offset)}
      );
    }
  }
  // Leave float fixtures without a sidecar to retain legacy/no-manifest coverage.
  if (quantized)
    write_terrain_manifest(terrain_manifest_path(directory), manifest);
}

void check_dataset_foundation(const std::filesystem::path &root) {
  const auto geographic = root / "geographic";
  write_fixture(geographic, true, 1.0 / 3600.0, 4326U, 7.0, 46.0);
  const std::array<TerrainDatasetConfig, 2> configs = {
      TerrainDatasetConfig{root / "quantized", 0.0},
      TerrainDatasetConfig{geographic, -2.5},
  };
  const auto datasets = discover_terrain_datasets(configs);
  require(datasets.size() == 2, "Dataset stack lost a configured dataset");
  require(datasets[0].epsg_code == 2056U, "Navigation dataset CRS changed");
  require(datasets[1].epsg_code == 4326U, "Fallback dataset CRS changed");
  require(datasets[1].config.vertical_offset_metres == -2.5, "Vertical offset changed");
  require(
      datasets[0].cell_count == datasets[1].cell_count &&
          datasets[0].lod_count == datasets[1].lod_count &&
          datasets[0].sample_type == datasets[1].sample_type,
      "Compatible dataset payloads were not retained"
  );
  require(
      datasets[0].sources.front().dataset_index == 0U &&
          datasets[1].sources.front().dataset_index == 1U,
      "Dataset-local tile keys did not receive distinct source identities"
  );
  const TerrainRenderFrame fixed = select_terrain_render_frame(
      std::span<const TerrainDataset>(datasets).first(1),
      {2600000, 1200000}
  );
  require(
      fixed.kind == TerrainRenderFrame::Kind::FixedEpsg && fixed.fixed_epsg == 2056U,
      "One projected-metre dataset did not retain its native render frame"
  );

  const MetalTileHeader header = read_metal_tile_header(datasets[1].sources.front().path);
  const TerrainRenderFrame frame = select_terrain_render_frame(datasets, {2600000.0, 1200000.0});
  require(
      frame.kind == TerrainRenderFrame::Kind::LocalAzimuthalEquidistant,
      "Mixed datasets did not select a local metric frame"
  );
  const std::array<Coord, 1> observer = {{{2600000.0, 1200000.0}}};
  const Coord local_observer = frame.project(2056U, observer).front();
  const Coord recovered_observer = frame.unproject(2056U, std::span(&local_observer, 1)).front();
  std::printf(
      "Local-frame centre residual: %.9f, round-trip residual: %.9f m.\n",
      std::hypot(local_observer.x, local_observer.y),
      std::hypot(recovered_observer.x - observer[0].x, recovered_observer.y - observer[0].y)
  );
  require(
      std::hypot(local_observer.x, local_observer.y) < 1e-3 &&
          std::hypot(recovered_observer.x - observer[0].x, recovered_observer.y - observer[0].y) <
              0.01,
      "Local render frame is not centred on or reversible at the observer"
  );
  const TerrainTileTransform transform = make_terrain_tile_transform(header, frame);
  const auto patches = make_terrain_transform_patches(header, frame);
  const Coord logical{5.25, 8.75};
  const Coord projected = transform.apply(logical.x, logical.y);
  const Coord recovered = transform.inverse(projected);
  require(
      std::hypot(recovered.x - logical.x, recovered.y - logical.y) < 1e-8,
      "Tile transform inverse changed logical coordinates"
  );
  require(
      transform.maximum_cell_size_metres() > 15.0 && transform.maximum_cell_size_metres() < 40.0 &&
          std::isfinite(transform.maximum_residual_metres),
      "Geographic terrain transform has implausible metre geometry"
  );
  require(
      std::all_of(
          patches.begin(),
          patches.end(),
          [](const auto &patch) { return patch.transform.maximum_residual_metres <= 1.0; }
      ),
      "Adaptive terrain transform exceeded its residual bound"
  );
  const std::array<TerrainTransformPatch, 2> split_ownership = {
      TerrainTransformPatch{0U, 0U, 256U, 512U, {}},
      TerrainTransformPatch{256U, 0U, 256U, 512U, {}},
  };
  const auto coverage = make_terrain_coverage_polygons(header, frame, split_ownership, 256U);
  require(
      coverage.size() == 2U && coverage[0].vertices.size() == 6U &&
          coverage[1].vertices.size() == 6U,
      "Coverage tessellation did not respect the shared grid"
  );
  const auto same_point = [](Coord left, Coord right) {
    return std::hypot(left.x - right.x, left.y - right.y) < 1e-9;
  };
  const auto shared_edge_count =
      [&same_point](const TerrainCoveragePolygon &left, const TerrainCoveragePolygon &right) {
        uint32_t count = 0U;
        for (size_t left_index = 0; left_index < left.vertices.size(); ++left_index) {
          const Coord left_start = left.vertices[left_index];
          const Coord left_end = left.vertices[(left_index + 1U) % left.vertices.size()];
          for (size_t right_index = 0; right_index < right.vertices.size(); ++right_index) {
            const Coord right_start = right.vertices[right_index];
            const Coord right_end = right.vertices[(right_index + 1U) % right.vertices.size()];
            if ((same_point(left_start, right_start) && same_point(left_end, right_end)) ||
                (same_point(left_start, right_end) && same_point(left_end, right_start)))
              ++count;
          }
        }
        return count;
      };
  require(
      shared_edge_count(coverage[0], coverage[1]) == 2U,
      "Coverage ownership regions did not share identical projected edges"
  );
  std::printf(
      "Dataset foundation: %zu + %zu sources, EPSG:%u -> local AEQD anchored from EPSG:%u, "
      "%.3f m/cell, "
      "%.6f m whole-tile residual.\n",
      datasets[0].sources.size(),
      datasets[1].sources.size(),
      datasets[1].epsg_code,
      datasets[0].epsg_code,
      transform.maximum_cell_size_metres(),
      transform.maximum_residual_metres
  );

  const TerrainCatalogue combined =
      TerrainCatalogue::discover(configs, {2600000.0, 1200000.0, 1000.0}, 200000.0F, 0U);
  require(combined.datasets().size() == 2U, "Combined catalogue lost dataset metadata");
  require(
      combined.render_frame().kind == TerrainRenderFrame::Kind::LocalAzimuthalEquidistant,
      "Combined mixed catalogue did not retain its local render frame"
  );
  const TileKey duplicate_key = datasets[0].sources.front().key;
  require(
      combined.find_source(0U, duplicate_key).has_value() &&
          combined.find_source(1U, duplicate_key).has_value() &&
          combined.find_source(0U, duplicate_key) != combined.find_source(1U, duplicate_key),
      "Dataset-local duplicate tile keys did not remain distinct"
  );
  for (const TerrainSource &source : combined.sources()) {
    require(!source.transform_patches.empty(), "Combined source has no render transform");
    require(!source.coverage_polygons.empty(), "Combined source has no coverage tessellation");
    require(
        std::all_of(
            source.transform_patches.begin(),
            source.transform_patches.end(),
            [](const auto &patch) {
              return std::isfinite(patch.transform.maximum_residual_metres) &&
                     patch.transform.maximum_residual_metres <= 0.25;
            }
        ),
        "Combined source geometry transform exceeded its seam bound"
    );
  }

  const std::array<TerrainDatasetConfig, 2> overlapping = {
      TerrainDatasetConfig{root / "quantized", 0.0},
      TerrainDatasetConfig{root / "overlap", 0.0},
  };
  const TerrainCatalogue owned =
      TerrainCatalogue::discover(overlapping, {2600045.0, 1199945.0, 1120.0}, 1000.0F, 0U);
  require(
      std::none_of(
          owned.sources().begin(),
          owned.sources().end(),
          [](const TerrainSource &source) { return source.dataset_index == 1U; }
      ),
      "Fully covered lower-priority sources retained GPU ownership"
  );
}

void check_prepared_dataset_stack(
    const std::filesystem::path &primary,
    const std::filesystem::path &fallback
) {
  const std::array<TerrainDatasetConfig, 2> configs = {
      TerrainDatasetConfig{primary, 0.0},
      TerrainDatasetConfig{fallback, 0.0},
  };
  const auto datasets = discover_terrain_datasets(configs);
  const TerrainRenderFrame frame = select_terrain_render_frame(datasets, {2623452.4, 1100502.2});
  const MetalTileHeader header = read_metal_tile_header(datasets[1].sources.front().path);
  const TerrainTileTransform transform = make_terrain_tile_transform(header, frame);
  const auto patches = make_terrain_transform_patches(header, frame);
  const auto started = std::chrono::steady_clock::now();
  const TerrainCatalogue combined =
      TerrainCatalogue::discover(configs, {2623452.4, 1100502.2, 3415.0}, 600000.0F, 0U);
  size_t combined_patches = 0U;
  for (const TerrainSource &source : combined.sources())
    combined_patches += source.transform_patches.size();
  const double catalogue_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  std::printf(
      "Prepared stack: %zu EPSG:%u tiles over %zu EPSG:%u tiles; fallback %.3f m/cell, "
      "%.3f m representative affine residual, %zu patches at <= 1 m.\n",
      datasets[0].sources.size(),
      datasets[0].epsg_code,
      datasets[1].sources.size(),
      datasets[1].epsg_code,
      transform.maximum_cell_size_metres(),
      transform.maximum_residual_metres,
      patches.size()
  );
  std::printf(
      "Combined catalogue: %zu retained sources, %zu transform patches in %.1f ms.\n",
      combined.sources().size(),
      combined_patches,
      catalogue_ms
  );
}

void compare(
    TerrainTraceSession &software,
    TerrainTraceSession &hardware,
    const RayFieldRequest &field,
    float distance_tolerance = 0.01F,
    const MetalTileHeader *real_grid = nullptr
) {
  software.trace(field);
  hardware.trace(field);
  const auto *directions = static_cast<const RayDirection *>(hardware.ray_directions().contents);
  const auto *expected = static_cast<const float *>(software.distances().contents);
  const auto *actual = static_cast<const float *>(hardware.distances().contents);
  const auto *expected_z = static_cast<const float *>(software.elevations().contents);
  const auto *actual_z = static_cast<const float *>(hardware.elevations().contents);
  const auto *expected_n = static_cast<const uint32_t *>(software.surface_gradients().contents);
  const auto *actual_n = static_cast<const uint32_t *>(hardware.surface_gradients().contents);
  size_t mismatches = 0, hits = 0;
  float maximum_error = 0.0F;
  for (size_t i = 0; i < pixel_count(field.image); ++i) {
    if (expected[i] > 0)
      ++hits;
    const float delta = std::abs(expected[i] - actual[i]);
    maximum_error = std::max(maximum_error, delta);
    const float elevation_tolerance =
        real_grid == nullptr
            ? 0.02F
            : 0.002F + distance_tolerance * (std::abs(directions[i].slope) +
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
            directions[i].x,
            directions[i].y,
            directions[i].slope
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
            observer.easting + expected[i] * double(directions[i].x) - real_grid->lower_left_x;
        const double y =
            observer.northing + expected[i] * double(directions[i].y) - real_grid->lower_left_y;
        const double actual_x =
            observer.easting + actual[i] * double(directions[i].x) - real_grid->lower_left_x;
        const double actual_y =
            observer.northing + actual[i] * double(directions[i].y) - real_grid->lower_left_y;
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
                expected[i] * directions[i].x,
                expected[i] * directions[i].y
            );
          ++mismatches;
        }
      }
    }
  }
  std::printf(
      "Parity: %zu rays, %zu hits, max distance error %.7g m, %zu mismatches.\n",
      pixel_count(field.image),
      hits,
      maximum_error,
      mismatches
  );
  require(hits > 0, "Fixture must hit terrain");
  require(mismatches == 0, "BVH/software parity failed");
}

void check_mixed_priority(const std::filesystem::path &root) {
  const auto field = angular_field({129, 65}, {0.0, 2 * std::numbers::pi, -1.3, 0.1});
  RaytraceConfig reference{root / "quantized",
                           {2600045.0, 1199945.0, 1120.0},
                           480.0F,
                           0U,
                           16384U,
                           2U,
                           true,
                           true,
                           false};
  TerrainTraceSession software(reference, field, {true, true, true});
  RaytraceConfig mixed = reference;
  mixed.raytracer = Raytracer::MetalBvh;
  mixed.terrain_datasets = {
      {root / "quantized", 0.0},
      {root / "overlap", 0.0},
  };
  const TerrainCatalogue owned = TerrainCatalogue::discover(
      mixed.terrain_datasets,
      mixed.observer,
      mixed.max_distance,
      mixed.max_tile_count
  );
  require(
      std::none_of(
          owned.sources().begin(),
          owned.sources().end(),
          [](const TerrainSource &source) { return source.dataset_index == 1U; }
      ),
      "Fully covered lower-priority sources were retained"
  );
  const std::array<TerrainDatasetConfig, 2> partial_configs = {
      TerrainDatasetConfig{root / "quantized", 0.0},
      TerrainDatasetConfig{root / "partial-overlap", 0.0},
  };
  const TerrainCatalogue partial =
      TerrainCatalogue::discover(partial_configs, {2600045.0, 1199945.0, 1120.0}, 1000.0F, 0U);
  const auto primary_location = partial.locate_source({2600100.0, 1199945.0});
  const auto fallback_location = partial.locate_source({2600200.0, 1199945.0});
  require(
      primary_location.has_value() &&
          partial.sources()[primary_location->source_index].dataset_index == 0U,
      "Higher-priority ownership was not retained in an overlap"
  );
  require(
      fallback_location.has_value() &&
          partial.sources()[fallback_location->source_index].dataset_index == 1U,
      "Lower-priority terrain did not fill a higher-priority coverage gap"
  );
  TerrainTraceSession hardware(mixed, field, {true, true, true});
  const MetalTileHeader grid = read_metal_tile_header(owned.origin().path);
  compare(software, hardware, field, 0.1F, &grid);
  std::puts("Mixed ownership: higher-priority surface matched with a closer fallback underneath.");
}

void write_flat_coverage_fixture(
    const std::filesystem::path &directory,
    double x_min,
    double spacing,
    float elevation
) {
  std::filesystem::create_directories(directory);
  constexpr uint32_t cells = 16U;
  const std::vector<uint16_t> heights(17U * 17U, uint16_t(std::lround(elevation * 10.0F) - 9000));
  const MetalTileLod lod = {1U,
                            cells,
                            5U,
                            9000,
                            elevation,
                            0U,
                            kMetalTileLodHeaderSize + sizeof(MetalTileLod),
                            heights.size() * sizeof(uint16_t)};
  const MetalTileHeader header = {kMetalTileLodMagic,
                                  kMetalTileLodVersion,
                                  kMetalTileLodHeaderSize,
                                  MetalTileCompression::None,
                                  2056U,
                                  cells,
                                  5U,
                                  elevation,
                                  MetalTileSampleType::Uint16Decimeters,
                                  9000,
                                  0U,
                                  0,
                                  0,
                                  x_min,
                                  1199840.0,
                                  spacing,
                                  lod.vertex_offset,
                                  lod.vertex_byte_count,
                                  1U,
                                  uint32_t(sizeof(MetalTileLod)),
                                  kMetalTileLodHeaderSize,
                                  sizeof(MetalTileLod)};
  std::vector<std::byte> payload(lod.vertex_byte_count);
  std::memcpy(payload.data(), heights.data(), payload.size());
  const std::array<MetalTileLod, 1> lods = {lod};
  const std::array<TerrainManifestEntry, 1> manifest = {
      TerrainManifestEntry{0, 0, elevation, elevation}};
  write_metal_tile_lods(directory / "flat_r0_c0.ptile", header, lods, payload);
  write_terrain_manifest(terrain_manifest_path(directory), manifest);
}

void check_misaligned_coverage(const std::filesystem::path &root) {
  // The primary ends at x=160. Fallback cell centres at x=153 and x=173
  // round the ownership handoff to x=163, leaving a 3 m ownership sliver
  // even though the two physical datasets overlap by 77 m.
  const auto primary = root / "coverage-primary";
  const auto fallback = root / "coverage-fallback";
  const auto gap = root / "coverage-gap";
  const auto raised = root / "coverage-raised";
  const auto reference = root / "coverage-reference";
  write_flat_coverage_fixture(primary, 2600000.0, 10.0, 1000.0F);
  write_flat_coverage_fixture(fallback, 2600083.0, 20.0, 1000.0F);
  write_flat_coverage_fixture(gap, 2600183.0, 20.0, 1000.0F);
  write_flat_coverage_fixture(raised, 2600083.0, 20.0, 1010.0F);
  write_flat_coverage_fixture(reference, 2600000.0, 40.0, 1000.0F);
  const auto far_field = angular_field({9, 3}, {1.5, 1.6, std::atan(-0.11), std::atan(-0.09)});
  const auto near_field = angular_field({9, 3}, {1.5, 1.6, std::atan(-0.21), std::atan(-0.19)});
  RaytraceConfig config{reference, {2600045, 1199945, 1020}, 500, 0, 16384, 2, true, true, false};
  TerrainTraceSession expected(config, far_field, {true, true, true});
  expected.trace(far_field);
  const auto *distances = static_cast<const float *>(expected.distances().contents);
  require(
      std::all_of(
          distances,
          distances + pixel_count(far_field.image),
          [](float t) { return t > 180.0F && t < 225.0F; }
      ),
      "Coverage regression reference did not hit terrain beyond the ownership sliver"
  );
  config.tile_dir = primary;
  config.raytracer = Raytracer::MetalBvh;
  auto *geometry = [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
  geometry.boundingBoxCount = 16;
  geometry.boundingBoxStride = 24;
  geometry.opaque = NO;
  geometry.allowDuplicateIntersectionFunctionInvocation = NO;
  auto *descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
  descriptor.geometryDescriptors = @[ geometry ];
  const auto sizes = [expected.device() accelerationStructureSizesWithDescriptor:descriptor];
  const uint64_t one_tile_budget = 17U * 17U * 2U + 16U * (sizeof(BvhBlock) + sizeof(BvhBounds)) +
                                   sizeof(BvhAffinePatch) + sizeof(BvhTile) + sizeof(uint64_t) +
                                   2U * sizes.accelerationStructureSize +
                                   sizes.buildScratchBufferSize;
  for (bool bounded : {false, true}) {
    for (bool bilinear : {false, true}) {
      @autoreleasepool {
        config.bvh_cache_size_bytes = bounded ? one_tile_budget : 1048576U;
        config.bilinear_collisions = bilinear;
        config.terrain_datasets = {{primary, 0.0}, {fallback, 0.0}};
        TerrainTraceSession mixed(config, far_field, {true, true, true});
        compare(expected, mixed, far_field, 0.01F);
        compare(expected, mixed, far_field, 0.01F);
        if (bounded)
          require(mixed.bvh_statistics().evictions > 0, "Coverage fixture did not force streaming");
        config.terrain_datasets = {{primary, 0.0}, {gap, 0.0}};
        TerrainTraceSession missing(config, far_field, {true, true, true});
        missing.trace(far_field);
        const auto *missing_distances = static_cast<const float *>(missing.distances().contents);
        require(
            std::none_of(
                missing_distances,
                missing_distances + pixel_count(far_field.image),
                [](float t) { return t > 0.0F; }
            ),
            "Physical coverage proof accepted terrain beyond a real missing-data gap"
        );
        config.terrain_datasets = {{primary, 0.0}, {raised, 0.0}};
        TerrainTraceSession priority(config, near_field, {true, true, true});
        compare(expected, priority, near_field, 0.01F);
      }
    }
  }
  std::puts(
      "Mixed coverage: crossed grid-rounding slivers, stopped at real gaps, retained priority."
  );
}

void check_tile_selection(const std::filesystem::path &directory, bool retain, double spacing) {
  RaytraceConfig config{directory,
                        {2600000.0 + 4.5 * spacing, 1200000.0 - 5.5 * spacing, 1120.0},
                        float(48.0 * spacing),
                        0U,
                        16384U,
                        2U,
                        retain,
                        true,
                        false};
  auto field = angular_field({129, 65}, {0, 2 * std::numbers::pi, -1.3, 0.1});
  config.use_tile_bvh = false;
  TerrainTraceSession grid(config, field, {true, true, true});
  config.use_tile_bvh = true;
  TerrainTraceSession shared(config, field, {true, true, true});
  const float tolerance = spacing > 10 ? 0.25F : 0.01F;
  compare(grid, shared, field, tolerance);
  compare(grid, shared, field, tolerance);
  field = angular_field({97, 33}, {0.35, 6.6, -1.25, 0.05});
  grid.set_collision_options(false, true);
  shared.set_collision_options(false, true);
  compare(grid, shared, field, tolerance);
  auto moved = config.observer;
  moved.easting -= 8 * spacing;
  moved.northing += 3 * spacing;
  require(
      grid.relocate_observer(moved) && shared.relocate_observer(moved),
      "Tile selector relocation"
  );
  compare(grid, shared, field, tolerance);
  grid.set_collision_options(true, true);
  shared.set_collision_options(true, true);
  grid.set_lod_scale(3);
  shared.set_lod_scale(3);
  compare(grid, shared, field, tolerance);
  grid.trace_shadows(2.1, 0.35);
  shared.trace_shadows(2.1, 0.35);
  require(
      std::memcmp(
          grid.shadow_visibility().contents,
          shared.shadow_visibility().contents,
          pixel_count(field.image)
      ) == 0,
      "Tile selector changed shadows"
  );
  ObserverLocation corner{2600000, 1200000, 1120};
  require(grid.relocate_observer(corner) && shared.relocate_observer(corner), "Corner relocation");
  compare(grid, shared, field, tolerance);
  grid.set_lod_scale(3);
  shared.set_lod_scale(3);
  compare(grid, shared, field, tolerance);
  // Both backends must consume the existing catalogue after switching.
  shared.set_raytracer(Raytracer::MetalBvh);
  compare(grid, shared, field, tolerance);
  require(shared.bvh_statistics().catalogue_builds == 0, "Backend switch rebuilt shared catalogue");
  shared.set_raytracer(Raytracer::Software);
  compare(grid, shared, field, tolerance);
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
  auto field = angular_field({129, 65}, {0.0, 2 * std::numbers::pi, -1.3, 0.1});
  const GpuTraceOutputRequirements outputs{true, true, true};
  config.use_tile_bvh = false;
  TerrainTraceSession software(config, field, outputs);
  config.raytracer = Raytracer::MetalBvh;
  config.bvh_block_cells = block;
  TerrainTraceSession hardware(config, field, outputs);
  std::printf("Fixture: retain=%d spacing=%g block=%u\n", retain, spacing, block);
  compare(software, hardware, field, spacing > 10 ? 0.25F : 0.01F);
  // The first repeat assembles the resident scene. Further repeats must use
  // that hierarchy in one dispatch without CPU grouping or instance rebuilds.
  compare(software, hardware, field, spacing > 10 ? 0.25F : 0.01F);
  const auto warm = hardware.bvh_statistics();
  compare(software, hardware, field, spacing > 10 ? 0.25F : 0.01F);
  const auto repeated = hardware.bvh_statistics();
  require(
      repeated.scene_passes == warm.scene_passes + 1U &&
          repeated.submissions == warm.submissions + 1U &&
          repeated.instance_builds == warm.instance_builds && repeated.builds == warm.builds &&
          repeated.scene_fallback_rays == warm.scene_fallback_rays &&
          repeated.grouping_cpu_ms == warm.grouping_cpu_ms,
      "Warm scene did not reuse one GPU traversal"
  );
  // Same session, different camera and dimensions: residency must be declared again.
  auto next = angular_field({97, 33}, {0.35, 6.6, -1.25, 0.05});
  software.set_collision_options(false, true);
  hardware.set_collision_options(false, true);
  compare(software, hardware, next, spacing > 10 ? 0.25F : 0.01F);
  // A changed view begins with the existing scene. Direct repair may then
  // admit newly requested tiles and publish an updated scene immediately.
  const auto changed = hardware.bvh_statistics();
  require(
      (changed.builds == repeated.builds && changed.scene_builds == repeated.scene_builds) ||
          (changed.builds > repeated.builds && changed.scene_builds > repeated.scene_builds),
      "Camera change rebuilt the scene without admitting detailed terrain"
  );
  const uint64_t builds_before_move = changed.builds;
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
          pixel_count(next.image)
      ) == 0,
      "Shadow output changed"
  );
}

void check_limits_and_optional_outputs(const std::filesystem::path &directory) {
  auto field = angular_field({33, 17}, {0.0, 6.3, -1.3, 0.1});
  RaytraceConfig
      config{directory, {2600045.0, 1199945.0, 1120.0}, 40.0F, 0U, 16384U, 2U, true, true, false};
  config.raytracer = Raytracer::MetalBvh;
  for (bool bilinear : {false, true}) {
    config.bilinear_collisions = bilinear;
    TerrainTraceSession trace(config, field, {false, false, false});
    trace.trace(field);
    trace.trace(field);
    const auto warm = trace.bvh_statistics();
    const auto *values = static_cast<const float *>(trace.distances().contents);
    for (size_t i = 0; i < pixel_count(field.image); ++i) {
      require(
          std::isfinite(values[i]) && values[i] >= 0 && values[i] <= config.max_distance,
          "BVH escaped its finite distance interval"
      );
    }
    auto observer = config.observer;
    observer.elevation += 40.0;
    require(trace.relocate_observer(observer), "Height-only relocation failed");
    trace.trace(field);
    require(
        trace.bvh_statistics().scene_builds == warm.scene_builds &&
            trace.bvh_statistics().catalogue_builds == warm.catalogue_builds,
        "Height-only relocation rebuilt the scene"
    );
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
  config.bvh_block_cells = 4;
  config.bvh_cache_size_bytes = 0;
  rejected = false;
  try {
    TerrainTraceSession invalid(config, field, {false, false, false});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "Zero BVH cache size must be rejected");
  config.bvh_cache_size_bytes = 1;
  config.max_distance = 480;
  TerrainTraceSession too_small(config, field, {false, false, false});
  rejected = false;
  try {
    too_small.trace(field);
  } catch (const std::runtime_error &error) {
    rejected = std::string_view(error.what()).find("--bvh-cache-mib") != std::string_view::npos;
  }
  require(rejected, "Undersized BVH cache must explain the required budget");
  const auto failed = too_small.bvh_statistics();
  require(
      failed.builds == 0 && failed.resident_bytes == 0 && failed.peak_bytes == 0,
      "Undersized cache allocated detailed tile resources before rejection"
  );
}

void check_scene_misses(const std::filesystem::path &directory) {
  RaytraceConfig
      config{directory, {2600045.0, 1199945.0, 1120.0}, 480.0F, 0U, 16384U, 2U, true, true, false};
  // A steep ray warms only nearby terrain. A wider view then contains both
  // resolvable resident hits and rays that must load previously unseen tiles.
  const auto narrow = angular_field({1, 1}, {0.0, 0.01, -1.55, -1.54});
  const auto wide = angular_field({129, 65}, {0.0, 6.3, -1.3, 0.1});
  config.use_tile_bvh = false;
  TerrainTraceSession software(config, narrow, {true, true, true});
  config.raytracer = Raytracer::MetalBvh;
  TerrainTraceSession hardware(config, narrow, {true, true, true});
  compare(software, hardware, narrow, 0.01F);
  const auto before = hardware.bvh_statistics();
  compare(software, hardware, wide, 0.01F);
  const auto after = hardware.bvh_statistics();
  require(
      after.scene_passes > before.scene_passes + 1U &&
          after.scene_fallback_rays > before.scene_fallback_rays &&
          after.scene_fallback_rays - before.scene_fallback_rays < pixel_count(wide.image) &&
          after.builds > before.builds && after.streaming_rays == before.streaming_rays,
      "Partial scene did not repair requested sources directly"
  );
  compare(software, hardware, wide, 0.01F);
  const auto reused = hardware.bvh_statistics();
  require(
      reused.scene_builds == after.scene_builds &&
          reused.scene_fallback_rays == after.scene_fallback_rays &&
          reused.submissions == after.submissions + 1U,
      "Repaired scene was not immediately reusable"
  );
  compare(software, hardware, wide, 0.01F);
  require(
      hardware.bvh_statistics().submissions == reused.submissions + 1U,
      "Updated scene did not reuse one GPU traversal"
  );
}

void check_session_replacement(
    const std::filesystem::path &directory,
    Raytracer backend,
    bool pinhole = false
) {
  const RayFieldRequest camera{
      {97, 33},
      CameraProjection{{0, -1.4, 0.13},
                       CameraIntrinsics::from_vertical_field_of_view({97, 33}, 0.2),
                       NoDistortion{}}};
  const auto field = pinhole ? camera_field(camera.image, camera.projection)
                             : angular_field({97, 33}, {0, 6.3, -1.3, 0.1});
  // One retained tile forces the same replacement path as leaving the viewer's catalogue.
  RaytraceConfig config{directory, {2600045, 1199945, 1120}, 480, 1, 16384, 2, true, true, false};
  config.raytracer = backend;
  const GpuTraceOutputRequirements outputs{true, true, true};
  const auto make_session = [&](id<MTLCommandQueue> queue = nil) {
    return std::make_unique<TerrainTraceSession>(config, field, outputs, queue);
  };
  auto session = make_session();
  const auto device = session->device();
  const auto queue = session->command_queue();
  const GpuPresentationRequirements products{false, false, false, true, true, true};
  GpuImageRenderer image(device, queue, session->library(), field.image, products);
  TerrainPresentationSettings settings{};
  settings.colour_range = {0, 480};
  settings.appearance.colour_source = TerrainColourSource::Distance;
  settings.appearance.ambient_light = 0.3F;
  settings.appearance.raytraced_shadows = true;
  settings.appearance.sun_azimuth = 2.1;
  settings.appearance.sun_elevation = 0.35;
  Timer timer("Session replacement");
  (void)render_terrain_frame(*session, &field, image, settings, {});

  for (const ObserverLocation observer :
       {ObserverLocation{2599965, 1199975, 1120}, ObserverLocation{2600045, 1199945, 1120}}) {
    @autoreleasepool {
      require(!session->relocate_observer(observer), "Fixture did not leave the catalogue");
      config.observer = observer;
      session = make_session(queue);
      require(
          session->command_queue() == queue && session->device() == device,
          "Session replacement changed the viewer's device or queue"
      );
      // Keep the original presentation renderer and queue alive across replacements.
      (void)render_terrain_frame(*session, &field, image, settings, {});
      const auto rendered = image.readback(timer);

      auto reference_config = config;
      reference_config.raytracer = Raytracer::Software;
      reference_config.use_tile_bvh = false;
      TerrainTraceSession reference(reference_config, field, outputs);
      GpuImageRenderer expected(
          reference.device(),
          reference.command_queue(),
          reference.library(),
          field.image,
          products
      );
      (void)render_terrain_frame(reference, &field, expected, settings);
      const auto baseline = expected.readback(timer);
      require(
          rendered.bytes.size() == baseline.bytes.size(),
          "Replacement image dimensions changed"
      );
      require(
          std::any_of(
              static_cast<const float *>(reference.distances().contents),
              static_cast<const float *>(reference.distances().contents) + pixel_count(field.image),
              [](float value) { return value > 0; }
          ),
          "Replacement fixture did not render terrain"
      );
      for (size_t i = 0; i < rendered.bytes.size(); ++i)
        require(
            std::abs(int(rendered.bytes[i]) - int(baseline.bytes[i])) <= 2,
            "Replacement session changed rendered terrain"
        );
    }
  }
  std::printf(
      "Session replacement preserved device, queue and image parity (%s).\n",
      backend == Raytracer::Software ? "Mipmap" : "BVH"
  );
}

RayFieldRequest camera_request(ImageSize image, bool tiny = false) {
  return {image,
          CameraProjection{{0.0, tiny ? -1.54 : -0.65, 0.13},
                           CameraIntrinsics::from_vertical_field_of_view(image, 1.2),
                           NoDistortion{}}};
}

void check_gpu_projection(const std::filesystem::path &directory) {
  RaytraceConfig config{directory, {2600045, 1199945, 1120}, 480, 0, 16384, 2, true, true, false};
  TileManager tiles(config);
  GpuRaytraceResources resources(1U, tiles.sources(), true, true, false, {false, false, false});
  GpuCamera camera(resources.device(), resources.command_queue(), resources.library(), tiles);
  const auto generate = [&](const RayFieldRequest &request) {
    resources.resize_rays(validate_camera_request(request));
    camera.prepare(request, config.observer, 1.5F, tiles);
    auto command = [resources.command_queue() commandBuffer];
    camera.encode_rays(command, resources.ray_directions());
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "GPU projection command failed");
    camera.validate_completed_rays();
    const auto *begin = static_cast<const RayDirection *>(resources.ray_directions().contents);
    return std::vector<RayDirection>(begin, begin + pixel_count(request.image));
  };
  for (ImageSize image : {ImageSize{1, 1}, {1, 7}, {9, 1}, {129, 65}, {320, 180}}) {
    for (double fov : {0.000001, 0.15, 1.2, 2.5}) {
      CameraProjection p{{0, 0, 0},
                         CameraIntrinsics::from_vertical_field_of_view(image, fov),
                         NoDistortion{}};
      RayFieldRequest request{image, p};
      const auto rays = generate(request);
      // Project GPU directions back into the image: this checks the geometric
      // contract without maintaining a second ray-generation implementation.
      for (size_t i = 0; i < rays.size(); ++i) {
        const auto &r = rays[i];
        require(std::abs(std::hypot(r.x, r.y) - 1) < 1e-5, "GPU horizontal ray is not unit length");
        require(
            std::abs(
                r.x / r.y * p.intrinsics.focal_x + p.intrinsics.principal_x -
                (double(i % image.width) + .5)
            ) < .001,
            "GPU pinhole horizontal reprojection failed"
        );
        require(
            std::abs(
                -r.slope / r.y * p.intrinsics.focal_y + p.intrinsics.principal_y -
                (double(i / image.width) + .5)
            ) < .001,
            "GPU pinhole vertical reprojection failed"
        );
      }
      const float angle = *static_cast<const float *>(camera.pixel_angle().contents);
      require(std::isfinite(angle) && angle > 0, "GPU footprint is invalid");
      if (rays.size() > 1 && fov >= .15) {
        double minimum = INFINITY;
        const auto measure = [&](const RayDirection &a, const RayDirection &b) {
          const double al = std::hypot(1.0, a.slope), bl = std::hypot(1.0, b.slope);
          minimum = std::min(
              minimum,
              std::hypot(
                  std::hypot(a.x / al - b.x / bl, a.y / al - b.y / bl),
                  a.slope / al - b.slope / bl
              )
          );
        };
        for (size_t i = 0; i < rays.size(); ++i) {
          if (i % image.width + 1 < image.width)
            measure(rays[i], rays[i + 1]);
          if (i + image.width < rays.size())
            measure(rays[i], rays[i + image.width]);
        }
        require(std::abs(angle / minimum - 1) < .001, "GPU footprint disagrees with adjacent rays");
      }
      for (float scale : {0.0F, 1.5F, 10.0F, 100.0F}) {
        camera.prepare(request, config.observer, scale, tiles);
        for (uint32_t i = 0; i < tiles.sources().size(); ++i) {
          const double ratio = scale *
                               tile_minimum_distance(
                                   tiles.catalogue().grid(),
                                   tiles.sources()[i].key,
                                   config.observer
                               ) *
                               angle / tiles.origin_geometry().cell_size;
          const uint32_t lod = tiles.lod_for_source(i), maximum = tiles.sources()[i].lod_count;
          require(lod >= 1 && lod <= maximum, "GPU LOD out of bounds");
          const double spacing = std::ldexp(1.0, int(lod - 1));
          require(lod == 1 || spacing <= ratio * 1.000001, "GPU LOD exceeds pixel footprint");
          require(lod == maximum || 2 * spacing >= ratio / 1.000001, "GPU LOD failed to coarsen");
        }
      }
      camera.prepare(request, config.observer, 1.5F, tiles);
      const auto before = camera.statistics();
      std::get<CameraProjection>(request.projection).orientation = {.3, -.2, .1};
      camera.prepare(request, config.observer, 1.5F, tiles);
      require(
          camera.statistics().plan_updates == before.plan_updates &&
              camera.statistics().footprint_updates == before.footprint_updates,
          "Rotation must reuse GPU footprint and LOD plan"
      );
    }
  }
  auto threshold = camera_request({129, 65});
  camera.prepare(threshold, config.observer, 1, tiles);
  const float footprint = *static_cast<const float *>(camera.pixel_angle().contents);
  for (uint32_t i = 0; i < tiles.sources().size(); ++i) {
    const double distance =
        tile_minimum_distance(tiles.catalogue().grid(), tiles.sources()[i].key, config.observer);
    if (distance == 0)
      continue;
    for (uint32_t level : {2U, 3U}) {
      for (double factor : {.99999, 1.0, 1.00001}) {
        const float scale = float(
            std::ldexp(1.0, int(level - 1)) * factor * tiles.origin_geometry().cell_size /
            (distance * footprint)
        );
        camera.prepare(threshold, config.observer, scale, tiles);
        const uint32_t expected =
            std::min(tiles.sources()[i].lod_count, factor < 1 ? level - 1 : level);
        const uint32_t actual = tiles.lod_for_source(i);
        require(
            actual == expected || (factor == 1 && actual + 1 == expected),
            "GPU LOD boundary changed"
        );
      }
    }
  }
  auto axis = camera_field({1, 1}, CameraProjection{{0, 0, 0}, {1, 1, .5, .5}, NoDistortion{}});
  auto ray = generate(axis).front();
  require(
      ray.x == 0 && ray.y == 1 && ray.slope == 0 && std::isinf(ray.inverse_x),
      "Axis ray changed"
  );
  std::get<CameraProjection>(axis.projection).orientation = {std::numbers::pi / 2,
                                                             std::numbers::pi / 4,
                                                             0};
  ray = generate(axis).front();
  require(
      std::abs(ray.x - 1) < 1e-6 && std::abs(ray.y) < 1e-6 && std::abs(ray.slope - 1) < 1e-6,
      "Camera heading/pitch changed"
  );
  const auto angular = angular_field(
      {4, 2},
      {-std::numbers::pi / 4, 7 * std::numbers::pi / 4, -std::numbers::pi / 2, std::numbers::pi / 2}
  );
  const auto angular_rays = generate(angular);
  const double x[] = {0, 1, 0, -1}, y[] = {1, 0, -1, 0};
  for (size_t i = 0; i < angular_rays.size(); ++i) {
    require(
        std::abs(angular_rays[i].x - x[i % 4]) < 1e-6 &&
            std::abs(angular_rays[i].y - y[i % 4]) < 1e-6 &&
            std::abs(angular_rays[i].slope - (i < 4 ? -1 : 1)) < 1e-6,
        "Angular pixel centres changed"
    );
  }
  require(
      std::abs(*static_cast<const float *>(camera.pixel_angle().contents) - std::numbers::pi / 2) <
          1e-6,
      "Angular LOD footprint changed"
  );
  const ImageSize image{129, 65};
  const CameraIntrinsics intrinsics{90, 85, 60, 35};
  const BrownConradyDistortion distortion{.08, -.015, .001, .002, -.003};
  auto lens = camera_field(image, CameraProjection{{0, 0, 0}, intrinsics, distortion});
  const auto lens_rays = generate(lens);
  for (size_t i = 0; i < lens_rays.size(); ++i) {
    const auto &r = lens_rays[i];
    const double u = r.x / r.y, v = -r.slope / r.y, r2 = u * u + v * v;
    // Independent forward calibration validates the GPU's inverse solver.
    const double radial =
        1 + r2 * (distortion.radial_1 + r2 * (distortion.radial_2 + r2 * distortion.radial_3));
    const double px = (u * radial + 2 * distortion.tangential_1 * u * v +
                       distortion.tangential_2 * (r2 + 2 * u * u)) *
                          intrinsics.focal_x +
                      intrinsics.principal_x;
    const double py = (v * radial + distortion.tangential_1 * (r2 + 2 * v * v) +
                       2 * distortion.tangential_2 * u * v) *
                          intrinsics.focal_y +
                      intrinsics.principal_y;
    require(
        std::abs(px - (double(i % image.width) + .5)) < .001 &&
            std::abs(py - (double(i / image.width) + .5)) < .001,
        "GPU inverse distortion reprojection failed"
    );
  }
  for (bool invalid_lens : {false, true}) {
    auto invalid = axis;
    auto &p = std::get<CameraProjection>(invalid.projection);
    p.orientation = {0, std::numbers::pi / 2, 0};
    if (invalid_lens) {
      invalid = lens;
      std::get<CameraProjection>(invalid.projection).distortion =
          BrownConradyDistortion{-20, 0, 0, 0, 0};
    }
    bool rejected = false;
    try {
      (void)generate(invalid);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "Invalid projection was not rejected");
  }
  for (ImageSize invalid : {ImageSize{0, 1}, {1, 0}, {0xffffffffU, 2}}) {
    axis.image = invalid;
    bool rejected = false;
    try {
      validate_camera_request(axis);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "Invalid image dimensions were accepted");
  }
}

void check_producer(
    const std::filesystem::path &directory,
    bool bilinear,
    bool partial = false,
    bool pinhole = false
) {
  RaytraceConfig
      config{directory, {2600045, 1199945, 1120}, 480, 0, 16384, 2, true, bilinear, false};
  auto field = angular_field({129, 65}, {0, 6.3, -1.3, 0.1});
  if (partial)
    field = angular_field({1, 1}, {0, 0.01, -1.55, -1.54});
  auto camera = camera_request(field.image, partial);
  if (pinhole)
    field = camera_field(camera.image, camera.projection);
  TerrainTraceSession reference(config, field, {true, true, true});
  config.raytracer = Raytracer::MetalBvh;
  TerrainTraceSession trace(config, field, {true, true, true});
  const GpuPresentationRequirements products{false, false, false, true, true, true};
  GpuImageRenderer expected(
      reference.device(),
      reference.command_queue(),
      reference.library(),
      field.image,
      products
  );
  GpuImageRenderer
      actual(trace.device(), trace.command_queue(), trace.library(), field.image, products);
  TerrainPresentationSettings settings{};
  settings.colour_range = {0, 480};
  settings.appearance.ambient_light = 0.3F;
  settings.appearance.sun_azimuth = 2.1;
  settings.appearance.sun_elevation = 0.35;
  Timer timer("Producer test");
  for (uint32_t frame = 0; frame < (partial ? 3U : 14U); ++frame) {
    settings.appearance.raytraced_shadows = frame != 12 && !(partial && frame == 0);
    // A pinhole view need not load off-screen shadow casters into the primary
    // BVH cache. Exercise its guaranteed resident path with shadows off first;
    // later frames verify exact shadow repair as well.
    if (pinhole && frame < 7 && !partial)
      settings.appearance.raytraced_shadows = false;
    if (partial && frame == 1)
      field = angular_field({129, 65}, {0, 6.3, -1.3, 0.1});
    settings.appearance.feature_outlines = frame == 2;
    settings.appearance.colour_source =
        frame == 3 ? TerrainColourSource::White : TerrainColourSource::Distance;
    settings.appearance.sun_elevation = frame == 4   ? -0.1
                                        : frame == 5 ? std::numbers::pi / 2
                                                     : 0.35;
    const bool appearance_only = frame == 3 || frame == 12;
    if (frame == 7)
      field = angular_field({161, 97}, {0.2, 6.5, -1.3, 0.1});
    if (frame == 8) {
      const ObserverLocation moved{2599965, 1199975, 1120};
      require(
          reference.relocate_observer(moved) && trace.relocate_observer(moved),
          "Producer relocation failed"
      );
    }
    if (frame == 9) {
      reference.set_lod_scale(3);
      trace.set_lod_scale(3);
    }
    if (frame == 10)
      trace.set_raytracer(Raytracer::Software);
    if (frame == 11)
      trace.set_raytracer(Raytracer::MetalBvh);
    if (pinhole) {
      camera = camera_request(field.image, partial && frame == 0);
      field = camera_field(camera.image, camera.projection);
    }
    if (!appearance_only)
      reference.trace(field);
    if (settings.appearance.raytraced_shadows)
      reference.trace_shadows(settings.appearance.sun_azimuth, settings.appearance.sun_elevation);
    expected.resize(field.image);
    expected.render_synthetic(
        reference.surface_gradients(),
        reference.distances(),
        reference.ray_directions(),
        settings.appearance.colour_source == TerrainColourSource::White ? nil
                                                                        : reference.distances(),
        settings.appearance.raytraced_shadows ? reference.shadow_visibility() : nil,
        settings.appearance,
        settings.colour_range,
        true,
        timer
    );
    const auto timing =
        render_terrain_frame(trace, appearance_only ? nullptr : &field, actual, settings, {});
    if (pinhole && frame > 0 && frame < 7 && !partial) {
      require(trace.camera_statistics().plan_updates == 1, "Unchanged camera rebuilt GPU LOD plan");
      require(
          trace.camera_statistics().footprint_updates == 1,
          "Unchanged camera rebuilt footprint"
      );
    }
    std::printf(
        "Producer bilinear=%d frame=%u: wall %.3f GPU %.3f submits %u streamed %d\n",
        bilinear,
        frame,
        timing.wall_milliseconds,
        timing.gpu_milliseconds,
        timing.producer_submissions,
        timing.streamed
    );
    if (frame == 0)
      require(timing.streamed, "Cold producer did not stream");
    else if (partial && frame == 1)
      require(
          timing.streamed && timing.producer_submissions == 2U,
          "Partial resident producer did not repair missing terrain before presentation"
      );
    else if (frame < 7 && !(pinhole && settings.appearance.raytraced_shadows))
      require(
          !timing.streamed && timing.producer_submissions == 1U,
          "Resident producer required intermediate submissions"
      );
    if (settings.appearance.raytraced_shadows)
      require(
          std::memcmp(
              reference.shadow_visibility().contents,
              trace.shadow_visibility().contents,
              pixel_count(field.image)
          ) == 0,
          "Producer shadow parity failed"
      );
    const auto a = expected.readback(timer);
    const auto b = actual.readback(timer);
    require(a.bytes.size() == b.bytes.size(), "Producer image size mismatch");
    uint32_t different = 0;
    for (size_t i = 0; i < a.bytes.size(); ++i)
      if (std::abs(int(a.bytes[i]) - int(b.bytes[i])) > 2)
        ++different;
    std::printf("Producer image mismatched channels: %u\n", different);
    require(different == 0, "Producer colouring parity failed");
    if (frame == 6) {
      const auto published_texture = actual.texture();
      const std::vector<uint8_t> published_bytes(b.bytes.begin(), b.bytes.end());
      // Abandon two successive encodes. Neither may rotate a published target
      // back into the producer, and a subsequent valid frame must still work.
      for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        bool callback_reached = false, rejected = false;
        try {
          (void)render_terrain_frame(trace, nullptr, actual, settings, [&](id<MTLCommandBuffer>) {
            callback_reached = true;
            throw std::runtime_error("Injected producer failure");
          });
        } catch (const std::runtime_error &) {
          rejected = true;
        }
        require(callback_reached && rejected, "Failed producer was not exercised");
        require(
            actual.texture() == published_texture,
            "Failed producer rotated the published target"
        );
        const auto retained = actual.readback(timer);
        require(
            std::equal(published_bytes.begin(), published_bytes.end(), retained.bytes.begin()),
            "Failed producer changed the completed image"
        );
      }
      const auto recovered = render_terrain_frame(trace, nullptr, actual, settings);
      require(
          recovered.producer_submissions == 1U && !recovered.streamed,
          "Producer did not recover after an abandoned frame"
      );
    }
  }
  trace.trace(field);
  bool rejected = false;
  try {
    (void)trace.shadow_visibility();
  } catch (const std::logic_error &) {
    rejected = true;
  }
  require(rejected, "Primary update exposed stale resident shadows");
}

// Exercise the viewer's actual GPU ray/LOD -> complete terrain/shadows ->
// colour -> MetalFX chain, including native/reduced transitions and repair.
void check_metalfx_producer(const std::filesystem::path &directory) {
  using namespace panorama::app;
  RaytraceConfig config{directory, {2600045, 1199945, 1120}, 480, 0, 16384, 2, true, true, false};
  config.lod_scale = 1.5F;
  auto output = camera_request({257, 129}, false);
  TerrainTraceSession reference(config, output, {true, true, true});
  config.raytracer = Raytracer::MetalBvh;
  TerrainTraceSession trace(config, output, {true, true, true});
  MetalFxUpscaler scaler(trace.device(), MTLPixelFormatBGRA8Unorm);
  MetalFxUpscaler expected_scaler(reference.device(), MTLPixelFormatBGRA8Unorm);
  if (!scaler.supported()) {
    std::puts("MetalFX terrain producer skipped: unsupported device");
    return;
  }
  const GpuPresentationRequirements products{false, false, false, true, true, false};
  GpuImageRenderer actual(
      trace.device(),
      trace.command_queue(),
      trace.library(),
      output.image,
      products,
      MTLPixelFormatBGRA8Unorm
  );
  GpuImageRenderer expected(
      reference.device(),
      reference.command_queue(),
      reference.library(),
      output.image,
      products,
      MTLPixelFormatBGRA8Unorm
  );
  const auto read = [&](id<MTLTexture> texture) {
    const NSUInteger row = (texture.width * 4 + 255) & ~NSUInteger(255);
    id<MTLBuffer> buffer = [trace.device() newBufferWithLength:row * texture.height
                                                       options:MTLResourceStorageModeShared];
    auto command = [trace.command_queue() commandBuffer];
    auto blit = [command blitCommandEncoder];
    [blit copyFromTexture:texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                        toBuffer:buffer
               destinationOffset:0
          destinationBytesPerRow:row
        destinationBytesPerImage:row * texture.height];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "MetalFX producer readback failed");
    std::vector<uint8_t> bytes(texture.width * texture.height * 4);
    for (NSUInteger y = 0; y < texture.height; ++y)
      std::memcpy(
          bytes.data() + y * texture.width * 4,
          static_cast<const uint8_t *>(buffer.contents) + y * row,
          texture.width * 4
      );
    return bytes;
  };
  TerrainPresentationSettings settings{};
  settings.colour_range = {0, 480};
  settings.appearance.colour_source = TerrainColourSource::Distance;
  settings.appearance.ambient_light = 0.3F;
  settings.appearance.sun_azimuth = 2.1;
  settings.appearance.sun_elevation = 0.35;
  id<MTLTexture> published = nil;
  std::vector<uint8_t> published_bytes;
  const MetalFxPreset presets[] = {MetalFxPreset::Off,
                                   MetalFxPreset::Quality,
                                   MetalFxPreset::Balanced,
                                   MetalFxPreset::Performance,
                                   MetalFxPreset::Off,
                                   MetalFxPreset::Performance,
                                   MetalFxPreset::Performance,
                                   MetalFxPreset::Performance};
  uint64_t stable_lod_plan_changes = 0U;
  for (size_t frame = 0; frame < std::size(presets); ++frame) {
    settings.appearance.raytraced_shadows = frame != 0;
    settings.appearance.feature_outlines = frame == 3;
    const auto resolution =
        metalfx_resolution(output.image, {MetalFxActivation::Always, presets[frame], false}, true);
    const auto field = metalfx_ray_request(output, resolution.trace);
    if (frame == 3) {
      require(
          trace.relocate_observer({2599965, 1199975, 1120}) &&
              reference.relocate_observer({2599965, 1199975, 1120}),
          "MetalFX observer relocation failed"
      );
    }
    if (resolution.enabled) {
      require(
          scaler.configure(field.image, output.image) &&
              expected_scaler.configure(field.image, output.image),
          "MetalFX configure failed"
      );
      actual.set_output_texture_usage(scaler.input_texture_usage());
      expected.set_output_texture_usage(expected_scaler.input_texture_usage());
      scaler.begin_frame();
      expected_scaler.begin_frame();
    }
    const bool appearance_only = frame == 7;
    const float lod_footprint_scale = float(field.image.height) / float(output.image.height);
    uint32_t callbacks = 0;
    id<MTLTexture> pending = resolution.enabled ? scaler.texture() : nil;
    const auto timing = render_terrain_frame(
        trace,
        appearance_only ? nullptr : &field,
        actual,
        settings,
        [&](id<MTLCommandBuffer> command) {
          ++callbacks;
          if (resolution.enabled) {
            scaler.encode(command, actual.texture());
            require(scaler.texture() == pending, "Terrain repair rotated MetalFX output");
          }
        },
        lod_footprint_scale
    );
    require(callbacks == timing.producer_submissions, "Missing MetalFX repair encode");
    (void)render_terrain_frame(
        reference,
        appearance_only ? nullptr : &field,
        expected,
        settings,
        [&](id<MTLCommandBuffer> command) {
          if (resolution.enabled)
            expected_scaler.encode(command, expected.texture());
        },
        lod_footprint_scale
    );
    if (!appearance_only) {
      const uint64_t changes = trace.bvh_statistics().lod_plan_changes;
      if (frame == 0)
        stable_lod_plan_changes = changes;
      else
        require(
            changes == stable_lod_plan_changes,
            "MetalFX internal resolution changed the terrain LOD plan"
        );
    }
    if (published != nil)
      require(read(published) == published_bytes, "Producer overwrote the displayed frame");
    published = resolution.enabled ? scaler.texture() : actual.texture();
    require(
        published.width == output.image.width && published.height == output.image.height,
        "MetalFX producer published incorrect output dimensions"
    );
    published_bytes = read(published);
    const auto expected_bytes =
        read(resolution.enabled ? expected_scaler.texture() : expected.texture());
    require(published_bytes.size() == expected_bytes.size(), "MetalFX output size mismatch");
    for (size_t i = 0; i < published_bytes.size(); ++i)
      require(
          std::abs(int(published_bytes[i]) - int(expected_bytes[i])) <= 3,
          "MetalFX terrain/shadow output differs from complete software reference"
      );
    if (frame >= 6)
      require(!timing.streamed, "Warmed MetalFX view unexpectedly repaired terrain");
    std::printf(
        "MetalFX producer frame %zu: %ux%u -> %ux%u, submits=%u, streamed=%d\n",
        frame,
        field.image.width,
        field.image.height,
        output.image.width,
        output.image.height,
        timing.producer_submissions,
        timing.streamed
    );
  }
}

void check_shadow_reuse(const std::filesystem::path &directory, bool bilinear, bool bounded) {
  RaytraceConfig
      config{directory, {2600045, 1199945, 1120}, 480, 0, 16384, 2, true, bilinear, false};
  auto camera = camera_request({129, 65});
  auto field = camera_field(camera.image, camera.projection);
  TerrainTraceSession reference(config, field, {true, true, true});
  config.raytracer = Raytracer::MetalBvh;
  if (bounded) {
    auto *geometry = [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    geometry.boundingBoxCount = 16;
    geometry.boundingBoxStride = 24;
    geometry.opaque = NO;
    geometry.allowDuplicateIntersectionFunctionInvocation = NO;
    auto *descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = @[ geometry ];
    const auto sizes = [reference.device() accelerationStructureSizesWithDescriptor:descriptor];
    const bool quantized =
        read_metal_tile_header(
            TerrainCatalogue::discover(directory, config.observer, config.max_distance, 0)
                .origin()
                .path
        )
            .sample_type == MetalTileSampleType::Uint16Decimeters;
    config.bvh_cache_size_bytes =
        17U * 17U * (quantized ? 2U : 4U) + 16U * (sizeof(BvhBlock) + sizeof(BvhBounds)) +
        sizeof(BvhAffinePatch) + sizeof(BvhTile) + sizeof(uint64_t) +
        2U * sizes.accelerationStructureSize + sizes.buildScratchBufferSize;
  }
  TerrainTraceSession trace(config, camera, {true, true, true});
  GpuImageRenderer image(
      trace.device(),
      trace.command_queue(),
      trace.library(),
      camera.image,
      {false, false, false, true, true, true}
  );
  TerrainPresentationSettings settings{};
  settings.appearance.raytraced_shadows = false;
  settings.appearance.sun_azimuth = 2.1;
  settings.colour_range = {0, 480};
  settings.appearance.sun_elevation = 0.35;
  (void)render_terrain_frame(trace, &camera, image, settings, {});
  require(trace.bvh_statistics().shadow_tiles_built == 0, "Disabled shadows loaded BVH casters");
  settings.appearance.raytraced_shadows = true;
  for (uint32_t frame = 0; frame < 8; ++frame) {
    // Odd frames repeat a view exactly; even frames exercise invalidation.
    if (frame == 2) {
      config.observer.easting -= 1;
      require(
          reference.relocate_observer(config.observer) && trace.relocate_observer(config.observer),
          "Shadow reuse relocation failed"
      );
      std::get<CameraProjection>(camera.projection).orientation.heading += 0.02;
    }
    if (frame == 4) {
      settings.appearance.sun_azimuth = -1.2;
      settings.appearance.sun_elevation = 0.15;
    }
    if (frame == 6) {
      reference.set_lod_scale(3);
      trace.set_lod_scale(3);
    }
    field = camera_field(camera.image, camera.projection);
    reference.trace(field);
    reference.trace_shadows(settings.appearance.sun_azimuth, settings.appearance.sun_elevation);
    const auto before = trace.bvh_statistics();
    const auto io_before = trace.tile_statistics().bytes_loaded_with_metal_io;
    const auto timing = render_terrain_frame(trace, &camera, image, settings, {});
    const auto after = trace.bvh_statistics();
    require(
        std::memcmp(
            reference.shadow_visibility().contents,
            trace.shadow_visibility().contents,
            pixel_count(field.image)
        ) == 0,
        "Cached shadow repair changed visibility"
    );
    require(
        after.peak_bytes <= config.bvh_cache_size_bytes &&
            after.resident_bytes <= config.bvh_cache_size_bytes,
        "Shadow reuse exceeded BVH cache budget"
    );
    if (!bounded && frame % 2 == 1) {
      require(
          !timing.streamed && timing.producer_submissions == 1,
          "Repeated shadows required repair after warming"
      );
      require(
          after.builds == before.builds &&
              trace.tile_statistics().bytes_loaded_with_metal_io == io_before,
          "Repeated shadows rebuilt or reloaded terrain"
      );
    }
  }
  const auto stats = trace.bvh_statistics();
  std::printf(
      "Shadow reuse bilinear=%d bounded=%d: caster builds=%llu, capacity fallbacks=%llu\n",
      bilinear,
      bounded,
      static_cast<unsigned long long>(stats.shadow_tiles_built),
      static_cast<unsigned long long>(stats.shadow_cache_fallbacks)
  );
  if (bounded)
    require(stats.shadow_cache_fallbacks > 0, "Small cache did not exercise exact shadow fallback");
  else {
    require(stats.shadow_tiles_built > 0, "Fixture did not request off-screen shadow terrain");
    require(stats.shadow_cache_fallbacks == 0, "Roomy shadow cache unexpectedly fell back");
  }
}

void check_streaming(const std::filesystem::path &directory) {
  const auto field = angular_field({129, 65}, {0, 2 * std::numbers::pi, -1.3, 0.1});
  RaytraceConfig
      config{directory, {2600045, 1199945, 1120}, 600000, 0, 16384, 2, true, true, false};
  config.use_tile_bvh = false;
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
  config.bvh_cache_size_bytes = 17U * 17U * 2U + 16U * (sizeof(BvhBlock) + sizeof(BvhBounds)) +
                                sizeof(BvhAffinePatch) + sizeof(BvhTile) + sizeof(uint64_t) +
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
      if (argc >= 2 && std::string_view(argv[1]) == "--benchmark-camera") {
        benchmark_camera(argc, argv);
        return EXIT_SUCCESS;
      }
      if (argc >= 2 && std::string_view(argv[1]) == "--benchmark") {
        benchmark(argc, argv);
        return EXIT_SUCCESS;
      }
      if (argc == 4 && std::string_view(argv[1]) == "--datasets") {
        check_prepared_dataset_stack(argv[2], argv[3]);
        return EXIT_SUCCESS;
      }
      const bool tile_selection = argc == 2 && std::string_view(argv[1]) == "--tile-selection";
      const bool camera = argc == 2 && std::string_view(argv[1]) == "--camera";
      const bool metalfx = argc == 2 && std::string_view(argv[1]) == "--metalfx";
      const bool producer = argc == 2 && std::string_view(argv[1]) == "--producer";
      const bool shadow_float = argc == 2 && std::string_view(argv[1]) == "--shadow-reuse-float";
      const bool shadow_quantized =
          argc == 2 && std::string_view(argv[1]) == "--shadow-reuse-quantized";
      const bool edge_cases = argc == 2 && std::string_view(argv[1]) == "--edge-cases";
      const bool streaming = argc == 2 && std::string_view(argv[1]) == "--streaming";
      const bool mixed_coverage = argc == 2 && std::string_view(argv[1]) == "--mixed-coverage";
      if (argc >= 2 && !edge_cases && !streaming && !producer && !tile_selection && !camera &&
          !shadow_float && !shadow_quantized && !metalfx && !mixed_coverage) {
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
        const auto field = angular_field({512, 128}, {0, 2 * std::numbers::pi, -0.6, 0.15});
        config.use_tile_bvh = false;
        TerrainTraceSession software(config, field, {true, true, true});
        config.raytracer = Raytracer::MetalBvh;
        TerrainTraceSession hardware(config, field, {true, true, true});
        const auto catalogue =
            TerrainCatalogue::discover(config.tile_dir, config.observer, config.max_distance, 0U);
        const auto header = read_metal_tile_header(catalogue.origin().path);
        {
          config.raytracer = Raytracer::Software;
          config.use_tile_bvh = true;
          TerrainTraceSession shared(config, field, {true, true, true});
          compare(software, shared, field, 0.15F, &header);
          compare(software, shared, field, 0.15F, &header);
        }
        compare(software, hardware, field, 0.15F, &header);
        compare(software, hardware, field, 0.15F, &header);
        require(
            hardware.bvh_statistics().peak_bytes <= config.bvh_cache_size_bytes,
            "Real BVH cache exceeded budget"
        );
        const auto *candidates = static_cast<const float *>(hardware.num_steps().contents);
        double total_candidates = 0.0;
        float maximum_candidates = 0.0F;
        for (size_t i = 0; i < pixel_count(field.image); ++i) {
          total_candidates += candidates[i];
          maximum_candidates = std::max(maximum_candidates, candidates[i]);
        }
        std::printf(
            "BVH candidates: mean %.3f, maximum %.0f per ray.\n",
            total_candidates / double(pixel_count(field.image)),
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
        write_fixture(root / "overlap", true, 10.0, 2056U, 2600000.0, 1200000.0, 100.0);
        write_fixture(root / "partial-overlap", true, 10.0, 2056U, 2600080.0, 1200000.0, 100.0);
        write_fixture(root / "distant", true, 1000.0);
        check_dataset_foundation(root);
        if (shadow_float || shadow_quantized) {
          for (bool bilinear : {false, true}) {
            for (bool bounded : {false, true}) {
              @autoreleasepool {
                check_shadow_reuse(
                    root / (shadow_float ? "float" : "quantized"),
                    bilinear,
                    bounded
                );
              }
            }
          }
        } else if (metalfx) {
          check_metalfx_producer(root / "quantized");
        } else if (camera) {
          @autoreleasepool {
            check_gpu_projection(root / "quantized");
          }
          for (Raytracer backend : {Raytracer::Software, Raytracer::MetalBvh}) {
            @autoreleasepool {
              check_session_replacement(root / "quantized", backend, true);
            }
          }
          for (bool bilinear : {false, true}) {
            @autoreleasepool {
              check_producer(root / "quantized", bilinear, false, true);
            }
          }
          @autoreleasepool {
            check_producer(root / "quantized", true, true, true);
          }
        } else if (tile_selection) {
          @autoreleasepool {
            check_tile_selection(root / "float", false, 10.0);
          }
          @autoreleasepool {
            check_tile_selection(root / "quantized", true, 10.0);
          }
          @autoreleasepool {
            check_tile_selection(root / "distant", true, 1000.0);
          }
        } else if (producer) {
          for (Raytracer backend : {Raytracer::Software, Raytracer::MetalBvh}) {
            @autoreleasepool {
              check_session_replacement(root / "quantized", backend);
            }
          }
          for (bool bilinear : {false, true}) {
            @autoreleasepool {
              check_producer(root / "quantized", bilinear);
            }
          }
          @autoreleasepool {
            check_producer(root / "quantized", true, true);
          }
        } else if (mixed_coverage) {
          check_misaligned_coverage(root);
        } else if (streaming) {
          check_streaming(root / "quantized");
        } else if (edge_cases) {
          @autoreleasepool {
            check_scene_misses(root / "quantized");
          }
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
          check_mixed_priority(root);
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
