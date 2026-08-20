#include "raytrace_setup.h"

#include "arguments.h"
#include "ray_projection_arguments.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;

/// Runtime-selectable settings for one projected terrain dataset.
struct EntrypointSettings {
  std::filesystem::path tile_dir = "data/swissalti3d-10-level-0";
  uint64_t tile_cache_size_bytes = 128ULL * kBytesPerMiB;
  uint32_t max_tile_preparation_workers = 8U;
  uint32_t max_tile_count = 0U;
  float max_distance = 600'000.0F;
  bool retain_quantized = false;
  bool compute_normals = true;
  double easting = 2623452.4;
  double northing = 1100502.2;
  double elevation = 3415.0;
  panorama::RayProjectionArguments projection;
};

/// Print the command-line options accepted by the raytracing executable.
void print_usage(const char *program) {
  std::printf(
      "usage: %s [options]\n"
      "  --tile-dir DIR        prepared level-0 tile directory\n"
      "                        (default: data/swissalti3d-10-level-0)\n"
      "  --tile-cache-mib N    resident terrain-cache budget in MiB (default: 128)\n"
      "  --workers N           preparation workers; 0 uses all hardware threads (default: 8)\n"
      "  --max-tiles N         limit available source tiles; 0 is unlimited (default: 0)\n"
      "  --max-distance M      horizontal trace range in metres (default: 600000)\n"
      "\n",
      program
  );
  panorama::print_ray_projection_usage();
  std::printf(
      "General output and observer options:\n"
      "  --retain-quantized    keep uint16 terrain quantized in the GPU atlas\n"
      "  --no-normals          skip collision-normal computation and normals.png\n"
      "  --easting M           observer easting in the tile CRS (default: 2623452.4)\n"
      "  --northing M          observer northing in the tile CRS (default: 1100502.2)\n"
      "  --elevation M         observer elevation in metres (default: 3415.0)\n"
      "  --help                show this message\n"
  );
}

/// Parse tracing settings and one of the supported output projections.
[[nodiscard]] EntrypointSettings parse_arguments(int argc, const char *argv[]) {
  EntrypointSettings settings;
  for (int index = 1; index < argc; index++) {
    const std::string_view option = argv[index];
    if (option == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (option == "--retain-quantized") {
      settings.retain_quantized = true;
      continue;
    }
    if (option == "--no-normals") {
      settings.compute_normals = false;
      continue;
    }

    const std::string_view value = panorama::arguments::option_value(argc, argv, index, option);
    if (option == "--tile-dir") {
      settings.tile_dir = value;
    } else if (option == "--tile-cache-mib") {
      const uint64_t mebibytes = panorama::arguments::parse_uint64(value, option);
      if (mebibytes == 0U || mebibytes > std::numeric_limits<uint64_t>::max() / kBytesPerMiB) {
        throw std::out_of_range("Tile-cache size is outside byte range");
      }
      settings.tile_cache_size_bytes = mebibytes * kBytesPerMiB;
    } else if (option == "--workers") {
      settings.max_tile_preparation_workers =
          panorama::arguments::parse_uint32(value, option, true);
    } else if (option == "--max-tiles") {
      settings.max_tile_count = panorama::arguments::parse_uint32(value, option, true);
    } else if (option == "--max-distance") {
      const double parsed = panorama::arguments::parse_finite_double(value, option);
      if (parsed <= 0.0 || parsed > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::out_of_range("Maximum distance must be a positive float32 value");
      }
      settings.max_distance = static_cast<float>(parsed);
    } else if (option == "--easting") {
      settings.easting = panorama::arguments::parse_finite_double(value, option);
    } else if (option == "--northing") {
      settings.northing = panorama::arguments::parse_finite_double(value, option);
    } else if (option == "--elevation") {
      settings.elevation = panorama::arguments::parse_finite_double(value, option);
    } else if (!settings.projection.parse_option(option, value)) {
      throw std::invalid_argument("Unknown option: " + std::string(option));
    }
  }
  settings.projection.validate();
  return settings;
}

} // namespace

/// Generate the selected output rays and trace one prepared terrain dataset.
int main(int argc, const char *argv[]) {
  try {
    const EntrypointSettings settings = parse_arguments(argc, argv);

    const panorama::RaytraceConfig config = {
        settings.tile_dir,
        {settings.easting, settings.northing, settings.elevation},
        settings.max_distance,
        settings.max_tile_count,
        settings.tile_cache_size_bytes,
        settings.max_tile_preparation_workers,
        settings.retain_quantized,
        settings.compute_normals,
    };
    const panorama::RayField rays = settings.projection.make_ray_field();
    std::printf(
        "Tracing terrain in %s from projected coordinate (%.3f, %.3f, %.1f).\n",
        settings.tile_dir.c_str(),
        settings.easting,
        settings.northing,
        settings.elevation
    );
    // Echo every benchmark-relevant setting so redirected timing logs remain
    // self-describing when several command-line configurations are compared.
    std::printf(
        "Settings: cache %.0f MiB, workers %u, max tiles %u, ",
        static_cast<double>(settings.tile_cache_size_bytes) / static_cast<double>(kBytesPerMiB),
        settings.max_tile_preparation_workers,
        settings.max_tile_count
    );
    settings.projection.print_settings();
    std::printf(
        ", range %.0f m, curvature %.4f m/mile^2, quantized atlas %s, normals %s.\n",
        settings.max_distance,
        panorama::kCurvatureCoefficient * 1609.344 * 1609.344,
        settings.retain_quantized ? "retained" : "disabled",
        settings.compute_normals ? "enabled" : "disabled"
    );
    panorama::raytrace_tiled_heightmap(config, rays);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
