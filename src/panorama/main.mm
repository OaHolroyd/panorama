#include "terrain_renderer.h"

#include "arguments.h"
#include "ray_projection_arguments.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
constexpr double kDegreesToRadians = std::numbers::pi / 180.0;

/// Runtime-selectable settings for one projected terrain dataset.
struct EntrypointSettings {
  std::filesystem::path tile_dir = "data/swissalti3d-10-level-0";
  uint64_t tile_cache_size_bytes = 128ULL * kBytesPerMiB;
  uint32_t max_tile_preparation_workers = 8U;
  uint32_t max_tile_count = 0U;
  float max_distance = 600'000.0F;
  bool retain_quantized = false;
  bool compute_elevations = true;
  bool compute_normals = true;
  bool write_diagnostics = true;
  bool write_synthetic = false;
  bool synthetic_setting_seen = false;
  panorama::SyntheticRenderOptions synthetic = {
      225.0 * kDegreesToRadians,
      35.0 * kDegreesToRadians,
      0.28F,
  };
  double easting = 2623452.4;
  double northing = 1100502.2;
  double elevation = 3415.0;
  panorama::RayProjectionArguments projection;
};

/// Validate combinations whose meaning spans more than one CLI switch.
void validate_output_settings(const EntrypointSettings &settings) {
  if (!settings.write_diagnostics && !settings.write_synthetic) {
    throw std::invalid_argument("At least one output type must be enabled");
  }
  // Rendering requires collision gradients, whereas mandatory distances and
  // optional elevation diagnostics remain usable without them.
  if (settings.write_synthetic && !settings.compute_normals) {
    throw std::invalid_argument("--synthetic-output cannot be combined with --no-normals");
  }
  // Treat lighting controls as configuration for an explicitly selected
  // product; silently creating another output would make scripted runs unclear.
  if (settings.synthetic_setting_seen && !settings.write_synthetic) {
    throw std::invalid_argument("Synthetic image options require --synthetic-output");
  }
}

/// Print the command-line options accepted by the panorama executable.
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
      "Output options:\n"
      "  --diagnostic-output   write diagnostic field PNGs (default: enabled)\n"
      "  --no-diagnostic-output\n"
      "                        skip diagnostic field PNGs\n"
      "  --synthetic-output    write a shaded synthetic.png (default: disabled)\n"
      "  --retain-quantized    keep uint16 terrain quantized in the GPU atlas\n"
      "  --no-elevations       skip collision-elevation storage and elevations.png\n"
      "  --no-normals          skip collision-normal computation and normals.png\n"
      "\n"
      "Synthetic image options (angles are degrees):\n"
      "  --sun-azimuth D       clockwise from grid north (default: 225)\n"
      "  --sun-elevation D     above the horizon (default: 35)\n"
      "  --ambient-light V     direction-independent light, 0 to 1 (default: 0.28)\n"
      "\n"
      "Observer options:\n"
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
    if (option == "--no-elevations") {
      settings.compute_elevations = false;
      continue;
    }
    if (option == "--diagnostic-output") {
      settings.write_diagnostics = true;
      continue;
    }
    if (option == "--no-diagnostic-output") {
      settings.write_diagnostics = false;
      continue;
    }
    if (option == "--synthetic-output") {
      settings.write_synthetic = true;
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
    } else if (option == "--sun-azimuth") {
      settings.synthetic.sun_azimuth =
          panorama::arguments::parse_finite_double(value, option) * kDegreesToRadians;
      settings.synthetic_setting_seen = true;
    } else if (option == "--sun-elevation") {
      const double degrees = panorama::arguments::parse_finite_double(value, option);
      if (degrees < -90.0 || degrees > 90.0) {
        throw std::out_of_range("Sun elevation must be between -90 and 90 degrees");
      }
      settings.synthetic.sun_elevation = degrees * kDegreesToRadians;
      settings.synthetic_setting_seen = true;
    } else if (option == "--ambient-light") {
      const double parsed = panorama::arguments::parse_finite_double(value, option);
      if (parsed < 0.0 || parsed > 1.0) {
        throw std::out_of_range("Ambient light must be between zero and one");
      }
      settings.synthetic.ambient_light = static_cast<float>(parsed);
      settings.synthetic_setting_seen = true;
    } else if (!settings.projection.parse_option(option, value)) {
      throw std::invalid_argument("Unknown option: " + std::string(option));
    }
  }
  settings.projection.validate();
  validate_output_settings(settings);
  return settings;
}

} // namespace

/// Generate the selected rays, trace terrain, and render the requested products.
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
        settings.compute_elevations,
        settings.compute_normals,
    };
    const panorama::RayField rays = settings.projection.make_ray_field();
    const panorama::TerrainRenderOutputs outputs = {
        settings.write_diagnostics,
        settings.write_synthetic,
        settings.synthetic,
    };
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
        ", range %.0f m, curvature %.4f m/mile^2, quantized atlas %s, "
        "elevations %s, normals %s, outputs %s.\n",
        settings.max_distance,
        panorama::kCurvatureCoefficient * 1609.344 * 1609.344,
        settings.retain_quantized ? "retained" : "disabled",
        settings.compute_elevations ? "enabled" : "disabled",
        settings.compute_normals ? "enabled" : "disabled",
        settings.write_diagnostics
            ? (settings.write_synthetic ? "diagnostic+synthetic" : "diagnostic")
            : "synthetic"
    );
    if (settings.write_synthetic) {
      std::printf(
          "Synthetic image: sun azimuth %.3f deg, elevation %.3f deg, ambient %.3f.\n",
          settings.synthetic.sun_azimuth / kDegreesToRadians,
          settings.synthetic.sun_elevation / kDegreesToRadians,
          settings.synthetic.ambient_light
      );
    }
    panorama::render_terrain(config, rays, outputs);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
