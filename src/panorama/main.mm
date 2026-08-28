#include "terrain_renderer.h"

#include "arguments.h"
#include "ray_projection_arguments.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
constexpr double kDegreesToRadians = std::numbers::pi / 180.0;

/// Runtime-selectable settings for one panorama invocation.
struct EntrypointSettings {
  std::filesystem::path tile_dir = "data/swissalti3d-10-level-0-u16-none";
  uint64_t tile_cache_size_bytes = 128ULL * kBytesPerMiB;
  uint32_t max_tile_preparation_workers = 8U;
  uint32_t max_tile_count = 0U;
  float max_distance = 600'000.0F;
  float lod_scale = 0.0F;
  bool discard_quantized = false;
  bool elevations_enabled = true;
  bool normals_enabled = true;
  bool write_diagnostics = true;
  bool write_synthetic = false;
  bool synthetic_setting_seen = false;
  bool colourmap_setting_seen = false;
  bool colour_scale_setting_seen = false;
  bool colour_range_setting_seen = false;
  float colour_minimum = 0.0F;
  std::optional<float> colour_maximum;
  panorama::SyntheticRenderOptions synthetic = {
      .sun_azimuth = 225.0 * kDegreesToRadians,
      .sun_elevation = 35.0 * kDegreesToRadians,
      .ambient_light = 0.28F,
      .ambient_detail = 0.65F,
      .diffusivity = 1.0F,
  };
  double easting = 2623452.4;
  double northing = 1100502.2;
  double elevation = 3415.0;
  panorama::RayProjectionArguments projection;
};

/// Parse a finite command-line value representable as float32.
[[nodiscard]] float parse_float32(std::string_view value, std::string_view option) {
  const double parsed = panorama::arguments::parse_finite_double(value, option);
  if (parsed < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      parsed > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::out_of_range(std::string(option) + " is outside the float32 range");
  }
  return static_cast<float>(parsed);
}

/// Resolve the range after parsing so its default follows `--max-distance`
/// regardless of command-line option order.
[[nodiscard]] panorama::ScalarColourRange scalar_colour_range(const EntrypointSettings &settings) {
  return {settings.colour_minimum, settings.colour_maximum.value_or(settings.max_distance)};
}

constexpr std::array kTerrainColours = {
    std::pair{"white", panorama::TerrainColourSource::White},
    std::pair{"distance", panorama::TerrainColourSource::Distance},
    std::pair{"elevation", panorama::TerrainColourSource::Elevation},
};

constexpr std::array kColourmaps = {
    std::pair{"viridis", panorama::PresetColourmap::Viridis},
    std::pair{"plasma", panorama::PresetColourmap::Plasma},
    std::pair{"inferno", panorama::PresetColourmap::Inferno},
    std::pair{"magma", panorama::PresetColourmap::Magma},
    std::pair{"cividis", panorama::PresetColourmap::Cividis},
    std::pair{"turbo", panorama::PresetColourmap::Turbo},
    std::pair{"viewfinder", panorama::PresetColourmap::Viewfinder},
};

constexpr std::array kColourScales = {
    std::pair{"linear", panorama::ScalarColourScale::Linear},
    std::pair{"logarithmic", panorama::ScalarColourScale::Logarithmic},
    std::pair{"square-root", panorama::ScalarColourScale::SquareRoot},
    std::pair{"quadratic", panorama::ScalarColourScale::Quadratic},
};

/// Parse and print small named enums from one canonical choice table.
template <typename Enum, size_t Count>
[[nodiscard]] Enum parse_choice(
    std::string_view value,
    const std::array<std::pair<const char *, Enum>, Count> &choices,
    std::string_view description,
    std::string_view expected
) {
  for (const auto &[name, choice] : choices) {
    if (value == name) {
      return choice;
    }
  }
  throw std::invalid_argument(
      "Invalid " + std::string(description) + ": " + std::string(value) + " (expected " +
      std::string(expected) + ")"
  );
}

template <typename Enum, size_t Count>
[[nodiscard]] const char *
choice_name(Enum value, const std::array<std::pair<const char *, Enum>, Count> &choices) {
  for (const auto &[name, choice] : choices) {
    if (value == choice) {
      return name;
    }
  }
  return "unknown";
}

/// Validate combinations whose meaning spans more than one CLI switch.
void validate_output_settings(const EntrypointSettings &settings) {
  if (!settings.write_diagnostics && !settings.write_synthetic) {
    throw std::invalid_argument("At least one output type must be enabled");
  }
  // Rendering requires collision gradients, whereas mandatory distances and
  // optional elevation diagnostics remain usable without them.
  if (settings.write_synthetic && !settings.normals_enabled) {
    throw std::invalid_argument("--synthetic-output cannot be combined with --no-normals");
  }
  if (settings.write_synthetic &&
      settings.synthetic.colour_source == panorama::TerrainColourSource::Elevation &&
      !settings.elevations_enabled) {
    throw std::invalid_argument(
        "--terrain-colour elevation cannot be combined with --no-elevations"
    );
  }
  // Treat lighting controls as configuration for an explicitly selected
  // product; silently creating another output would make scripted runs unclear.
  if (settings.synthetic_setting_seen && !settings.write_synthetic) {
    throw std::invalid_argument("Synthetic image options require --synthetic-output");
  }
  if (settings.colourmap_setting_seen &&
      settings.synthetic.colour_source == panorama::TerrainColourSource::White) {
    throw std::invalid_argument("--colourmap requires distance or elevation terrain colour");
  }
  if (settings.colour_scale_setting_seen &&
      settings.synthetic.colour_source == panorama::TerrainColourSource::White) {
    throw std::invalid_argument("--colour-scale requires distance or elevation terrain colour");
  }
  const panorama::ScalarColourRange range = scalar_colour_range(settings);
  if (!std::isfinite(range.minimum) || !std::isfinite(range.maximum) ||
      range.maximum <= range.minimum) {
    throw std::invalid_argument("--colour-max must be greater than --colour-min");
  }
  const bool scalar_output = settings.write_diagnostics ||
                             (settings.write_synthetic && settings.synthetic.colour_source !=
                                                              panorama::TerrainColourSource::White);
  if (settings.colour_range_setting_seen && !scalar_output) {
    throw std::invalid_argument(
        "Colour range options require diagnostics or distance/elevation terrain colour"
    );
  }
}

/// Print the command-line options accepted by the panorama executable.
void print_usage(const char *program) {
  std::printf(
      "usage: %s [options]\n"
      "\n"
      "Trace a prepared DTM on the GPU and write panoramic or camera-projected\n"
      "distance, elevation, normal, and optionally shaded terrain PNGs. Output\n"
      "files are written to the current directory.\n"
      "\n"
      "Terrain options:\n"
      "  --tile-dir DIR        prepared level-0 tile directory\n"
      "                        (default: data/swissalti3d-10-level-0)\n"
      "  --tile-cache-mib N    resident terrain-cache budget in MiB (default: 128)\n"
      "  --workers N           preparation workers; 0 uses all hardware threads (default: 8)\n"
      "  --max-tiles N         limit available source tiles; 0 is unlimited (default: 0)\n"
      "  --max-distance M      horizontal trace range in metres (default: 600000)\n"
      "  --lod-scale V         terrain cell footprint multiplier; 0 keeps full detail\n"
      "                        (default: 0)\n"
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
      "  --discard-quantized   expand uint16 terrain to Float32 in the GPU atlas\n"
      "                        (default: retain uint16)\n"
      "  --no-elevations       skip collision-elevation storage and elevations.png\n"
      "  --no-normals          skip collision-normal computation and normals.png\n"
      "\n"
      "Scalar colour options:\n"
      "  --colour-min V        value mapped to the first palette colour (default: 0)\n"
      "  --colour-max V        value mapped to the last palette colour\n"
      "                        (default: --max-distance)\n"
      "\n"
      "Synthetic image options (angles are degrees):\n"
      "  --sun-azimuth D       clockwise from grid north (default: 225)\n"
      "  --sun-elevation D     above the horizon (default: 35)\n"
      "  --ambient-light V     diffuse sky-light strength, 0 to 1 (default: 0.28)\n"
      "  --ambient-detail V    normal-dependent five-lobe skylight, 0 to 1 (default: 0.65)\n"
      "  --diffusivity V       directional diffuse-light strength, 0 to 1 (default: 1)\n"
      "  --raytraced-shadows   cast one hard terrain-shadow ray per lit collision\n"
      "  --feature-outlines    draw multiscale black surface-separation lines\n"
      "  --outline-detail N    outline detail from 0 to 10 (default: 7)\n"
      "  --terrain-colour MODE white, distance, or elevation (default: white)\n"
      "  --colourmap NAME      viridis, plasma, inferno, magma, cividis, turbo, or viewfinder\n"
      "                        (default: viridis)\n"
      "  --colour-scale NAME   linear, logarithmic, square-root, or quadratic\n"
      "                        (default: linear)\n"
      "\n"
      "Observer options:\n"
      "  --easting M           observer easting in the tile CRS (default: 2623452.4)\n"
      "  --northing M          observer northing in the tile CRS (default: 1100502.2)\n"
      "  --elevation M         observer elevation in metres (default: 3415.0)\n"
      "  --help                show this message\n"
  );
}

/// Parse tracing, projection, and output settings for one invocation.
[[nodiscard]] EntrypointSettings parse_arguments(int argc, const char *argv[]) {
  EntrypointSettings settings;
  for (int index = 1; index < argc; index++) {
    const std::string_view option = argv[index];
    if (option == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (option == "--discard-quantized") {
      settings.discard_quantized = true;
      continue;
    }
    if (option == "--no-normals") {
      settings.normals_enabled = false;
      continue;
    }
    if (option == "--no-elevations") {
      settings.elevations_enabled = false;
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
    if (option == "--feature-outlines") {
      settings.synthetic.feature_outlines = true;
      settings.synthetic_setting_seen = true;
      continue;
    }
    if (option == "--raytraced-shadows") {
      settings.synthetic.raytraced_shadows = true;
      settings.synthetic_setting_seen = true;
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
      const float parsed = parse_float32(value, option);
      if (parsed <= 0.0F) {
        throw std::out_of_range("Maximum distance must be a positive float32 value");
      }
      settings.max_distance = parsed;
    } else if (option == "--lod-scale") {
      const float parsed = parse_float32(value, option);
      if (parsed < 0.0F) {
        throw std::out_of_range("LOD scale must be a nonnegative float32 value");
      }
      settings.lod_scale = parsed;
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
    } else if (option == "--ambient-detail") {
      const double parsed = panorama::arguments::parse_finite_double(value, option);
      if (parsed < 0.0 || parsed > 1.0) {
        throw std::out_of_range("Ambient detail must be between zero and one");
      }
      settings.synthetic.ambient_detail = static_cast<float>(parsed);
      settings.synthetic_setting_seen = true;
    } else if (option == "--diffusivity") {
      const double parsed = panorama::arguments::parse_finite_double(value, option);
      if (parsed < 0.0 || parsed > 1.0) {
        throw std::out_of_range("Diffusivity must be between zero and one");
      }
      settings.synthetic.diffusivity = static_cast<float>(parsed);
      settings.synthetic_setting_seen = true;
    } else if (option == "--outline-detail") {
      const double parsed = panorama::arguments::parse_finite_double(value, option);
      if (parsed < 0.0 || parsed > 10.0) {
        throw std::out_of_range("Outline detail must be between 0 and 10");
      }
      settings.synthetic.feature_outline_detail = static_cast<float>(parsed / 10.0);
      settings.synthetic_setting_seen = true;
    } else if (option == "--colour-min") {
      settings.colour_minimum = parse_float32(value, option);
      settings.colour_range_setting_seen = true;
    } else if (option == "--colour-max") {
      settings.colour_maximum = parse_float32(value, option);
      settings.colour_range_setting_seen = true;
    } else if (option == "--terrain-colour") {
      settings.synthetic.colour_source =
          parse_choice(value, kTerrainColours, "terrain colour", "white, distance, or elevation");
      settings.synthetic_setting_seen = true;
    } else if (option == "--colourmap") {
      settings.synthetic.colourmap = parse_choice(
          value,
          kColourmaps,
          "colourmap",
          "viridis, plasma, inferno, magma, cividis, turbo, or viewfinder"
      );
      settings.synthetic_setting_seen = true;
      settings.colourmap_setting_seen = true;
    } else if (option == "--colour-scale") {
      settings.synthetic.colour_scale = parse_choice(
          value,
          kColourScales,
          "colour scale",
          "linear, logarithmic, square-root, or quadratic"
      );
      settings.synthetic_setting_seen = true;
      settings.colour_scale_setting_seen = true;
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
        !settings.discard_quantized,
        false,
        settings.lod_scale,
    };
    const panorama::RayField rays = settings.projection.make_ray_field();
    const panorama::ScalarColourRange colour_range = scalar_colour_range(settings);
    const panorama::TerrainRenderOutputs outputs = {
        settings.write_diagnostics,
        settings.elevations_enabled,
        settings.normals_enabled,
        settings.write_synthetic,
        settings.synthetic,
        colour_range,
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
        ", range %.0f m, LOD scale %.6g, curvature %.4f m/mile^2, quantized atlas %s, "
        "elevations %s, normals %s, outputs %s.\n",
        settings.max_distance,
        settings.lod_scale,
        panorama::kCurvatureCoefficient * 1609.344 * 1609.344,
        settings.discard_quantized ? "discarded" : "retained when available",
        outputs.requires_elevations() ? "enabled" : "disabled",
        outputs.requires_normals() ? "enabled" : "disabled",
        settings.write_diagnostics
            ? (settings.write_synthetic ? "diagnostic+synthetic" : "diagnostic")
            : "synthetic"
    );
    if (settings.write_synthetic) {
      std::printf(
          "Synthetic image: sun azimuth %.3f deg, elevation %.3f deg, ambient %.3f "
          "(detail %.3f), diffusivity %.3f, terrain colour %s",
          settings.synthetic.sun_azimuth / kDegreesToRadians,
          settings.synthetic.sun_elevation / kDegreesToRadians,
          settings.synthetic.ambient_light,
          settings.synthetic.ambient_detail,
          settings.synthetic.diffusivity,
          choice_name(settings.synthetic.colour_source, kTerrainColours)
      );
      if (settings.synthetic.colour_source != panorama::TerrainColourSource::White) {
        std::printf(
            " (%s, %s scale)",
            choice_name(settings.synthetic.colourmap, kColourmaps),
            choice_name(settings.synthetic.colour_scale, kColourScales)
        );
      }
      if (settings.synthetic.feature_outlines) {
        std::printf(
            ", feature outlines detail %.0f",
            10.0 * settings.synthetic.feature_outline_detail
        );
      }
      if (settings.synthetic.raytraced_shadows) {
        std::printf(", raytraced hard shadows");
      }
      std::printf(".\n");
    }
    if (settings.write_diagnostics ||
        (settings.write_synthetic &&
         settings.synthetic.colour_source != panorama::TerrainColourSource::White)) {
      std::printf(
          "Scalar colour range: %.6g to %.6g.\n",
          colour_range.minimum,
          colour_range.maximum
      );
    }
    panorama::render_terrain(config, rays, outputs);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
