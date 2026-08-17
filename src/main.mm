#include "raytrace_setup.h"

#include <cerrno>
#include <charconv>
#include <cmath>
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

constexpr double kSwissCellSize = 2.0;
constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;

/// Runtime-selectable settings for the fixed Swiss terrain demonstration.
struct EntrypointSettings {
  uint32_t rechunk_level = 10U;
  uint64_t tile_cache_size_bytes = 128ULL * kBytesPerMiB;
  uint32_t max_tile_preparation_workers = 8U;
  uint32_t max_tile_count = 0U;
  uint32_t num_azimuth = 1024U;
  uint32_t num_polar = 512U;
  float max_distance = 600'000.0F;
  double easting = 2623452.4;
  double northing = 1100502.2;
  double elevation = 3415.0;
};

/// Print the command-line options accepted by the temporary demonstration.
void print_usage(const char *program) {
  std::printf(
      "usage: %s [options]\n"
      "  --rechunk-level N     prepared level-0 tile power (default: 10)\n"
      "  --tile-cache-mib N    resident terrain-cache budget in MiB (default: 512)\n"
      "  --workers N           tile load/mipmap workers; 0 uses all hardware threads (default: 8)\n"
      "  --max-tiles N         limit available source tiles; 0 is unlimited (default: 0)\n"
      "  --max-distance M      horizontal trace range in metres (default: 600000)\n"
      "  --azimuth-count N     number of azimuth columns (default: 1024)\n"
      "  --polar-count N       number of polar rays per column (default: 512)\n"
      "  --easting M           observer LV95 easting (default: 2623452.4)\n"
      "  --northing M          observer LV95 northing (default: 1100502.2)\n"
      "  --elevation M         observer elevation in metres (default: 3415.0)\n"
      "  --help                show this message\n",
      program
  );
}

/// Parse one non-negative unsigned 64-bit command-line value.
[[nodiscard]] uint64_t parse_uint64(std::string_view text, std::string_view option) {
  uint64_t value = 0U;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::invalid_argument(
        "Invalid value for " + std::string(option) + ": " + std::string(text)
    );
  }
  return value;
}

/// Parse one finite floating-point command-line value.
[[nodiscard]] double parse_double(std::string_view text, std::string_view option) {
  const std::string copy(text);
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(copy.c_str(), &end);
  if (errno != 0 || end == copy.c_str() || *end != '\0' || !std::isfinite(value)) {
    throw std::invalid_argument(
        "Invalid value for " + std::string(option) + ": " + std::string(text)
    );
  }
  return value;
}

/// Return the argument immediately following one option, or report its absence.
[[nodiscard]] std::string_view
option_value(int argc, const char *argv[], int &index, std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument("Missing value for " + std::string(option));
  }
  ++index;
  return argv[index];
}

/// Parse the command-line settings used to construct `RaytraceConfig`.
[[nodiscard]] EntrypointSettings parse_arguments(int argc, const char *argv[]) {
  EntrypointSettings settings;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    const std::string_view value = option_value(argc, argv, index, option);
    if (option == "--rechunk-level") {
      const uint64_t parsed = parse_uint64(value, option);
      if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Rechunk level is outside uint32 range");
      }
      settings.rechunk_level = static_cast<uint32_t>(parsed);
    } else if (option == "--tile-cache-mib") {
      const uint64_t mebibytes = parse_uint64(value, option);
      if (mebibytes == 0U || mebibytes > std::numeric_limits<uint64_t>::max() / kBytesPerMiB) {
        throw std::out_of_range("Tile-cache size is outside byte range");
      }
      settings.tile_cache_size_bytes = mebibytes * kBytesPerMiB;
    } else if (option == "--workers") {
      const uint64_t parsed = parse_uint64(value, option);
      if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Worker count is outside uint32 range");
      }
      settings.max_tile_preparation_workers = static_cast<uint32_t>(parsed);
    } else if (option == "--max-tiles") {
      const uint64_t parsed = parse_uint64(value, option);
      if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Maximum tile count is outside uint32 range");
      }
      settings.max_tile_count = static_cast<uint32_t>(parsed);
    } else if (option == "--max-distance") {
      const double parsed = parse_double(value, option);
      if (parsed <= 0.0 || parsed > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::out_of_range("Maximum distance must be a positive float32 value");
      }
      settings.max_distance = static_cast<float>(parsed);
    } else if (option == "--azimuth-count") {
      const uint64_t parsed = parse_uint64(value, option);
      if (parsed == 0U || parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Azimuth count must be a positive uint32 value");
      }
      settings.num_azimuth = static_cast<uint32_t>(parsed);
    } else if (option == "--polar-count") {
      const uint64_t parsed = parse_uint64(value, option);
      if (parsed == 0U || parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Polar count must be a positive uint32 value");
      }
      settings.num_polar = static_cast<uint32_t>(parsed);
    } else if (option == "--easting") {
      settings.easting = parse_double(value, option);
    } else if (option == "--northing") {
      settings.northing = parse_double(value, option);
    } else if (option == "--elevation") {
      settings.elevation = parse_double(value, option);
    } else {
      throw std::invalid_argument("Unknown option: " + std::string(option));
    }
  }
  return settings;
}

/// Return the level-0 prepared Swiss tile containing an LV95 coordinate.
[[nodiscard]] std::filesystem::path
find_level_0_tile(uint32_t power, double easting, double northing) {
  const double tile_width = std::ldexp(kSwissCellSize, int(power));
  const double column_value = std::floor(easting / tile_width);
  const double row_value = std::floor(-northing / tile_width);
  if (column_value < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      column_value > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
      row_value < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      row_value > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::out_of_range("Origin is outside the supported Swiss tile grid");
  }
  const int64_t column = static_cast<int64_t>(column_value);
  const int64_t row = static_cast<int64_t>(row_value);
  const std::string level = std::to_string(power);
  const std::string stem = "swissalti3d_level-0_p" + level + "_r" + std::to_string(row) + "_c" +
                           std::to_string(column) + ".tif";
  return std::filesystem::path("data") / ("swissalti3d-" + level + "-level-0") / stem;
}

} // namespace

/// Trace the fixed level-0 Swiss terrain field around the temporary observer.
int main(int argc, const char *argv[]) {
  try {
    const EntrypointSettings settings = parse_arguments(argc, argv);
    const std::filesystem::path tile_path =
        find_level_0_tile(settings.rechunk_level, settings.easting, settings.northing);
    if (!std::filesystem::is_regular_file(tile_path)) {
      throw std::runtime_error(
          "No level-0 prepared tile contains this origin: " + tile_path.string()
      );
    }
    const panorama::RaytraceConfig config = {
        tile_path,
        {settings.easting, settings.northing, settings.elevation},
        settings.num_azimuth,
        settings.num_polar,
        0.0 * std::numbers::pi_v<double>,
        2.0 * std::numbers::pi_v<double>,
        -0.5 * std::numbers::pi_v<double>,
        0.5 * std::numbers::pi_v<double>,
        settings.max_distance,
        settings.max_tile_count,
        settings.tile_cache_size_bytes,
        settings.max_tile_preparation_workers,
    };
    std::printf(
        "Tracing %s from LV95 (%.3f, %.3f, %.1f).\n",
        tile_path.c_str(),
        settings.easting,
        settings.northing,
        settings.elevation
    );
    // Echo every benchmark-relevant setting so redirected timing logs remain
    // self-describing when several command-line configurations are compared.
    std::printf(
        "Settings: rechunk p%u, cache %.0f MiB, workers %u, max tiles %u, "
        "%u azimuths x %u polars, range %.0f m.\n",
        settings.rechunk_level,
        static_cast<double>(settings.tile_cache_size_bytes) / static_cast<double>(kBytesPerMiB),
        settings.max_tile_preparation_workers,
        settings.max_tile_count,
        settings.num_azimuth,
        settings.num_polar,
        settings.max_distance
    );
    panorama::perform_multi_tile_raytrace(config);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
