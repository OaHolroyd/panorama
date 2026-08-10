#include "raytrace_setup.h"

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

namespace {

constexpr double kSwissCellSize = 2.0;
constexpr uint32_t kRechunkLevel = 10;
constexpr double kObserverEasting = 2628183.79;
constexpr double kObserverNorthing = 1105668.21;
constexpr double kObserverElevation = 2080.0;

/// Return the level-1 prepared Swiss tile containing an LV95 coordinate.
[[nodiscard]] std::filesystem::path
find_level_1_tile(uint32_t power, double easting, double northing) {
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
  const std::string stem = "swissalti3d_level-1_p" + level + "_r" + std::to_string(row) + "_c" +
                           std::to_string(column) + ".tif";
  return std::filesystem::path("data") / ("swissalti3d-" + level + "-level-1") / stem;
}

} // namespace

/// Trace the fixed level-1 Swiss tile containing the temporary observer.
int main() {
  try {
    const std::filesystem::path tile_path =
        find_level_1_tile(kRechunkLevel, kObserverEasting, kObserverNorthing);
    if (!std::filesystem::is_regular_file(tile_path)) {
      throw std::runtime_error(
          "No level-1 prepared tile contains this origin: " + tile_path.string()
      );
    }
    const panorama::RaytraceConfig config = {
        tile_path,
        {kObserverEasting, kObserverNorthing, kObserverElevation},
        64,
        32,
        0.49 * std::numbers::pi_v<double>,
        0.51 * std::numbers::pi_v<double>,
        -0.5 * std::numbers::pi_v<double>,
        0.5 * std::numbers::pi_v<double>,
        20'000.0F,
    };
    std::printf(
        "Tracing %s from LV95 (%.3f, %.3f, %.1f).\n",
        tile_path.c_str(),
        kObserverEasting,
        kObserverNorthing,
        kObserverElevation
    );
    panorama::perform_single_tile_raytrace(config);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
