#include "terrain_catalogue.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace panorama {
namespace {

/// Parse one signed grid coordinate from a prepared terrain filename component.
[[nodiscard]] int64_t parse_tile_coordinate(std::string_view text, const char *name) {
  int64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::invalid_argument(
        std::string("Invalid ") + name + " tile coordinate: " + std::string(text)
    );
  }
  return value;
}

/// Extract the row/column key encoded in one prepared terrain filename.
[[nodiscard]] TileKey parse_tile_name(const std::filesystem::path &path) {
  const std::string name = path.filename().string();
  const size_t row_marker = name.rfind("_r");
  const size_t column_marker = name.rfind("_c");
  const size_t extension = name.rfind(".tif");
  if (row_marker == std::string::npos || column_marker == std::string::npos ||
      extension == std::string::npos || row_marker >= column_marker || column_marker >= extension) {
    throw std::invalid_argument("Prepared tile name must end in _rROW_cCOLUMN.tif: " + path.string());
  }
  return {
      parse_tile_coordinate(
          std::string_view(name).substr(row_marker + 2U, column_marker - row_marker - 2U),
          "row"
      ),
      parse_tile_coordinate(
          std::string_view(name).substr(column_marker + 2U, extension - column_marker - 2U),
          "column"
      ),
  };
}

} // namespace

bool TileKey::operator<(const TileKey &other) const {
  if (row != other.row) {
    return row < other.row;
  }
  return column < other.column;
}

bool TileKey::operator==(const TileKey &other) const {
  return row == other.row && column == other.column;
}

TileKey tile_key_at(const TileGrid &grid, double easting, double northing) {
  const double column = std::floor((easting - grid.origin_x) / grid.width);
  const double row = std::floor((grid.origin_y - northing) / grid.width);
  if (column < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      column > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
      row < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      row > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::out_of_range("Terrain coordinate is outside the supported tile grid");
  }
  return {static_cast<int64_t>(row), static_cast<int64_t>(column)};
}

double tile_minimum_distance(const TileGrid &grid, TileKey key, const ObserverLocation &observer) {
  const double x_min = grid.origin_x + static_cast<double>(key.column) * grid.width;
  const double y_max = grid.origin_y - static_cast<double>(key.row) * grid.width;
  const double x_max = x_min + grid.width;
  const double y_min = y_max - grid.width;
  const double dx = observer.easting < x_min   ? x_min - observer.easting
                    : observer.easting > x_max ? observer.easting - x_max
                                                : 0.0;
  const double dy = observer.northing < y_min   ? y_min - observer.northing
                    : observer.northing > y_max ? observer.northing - y_max
                                                 : 0.0;
  return std::hypot(dx, dy);
}

TerrainCatalogue::TerrainCatalogue(TileGrid grid, std::vector<TerrainSource> sources)
    : grid_(grid), sources_(std::move(sources)) {
  for (uint32_t index = 0U; index < sources_.size(); index++) {
    if (!source_index_by_key_.emplace(sources_[index].key, index).second) {
      throw std::runtime_error("Prepared-terrain directory contains duplicate tile keys");
    }
  }
}

TerrainCatalogue TerrainCatalogue::discover(
    const std::filesystem::path &tile_dir,
    TileGrid grid,
    const ObserverLocation &observer,
    float max_distance,
    uint32_t max_tile_count
) {
  const TileKey origin_key = tile_key_at(grid, observer.easting, observer.northing);
  std::vector<TerrainSource> sources;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(tile_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".tif") {
      continue;
    }
    try {
      const TileKey key = parse_tile_name(entry.path());
      if (key == origin_key || tile_minimum_distance(grid, key, observer) <= max_distance) {
        sources.push_back({key, entry.path()});
      }
    } catch (const std::invalid_argument &) {
      // Prepared-tile directories may contain unrelated GeoTIFFs.
    }
  }
  std::sort(sources.begin(), sources.end(), [origin_key](const TerrainSource &left,
                                                          const TerrainSource &right) {
    const uint64_t left_shell = static_cast<uint64_t>(std::llabs(left.key.row - origin_key.row)) +
                                static_cast<uint64_t>(std::llabs(left.key.column - origin_key.column));
    const uint64_t right_shell = static_cast<uint64_t>(std::llabs(right.key.row - origin_key.row)) +
                                 static_cast<uint64_t>(std::llabs(right.key.column - origin_key.column));
    if (left_shell != right_shell) {
      return left_shell < right_shell;
    }
    return left.key < right.key;
  });
  if (max_tile_count != 0U && sources.size() > max_tile_count) {
    sources.resize(max_tile_count);
  }
  if (sources.empty() || !(sources.front().key == origin_key)) {
    throw std::runtime_error("No prepared terrain tile contains the observer");
  }
  return TerrainCatalogue(grid, std::move(sources));
}

const TileGrid &TerrainCatalogue::grid() const { return grid_; }
const TerrainSource &TerrainCatalogue::origin() const { return sources_.front(); }
const std::vector<TerrainSource> &TerrainCatalogue::sources() const { return sources_; }

std::optional<uint32_t> TerrainCatalogue::find_source(TileKey key) const {
  const auto found = source_index_by_key_.find(key);
  if (found == source_index_by_key_.end()) {
    return std::nullopt;
  }
  return found->second;
}

} // namespace panorama
