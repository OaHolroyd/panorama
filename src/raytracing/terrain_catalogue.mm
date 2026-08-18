#include "terrain_catalogue.h"

#include "metal_tile.h"

#include <cpl_error.h>
#include <gdal_priv.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace panorama {
namespace {

/// Close a metadata-only GDAL dataset on every return path.
struct DatasetCloser {
  void operator()(GDALDataset *dataset) const { GDALClose(dataset); }
};

using DatasetPointer = std::unique_ptr<GDALDataset, DatasetCloser>;

/// Append GDAL's latest diagnostic to a catalogue-level error.
[[nodiscard]] std::string gdal_error(const std::string &context) {
  const char *detail = CPLGetLastErrorMsg();
  return context + (detail != nullptr && detail[0] != '\0' ? ": " + std::string(detail) : "");
}

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
  const size_t extension = name.find('.', column_marker);
  if (row_marker == std::string::npos || column_marker == std::string::npos ||
      extension == std::string::npos || row_marker >= column_marker || column_marker >= extension) {
    throw std::invalid_argument(
        "Prepared tile name must contain _rROW_cCOLUMN before its suffix: " + path.string()
    );
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

/// Derive the global tile grid from one prepared level-0 GeoTIFF.
///
/// The filename supplies only the signed row and column. The affine transform
/// and raster dimensions supply the cell spacing and physical tile width;
/// combining them reconstructs the common north-west grid origin without any
/// dataset-specific constants.
[[nodiscard]] TileGrid infer_tile_grid(const TerrainSource &source) {
  if (is_metal_tile_path(source.path)) {
    const MetalTileHeader header = read_metal_tile_header(source.path);
    if (header.row != source.key.row || header.column != source.key.column) {
      throw std::runtime_error("Metal tile filename and header contain different grid keys");
    }

    const double tile_width = static_cast<double>(header.cell_count) * header.cell_size;
    const double origin_x = header.lower_left_x - static_cast<double>(header.column) * tile_width;
    const double origin_y = header.lower_left_y + static_cast<double>(header.row + 1) * tile_width;
    if (!std::isfinite(tile_width) || !std::isfinite(origin_x) || !std::isfinite(origin_y) ||
        tile_width <= 0.0) {
      throw std::runtime_error("Metal tile has an invalid global grid");
    }
    return {origin_x, origin_y, tile_width};
  }

  GDALAllRegister();
  const char *drivers[] = {"GTiff", nullptr};
  GDALDataset *raw_dataset = static_cast<GDALDataset *>(GDALOpenEx(
      source.path.string().c_str(),
      GDAL_OF_RASTER | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR,
      drivers,
      nullptr,
      nullptr
  ));
  if (raw_dataset == nullptr) {
    throw std::runtime_error(gdal_error("Could not inspect prepared tile " + source.path.string()));
  }
  DatasetPointer dataset(raw_dataset);

  const int sample_width = dataset->GetRasterXSize();
  const int sample_height = dataset->GetRasterYSize();
  if (dataset->GetRasterCount() != 1 || sample_width < 2 || sample_width != sample_height) {
    throw std::runtime_error("Prepared level-0 terrain tile must be a square single-band raster");
  }

  double transform[6] = {};
  if (dataset->GetGeoTransform(transform) != CE_None || transform[1] <= 0.0 ||
      transform[5] >= 0.0 || transform[2] != 0.0 || transform[4] != 0.0 ||
      transform[1] != -transform[5]) {
    throw std::runtime_error("Prepared terrain tile must have a square, north-up affine grid");
  }

  // A level-0 tile has one more vertex sample than terrain cells. Tile keys
  // advance by the non-overlapping cell count, not by the stored sample count.
  const double tile_width = static_cast<double>(sample_width - 1) * transform[1];
  const double origin_x = transform[0] - static_cast<double>(source.key.column) * tile_width;
  const double origin_y = transform[3] + static_cast<double>(source.key.row) * tile_width;
  if (!std::isfinite(tile_width) || !std::isfinite(origin_x) || !std::isfinite(origin_y) ||
      tile_width <= 0.0) {
    throw std::runtime_error("Prepared terrain tile has an invalid global grid");
  }
  return {origin_x, origin_y, tile_width};
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
    const ObserverLocation &observer,
    float max_distance,
    uint32_t max_tile_count
) {
  if (!std::filesystem::is_directory(tile_dir)) {
    throw std::invalid_argument("Prepared terrain path is not a directory: " + tile_dir.string());
  }

  // Filenames provide stable integer keys, while one file's embedded metadata
  // establishes the dataset-independent physical grid shared by the directory.
  std::vector<TerrainSource> available_sources;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(tile_dir)) {
    if (!entry.is_regular_file() && !entry.is_symlink()) {
      continue;
    }
    if (entry.path().extension() != ".tif" && !is_metal_tile_path(entry.path())) {
      continue;
    }
    try {
      const TileKey key = parse_tile_name(entry.path());
      available_sources.push_back({key, entry.path()});
    } catch (const std::invalid_argument &) {
      // Prepared-tile directories may contain unrelated GeoTIFFs.
    }
  }
  if (available_sources.empty()) {
    throw std::runtime_error("Prepared-terrain directory contains no indexed terrain tiles");
  }
  std::sort(
      available_sources.begin(),
      available_sources.end(),
      [](const TerrainSource &left, const TerrainSource &right) { return left.key < right.key; }
  );

  const TileGrid grid = infer_tile_grid(available_sources.front());
  const TileKey origin_key = tile_key_at(grid, observer.easting, observer.northing);
  std::vector<TerrainSource> sources;
  for (TerrainSource &source : available_sources) {
    if (source.key == origin_key ||
        tile_minimum_distance(grid, source.key, observer) <= max_distance) {
      sources.push_back(std::move(source));
    }
  }
  std::sort(
      sources.begin(),
      sources.end(),
      [origin_key](const TerrainSource &left, const TerrainSource &right) {
        const uint64_t left_shell =
            static_cast<uint64_t>(std::llabs(left.key.row - origin_key.row)) +
            static_cast<uint64_t>(std::llabs(left.key.column - origin_key.column));
        const uint64_t right_shell =
            static_cast<uint64_t>(std::llabs(right.key.row - origin_key.row)) +
            static_cast<uint64_t>(std::llabs(right.key.column - origin_key.column));
        if (left_shell != right_shell) {
          return left_shell < right_shell;
        }
        return left.key < right.key;
      }
  );
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
