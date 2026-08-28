#include "terrain_catalogue.h"

#include "gdal_utils.h"
#include "metal_tile.h"
#include "terrain_manifest.h"

#include <gdal_priv.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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

  register_gdal_drivers();
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
  GdalDatasetPointer dataset(raw_dataset);

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

/// Return the one-based terrain LOD selected for one source tile.
uint32_t tile_lod(
    const TileGrid &grid,
    TileKey key,
    const ObserverLocation &observer,
    float base_cell_size,
    float pixel_angle,
    float lod_scale,
    uint32_t available_lod_count
) {
  if (!std::isfinite(base_cell_size) || base_cell_size <= 0.0F || !std::isfinite(pixel_angle) ||
      pixel_angle <= 0.0F || !std::isfinite(lod_scale) || lod_scale < 0.0F ||
      available_lod_count == 0U) {
    throw std::invalid_argument("Terrain LOD parameters must be finite and valid");
  }
  if (lod_scale == 0.0F) {
    return 1U;
  }

  // A level is doubled in cell spacing. With sin(a) ~= a, a pixel subtends
  // d * pixel_angle metres at the nearest point of this tile. Select the
  // coarsest representation no wider than lod_scale times that footprint.
  const double footprint = tile_minimum_distance(grid, key, observer) * pixel_angle;
  const double ratio = static_cast<double>(lod_scale) * footprint / base_cell_size;
  if (ratio < 1.0) {
    return 1U;
  }
  if (!std::isfinite(ratio)) {
    return available_lod_count;
  }
  const double logarithm = std::floor(std::log2(ratio));
  if (logarithm >= static_cast<double>(std::numeric_limits<uint32_t>::max() - 1U)) {
    return available_lod_count;
  }
  return std::min(available_lod_count, 1U + static_cast<uint32_t>(logarithm));
}

TerrainCatalogue::TerrainCatalogue(
    TileGrid grid,
    std::vector<TerrainSource> sources,
    ObserverLocation observer,
    std::vector<TileKey> coverage_tiles
)
    : grid_(grid), sources_(std::move(sources)), observer_(observer),
      coverage_{grid, std::move(coverage_tiles)} {
  float maximum_elevation = std::numeric_limits<float>::lowest();
  bool has_complete_maxima = true;
  for (uint32_t index = 0U; index < sources_.size(); index++) {
    if (!source_index_by_key_.emplace(sources_[index].key, index).second) {
      throw std::runtime_error("Prepared-terrain directory contains duplicate tile keys");
    }
    if (!sources_[index].maximum_elevation.has_value()) {
      has_complete_maxima = false;
      continue;
    }
    maximum_elevation = std::max(maximum_elevation, *sources_[index].maximum_elevation);
  }
  if (has_complete_maxima) {
    maximum_elevation_ = maximum_elevation;
  }
}

TerrainCatalogue TerrainCatalogue::discover(
    const std::filesystem::path &tile_dir,
    const ObserverLocation &observer,
    float max_distance,
    uint32_t max_tile_count,
    bool allow_observer_fallback
) {
  if (!std::filesystem::is_directory(tile_dir)) {
    throw std::invalid_argument("Prepared terrain path is not a directory: " + tile_dir.string());
  }

  std::map<TileKey, float> maximum_elevation_by_key;
  const std::filesystem::path manifest = terrain_manifest_path(tile_dir);
  if (std::filesystem::exists(manifest)) {
    for (const TerrainManifestEntry &entry : read_terrain_manifest(manifest)) {
      const TileKey key = {entry.row, entry.column};
      if (!maximum_elevation_by_key.emplace(key, entry.maximum_elevation).second) {
        throw std::runtime_error("Terrain manifest contains duplicate tile keys");
      }
    }
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
      const auto maximum = maximum_elevation_by_key.find(key);
      const bool metal = is_metal_tile_path(entry.path());
      available_sources.push_back(
          {
              key,
              entry.path(),
              !metal || maximum == maximum_elevation_by_key.end()
                  ? std::nullopt
                  : std::optional<float>(maximum->second),
              1U,
          }
      );
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
  // Every tile directory is one tile-generation output and therefore has one
  // common set of LOD representations. Reading a compressed header requires a
  // synchronous Metal I/O decompression, so inspect one representative custom
  // file instead of serially opening every source before tracing can begin.
  const auto custom_source = std::find_if(
      available_sources.begin(),
      available_sources.end(),
      [](const TerrainSource &source) { return is_metal_tile_path(source.path); }
  );
  if (custom_source != available_sources.end()) {
    const MetalTileHeader header = read_metal_tile_header(custom_source->path);
    const uint32_t lod_count = header.version == kMetalTileLodVersion ? header.lod_count : 1U;
    for (TerrainSource &source : available_sources) {
      if (is_metal_tile_path(source.path)) {
        source.lod_count = lod_count;
      }
    }
  }
  std::vector<TileKey> coverage_tiles;
  coverage_tiles.reserve(available_sources.size());
  for (const TerrainSource &source : available_sources) {
    coverage_tiles.push_back(source.key);
  }

  ObserverLocation resolved_observer = observer;
  TileKey origin_key = tile_key_at(grid, observer.easting, observer.northing);
  const auto has_origin = [&] {
    const auto found = std::lower_bound(
        available_sources.begin(),
        available_sources.end(),
        origin_key,
        [](const auto &source, const auto &key) { return source.key < key; }
    );
    return found != available_sources.end() && found->key == origin_key;
  };
  if (!has_origin()) {
    if (!allow_observer_fallback) {
      throw std::runtime_error("No prepared terrain tile contains the observer");
    }

    // Prefer a real tile nearest the centre of the dataset's key-space bounds.
    // This avoids selecting a missing tile in a coverage hole and gives sparse
    // or irregular datasets a deterministic, broadly representative start.
    int64_t minimum_row = available_sources.front().key.row;
    int64_t maximum_row = minimum_row;
    int64_t minimum_column = available_sources.front().key.column;
    int64_t maximum_column = minimum_column;
    for (const TerrainSource &source : available_sources) {
      minimum_row = std::min(minimum_row, source.key.row);
      maximum_row = std::max(maximum_row, source.key.row);
      minimum_column = std::min(minimum_column, source.key.column);
      maximum_column = std::max(maximum_column, source.key.column);
    }
    const double centre_row =
        0.5 * static_cast<double>(minimum_row) + 0.5 * static_cast<double>(maximum_row);
    const double centre_column =
        0.5 * static_cast<double>(minimum_column) + 0.5 * static_cast<double>(maximum_column);
    const TerrainSource *fallback = &available_sources.front();
    double fallback_distance = std::numeric_limits<double>::infinity();
    for (const TerrainSource &source : available_sources) {
      const double row_offset = static_cast<double>(source.key.row) - centre_row;
      const double column_offset = static_cast<double>(source.key.column) - centre_column;
      const double distance = row_offset * row_offset + column_offset * column_offset;
      if (distance < fallback_distance) {
        fallback = &source;
        fallback_distance = distance;
      }
    }
    origin_key = fallback->key;
    resolved_observer.easting =
        grid.origin_x + (static_cast<double>(origin_key.column) + 0.5) * grid.width;
    resolved_observer.northing =
        grid.origin_y - (static_cast<double>(origin_key.row) + 0.5) * grid.width;
  }
  std::vector<TerrainSource> sources;
  for (TerrainSource &source : available_sources) {
    if (source.key == origin_key ||
        tile_minimum_distance(grid, source.key, resolved_observer) <= max_distance) {
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
    throw std::logic_error("Terrain catalogue lost its resolved observer tile");
  }
  return TerrainCatalogue(grid, std::move(sources), resolved_observer, std::move(coverage_tiles));
}

const TileGrid &TerrainCatalogue::grid() const { return grid_; }
const TerrainSource &TerrainCatalogue::origin() const { return sources_.front(); }
const ObserverLocation &TerrainCatalogue::observer() const { return observer_; }
const TerrainCoverage &TerrainCatalogue::coverage() const { return coverage_; }
const std::vector<TerrainSource> &TerrainCatalogue::sources() const { return sources_; }
std::optional<float> TerrainCatalogue::maximum_elevation() const { return maximum_elevation_; }

std::optional<uint32_t> TerrainCatalogue::find_source(TileKey key) const {
  const auto found = source_index_by_key_.find(key);
  if (found == source_index_by_key_.end()) {
    return std::nullopt;
  }
  return found->second;
}

} // namespace panorama
