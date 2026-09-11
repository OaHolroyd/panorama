#include "terrain_catalogue.h"

#include "metal_tile.h"
#include "terrain_manifest.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
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

/// Derive the global tile grid from one indexed Metal tile.
///
/// The filename supplies only the signed row and column. The affine transform
/// and raster dimensions supply the cell spacing and physical tile width;
/// combining them reconstructs the common north-west grid origin without any
/// dataset-specific constants.
[[nodiscard]] TileGrid infer_tile_grid(const TerrainSource &source) {
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

[[nodiscard]] TerrainDataset
discover_dataset(const TerrainDatasetConfig &config, uint32_t dataset_index) {
  const std::filesystem::path &tile_dir = config.directory;
  if (!std::filesystem::is_directory(tile_dir)) {
    throw std::invalid_argument("Prepared terrain path is not a directory: " + tile_dir.string());
  }
  if (!std::isfinite(config.vertical_offset_metres))
    throw std::invalid_argument("Terrain vertical offset must be finite");

  std::map<TileKey, TerrainManifestEntry> elevation_by_key;
  const std::filesystem::path manifest = terrain_manifest_path(tile_dir);
  if (std::filesystem::exists(manifest)) {
    for (const TerrainManifestEntry &entry : read_terrain_manifest(manifest)) {
      const TileKey key = {entry.row, entry.column};
      if (!elevation_by_key.emplace(key, entry).second)
        throw std::runtime_error("Terrain manifest contains duplicate tile keys");
    }
  }

  std::vector<TerrainSource> sources;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(tile_dir)) {
    if ((!entry.is_regular_file() && !entry.is_symlink()) || !is_metal_tile_path(entry.path()))
      continue;
    try {
      const TileKey key = parse_tile_name(entry.path());
      const auto elevation = elevation_by_key.find(key);
      sources.push_back(
          {
              key,
              entry.path(),
              elevation == elevation_by_key.end()
                  ? std::nullopt
                  : std::optional<float>(elevation->second.maximum_elevation),
              1U,
              elevation == elevation_by_key.end() ? std::nullopt
                                                  : elevation->second.minimum_elevation,
              dataset_index,
              {},
              {},
              {},
              0.0,
              config.vertical_offset_metres,
              {},
          }
      );
    } catch (const std::invalid_argument &) {
      // Prepared directories may contain unrelated files.
    }
  }
  if (sources.empty())
    throw std::runtime_error("Prepared-terrain directory contains no indexed .ptile files");
  std::sort(sources.begin(), sources.end(), [](const auto &left, const auto &right) {
    return left.key < right.key;
  });
  for (size_t index = 1; index < sources.size(); ++index)
    if (sources[index - 1].key == sources[index].key)
      throw std::runtime_error("Prepared-terrain directory contains duplicate tile keys");

  const MetalTileHeader header = read_metal_tile_header(sources.front().path);
  const TileGrid grid = infer_tile_grid(sources.front());
  for (TerrainSource &source : sources) {
    source.lod_count = header.lod_count;
    const auto entry = elevation_by_key.find(source.key);
    auto coverage = entry == elevation_by_key.end() ? std::optional<TerrainCellCoverage>{}
                                                    : entry->second.coverage;
    if (!coverage)
      coverage = read_metal_tile_coverage(source.path, read_metal_tile_header(source.path));
    if (coverage) {
      if (coverage->cell_count != header.cell_count)
        throw std::runtime_error("Terrain coverage disagrees with its grid");
      source.valid_cells = std::make_shared<TerrainCellCoverage>(std::move(*coverage));
    }
  }
  return {
      config,
      dataset_index,
      grid,
      header.epsg_code,
      header.cell_count,
      header.level_count,
      header.lod_count,
      header.sample_type,
      header.compression,
      std::move(sources),
  };
}

} // namespace

std::vector<TerrainDataset>
discover_terrain_datasets(std::span<const TerrainDatasetConfig> configs) {
  if (configs.empty())
    throw std::invalid_argument("At least one terrain dataset is required");
  std::set<std::filesystem::path> directories;
  std::vector<TerrainDataset> datasets;
  datasets.reserve(configs.size());
  for (size_t index = 0; index < configs.size(); ++index) {
    const auto directory = std::filesystem::weakly_canonical(configs[index].directory);
    if (!directories.insert(directory).second)
      throw std::invalid_argument(
          "Terrain dataset is configured more than once: " + directory.string()
      );
    if (index > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("Too many terrain datasets");
    datasets.push_back(discover_dataset(configs[index], static_cast<uint32_t>(index)));
  }
  const TerrainDataset &layout = datasets.front();
  for (size_t index = 1; index < datasets.size(); ++index) {
    const TerrainDataset &dataset = datasets[index];
    if (dataset.cell_count != layout.cell_count || dataset.level_count != layout.level_count ||
        dataset.lod_count != layout.lod_count || dataset.sample_type != layout.sample_type ||
        dataset.compression != layout.compression) {
      throw std::runtime_error(
          "Terrain dataset payload layout differs from the first dataset: " +
          dataset.config.directory.string()
      );
    }
  }
  return datasets;
}

TerrainRenderFrame
select_terrain_render_frame(std::span<const TerrainDataset> datasets, Coord navigation_observer) {
  if (datasets.empty())
    throw std::invalid_argument("Cannot select a render frame without terrain datasets");
  if (datasets.size() == 1U && epsg_uses_projected_metres(datasets.front().epsg_code))
    return TerrainRenderFrame::fixed(datasets.front().epsg_code);
  const std::array<Coord, 1> observer = {navigation_observer};
  const Coord geographic =
      transform_coordinates(datasets.front().epsg_code, 4326U, observer).front();
  return TerrainRenderFrame::local_aeqd({geographic.y, geographic.x});
}

bool TileKey::operator<(const TileKey &other) const {
  if (row != other.row) {
    return row < other.row;
  }
  return column < other.column;
}

bool TileKey::operator==(const TileKey &other) const {
  return row == other.row && column == other.column;
}

bool TerrainCatalogue::SourceKey::operator<(const SourceKey &other) const {
  if (dataset != other.dataset)
    return dataset < other.dataset;
  return tile < other.tile;
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

namespace {

enum class CoverageRelation { Uncovered, Covered, Partial };

[[nodiscard]] const TerrainSource *dataset_source(const TerrainDataset &dataset, TileKey key) {
  const auto found = std::lower_bound(
      dataset.sources.begin(),
      dataset.sources.end(),
      key,
      [](const TerrainSource &source, TileKey wanted) { return source.key < wanted; }
  );
  return found != dataset.sources.end() && found->key == key ? &*found : nullptr;
}

/// Match subdivisions on each shared tile boundary as well as inside a tile.
/// The adjacent source's valid-cell endpoints can meet the middle of our edge.
[[nodiscard]] std::vector<std::array<uint32_t, 2>>
coverage_boundary_junctions(const TerrainDataset &dataset, TileKey key) {
  const uint32_t side = dataset.cell_count;
  std::vector<std::array<uint32_t, 2>> points;
  for (uint32_t edge = 0; edge < 4U; ++edge) {
    const TileKey neighbour = {key.row + (edge == 2U   ? 1
                                          : edge == 3U ? -1
                                                       : 0),
                               key.column + (edge == 0U   ? -1
                                             : edge == 1U ? 1
                                                          : 0)};
    const auto *source = dataset_source(dataset, neighbour);
    if (!source || !source->valid_cells || source->valid_cells->full())
      continue;
    for (const auto &rect : source->valid_cells->rectangles) {
      if ((edge == 0U && rect.column + rect.width == side) || (edge == 1U && rect.column == 0U)) {
        points.push_back({edge == 0U ? 0U : side, rect.row});
        points.push_back({edge == 0U ? 0U : side, rect.row + rect.height});
      }
      if ((edge == 2U && rect.row + rect.height == side) || (edge == 3U && rect.row == 0U)) {
        points.push_back({rect.column, edge == 2U ? 0U : side});
        points.push_back({rect.column + rect.width, edge == 2U ? 0U : side});
      }
    }
  }
  return points;
}

/// Use a logical rectangle's corners, edge midpoints, and centre to bound its
/// footprint in a higher-priority grid. A completely populated key rectangle
/// proves coverage; an empty one proves no overlap. Mixed rectangles are
/// subdivided until only cells crossed by a coverage edge need exact handling.
[[nodiscard]] CoverageRelation coarse_coverage(
    const MetalTileHeader &header,
    const TerrainDataset &higher,
    const CoordinateTransform &to_higher,
    uint32_t logical_minimum_column,
    uint32_t logical_minimum_row,
    uint32_t cell_width,
    uint32_t cell_height
) {
  std::array<Coord, 9> samples;
  size_t index = 0U;
  for (double y : {0.0, 0.5, 1.0})
    for (double x : {0.0, 0.5, 1.0})
      samples[index++] = {
          header.lower_left_x + (logical_minimum_column + x * cell_width) * header.cell_size,
          header.lower_left_y + (logical_minimum_row + y * cell_height) * header.cell_size};
  const auto transformed = to_higher.apply(samples);
  int64_t minimum_row = std::numeric_limits<int64_t>::max();
  int64_t maximum_row = std::numeric_limits<int64_t>::min();
  int64_t minimum_column = std::numeric_limits<int64_t>::max();
  int64_t maximum_column = std::numeric_limits<int64_t>::min();
  for (Coord point : transformed) {
    const TileKey key = tile_key_at(higher.grid, point.x, point.y);
    minimum_row = std::min(minimum_row, key.row);
    maximum_row = std::max(maximum_row, key.row);
    minimum_column = std::min(minimum_column, key.column);
    maximum_column = std::max(maximum_column, key.column);
  }
  double x0 = transformed.front().x, x1 = x0;
  double y0 = transformed.front().y, y1 = y0;
  for (Coord point : transformed) {
    x0 = std::min(x0, point.x);
    x1 = std::max(x1, point.x);
    y0 = std::min(y0, point.y);
    y1 = std::max(y1, point.y);
  }
  bool any = false, all = true;
  for (int64_t row = minimum_row; row <= maximum_row; ++row) {
    for (int64_t column = minimum_column; column <= maximum_column; ++column) {
      const double left = higher.grid.origin_x + double(column) * higher.grid.width;
      const double bottom = higher.grid.origin_y - double(row + 1) * higher.grid.width;
      const double ax = std::max(x0, left), bx = std::min(x1, left + higher.grid.width);
      const double ay = std::max(y0, bottom), by = std::min(y1, bottom + higher.grid.width);
      if (ax >= bx || ay >= by)
        continue;
      const TerrainSource *source = dataset_source(higher, {row, column});
      if (!source) {
        all = false;
        continue;
      }
      if (!source->valid_cells || source->valid_cells->full()) {
        any = true;
        continue;
      }
      const auto &coverage = *source->valid_cells;
      const double scale = coverage.cell_count / higher.grid.width;
      const auto low = [&](double v) {
        return uint32_t(std::clamp(std::floor(v * scale), 0.0, double(coverage.cell_count)));
      };
      const auto high = [&](double v) {
        return uint32_t(std::clamp(std::ceil(v * scale), 0.0, double(coverage.cell_count)));
      };
      const uint32_t cx0 = low(ax - left), cx1 = high(bx - left);
      const uint32_t cy0 = low(ay - bottom), cy1 = high(by - bottom);
      const uint64_t area = coverage.covered_area(cx0, cy0, cx1, cy1);
      any = any || area != 0U;
      all = all && area == uint64_t(cx1 - cx0) * (cy1 - cy0);
    }
  }
  return all   ? CoverageRelation::Covered
         : any ? CoverageRelation::Partial
               : CoverageRelation::Uncovered;
}

[[nodiscard]] CoverageRelation coarse_coverage(
    const MetalTileHeader &header,
    const TerrainDataset &higher,
    const CoordinateTransform &to_higher
) {
  return coarse_coverage(header, higher, to_higher, 0U, 0U, header.cell_count, header.cell_count);
}

[[nodiscard]] std::vector<TerrainTransformPatch> intersect_owned_rectangles(
    std::span<const TerrainTransformPatch> transforms,
    std::span<const TerrainTransformPatch> ownership
) {
  std::vector<TerrainTransformPatch> result;
  for (const TerrainTransformPatch &owned : ownership) {
    const uint32_t ox1 = owned.minimum_column + owned.cell_width;
    const uint32_t oy1 = owned.minimum_row + owned.cell_height;
    for (const TerrainTransformPatch &transform : transforms) {
      const uint32_t tx1 = transform.minimum_column + transform.cell_width;
      const uint32_t ty1 = transform.minimum_row + transform.cell_height;
      const uint32_t x0 = std::max(owned.minimum_column, transform.minimum_column);
      const uint32_t y0 = std::max(owned.minimum_row, transform.minimum_row);
      const uint32_t x1 = std::min(ox1, tx1);
      const uint32_t y1 = std::min(oy1, ty1);
      if (x0 < x1 && y0 < y1)
        result.push_back({x0, y0, x1 - x0, y1 - y0, transform.transform});
    }
  }
  return result;
}

/// Return affine rectangles owned by this source after higher-priority tile
/// coverage is removed. Boundary cells remain candidates; GPU hit-position
/// checks remove their overlap with higher-priority coverage.
[[nodiscard]] std::vector<TerrainTransformPatch> owned_transform_patches(
    const MetalTileHeader &header,
    std::span<const TerrainTransformPatch> transforms,
    std::span<const TerrainDataset> higher_datasets
) {
  if (higher_datasets.empty())
    return {transforms.begin(), transforms.end()};
  std::vector<std::unique_ptr<CoordinateTransform>> to_higher;
  to_higher.reserve(higher_datasets.size());
  bool any_possible_overlap = false;
  for (const TerrainDataset &higher : higher_datasets) {
    to_higher.push_back(std::make_unique<CoordinateTransform>(header.epsg_code, higher.epsg_code));
    const CoverageRelation relation = coarse_coverage(header, higher, *to_higher.back());
    if (relation == CoverageRelation::Covered)
      return {};
    any_possible_overlap = any_possible_overlap || relation == CoverageRelation::Partial;
  }
  if (!any_possible_overlap)
    return {transforms.begin(), transforms.end()};

  const uint32_t side = header.cell_count;
  struct Region {
    uint32_t column, row, width, height;
  };
  std::vector<Region> pending = {{0U, 0U, side, side}};
  std::vector<TerrainTransformPatch> owned;
  while (!pending.empty()) {
    const Region region = pending.back();
    pending.pop_back();
    bool covered = false;
    bool partial = false;
    for (size_t higher_index = 0; higher_index < higher_datasets.size(); ++higher_index) {
      const CoverageRelation relation = coarse_coverage(
          header,
          higher_datasets[higher_index],
          *to_higher[higher_index],
          region.column,
          region.row,
          region.width,
          region.height
      );
      covered = covered || relation == CoverageRelation::Covered;
      partial = partial || relation == CoverageRelation::Partial;
    }
    if (covered)
      continue;
    if (!partial) {
      owned.push_back({region.column, region.row, region.width, region.height, {}});
      continue;
    }
    if (region.width == 1U && region.height == 1U) {
      // Keep cells crossed by a priority edge. The exact hit position decides
      // ownership on GPU, so rounding never opens a crack between grids.
      owned.push_back({region.column, region.row, 1U, 1U, {}});
      continue;
    }
    const uint32_t left = region.width > 1U ? region.width / 2U : region.width;
    const uint32_t right = region.width - left;
    const uint32_t bottom = region.height > 1U ? region.height / 2U : region.height;
    const uint32_t top = region.height - bottom;
    pending.push_back({region.column, region.row, left, bottom});
    if (right != 0U)
      pending.push_back({region.column + left, region.row, right, bottom});
    if (top != 0U)
      pending.push_back({region.column, region.row + bottom, left, top});
    if (right != 0U && top != 0U)
      pending.push_back({region.column + left, region.row + bottom, right, top});
  }

  // The subdivision naturally produces a quadtree mosaic. Canonicalise it
  // into long row runs so a smooth dataset boundary does not become thousands
  // of catalogue primitives.
  std::vector<uint8_t> ownership(static_cast<size_t>(side) * side, 0U);
  for (const TerrainTransformPatch &region : owned) {
    for (uint32_t row = region.minimum_row; row < region.minimum_row + region.cell_height; ++row) {
      auto begin =
          ownership.begin() + static_cast<ptrdiff_t>(size_t(row) * side + region.minimum_column);
      std::fill(begin, begin + region.cell_width, uint8_t{1});
    }
  }
  std::vector<TerrainTransformPatch> merged;
  std::map<std::pair<uint32_t, uint32_t>, size_t> active;
  for (uint32_t row = 0; row < side; ++row) {
    std::map<std::pair<uint32_t, uint32_t>, size_t> next;
    for (uint32_t column = 0; column < side;) {
      while (column < side && ownership[size_t(row) * side + column] == 0U)
        ++column;
      const uint32_t begin = column;
      while (column < side && ownership[size_t(row) * side + column] != 0U)
        ++column;
      if (begin == column)
        continue;
      const auto run = std::pair{begin, column};
      if (const auto previous = active.find(run); previous != active.end()) {
        merged[previous->second].cell_height++;
        next.emplace(run, previous->second);
      } else {
        next.emplace(run, merged.size());
        merged.push_back({begin, row, column - begin, 1U, {}});
      }
    }
    active = std::move(next);
  }
  return intersect_owned_rectangles(transforms, merged);
}

} // namespace

TerrainCatalogue::TerrainCatalogue(
    TileGrid grid,
    std::vector<TerrainSource> sources,
    ObserverLocation observer,
    std::vector<TileKey> coverage_tiles,
    std::vector<TerrainDataset> datasets,
    std::optional<TerrainRenderFrame> render_frame
)
    : grid_(grid), sources_(std::move(sources)), observer_(observer),
      coverage_{grid, std::move(coverage_tiles)}, datasets_(std::move(datasets)),
      render_frame_(std::move(render_frame)) {
  if (!datasets_.empty()) {
    navigation_to_dataset_.reserve(datasets_.size());
    for (const TerrainDataset &dataset : datasets_)
      navigation_to_dataset_.push_back(
          std::make_unique<CoordinateTransform>(datasets_.front().epsg_code, dataset.epsg_code)
      );
  }
  float maximum_elevation = std::numeric_limits<float>::lowest();
  bool has_complete_maxima = true;
  for (uint32_t index = 0U; index < sources_.size(); index++) {
    if (!source_index_by_key_
             .emplace(SourceKey{sources_[index].dataset_index, sources_[index].key}, index)
             .second) {
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
    std::span<const TerrainDatasetConfig> configs,
    const ObserverLocation &observer,
    float max_distance,
    uint32_t max_tile_count,
    bool allow_observer_fallback
) {
  if (!std::isfinite(observer.easting) || !std::isfinite(observer.northing) ||
      !std::isfinite(observer.elevation) || !std::isfinite(max_distance) || max_distance <= 0.0F)
    throw std::invalid_argument("Combined terrain catalogue requires finite observer and range");
  std::vector<TerrainDataset> datasets = discover_terrain_datasets(configs);
  const TerrainRenderFrame frame =
      select_terrain_render_frame(datasets, {observer.easting, observer.northing});
  const std::array<Coord, 1> observer_navigation = {{{observer.easting, observer.northing}}};
  const Coord render_observer =
      frame.project(datasets.front().epsg_code, observer_navigation).front();

  struct Candidate {
    TerrainSource source;
    double distance;
    bool contains_observer;
  };
  std::vector<Candidate> candidates;
  for (const TerrainDataset &dataset : datasets) {
    const Coord native_observer =
        transform_coordinates(datasets.front().epsg_code, dataset.epsg_code, observer_navigation)
            .front();
    const TileKey observer_key = tile_key_at(dataset.grid, native_observer.x, native_observer.y);
    const MetalTileHeader representative = read_metal_tile_header(dataset.sources.front().path);
    for (const TerrainSource &available : dataset.sources) {
      MetalTileHeader header = representative;
      header.row = available.key.row;
      header.column = available.key.column;
      header.lower_left_x =
          dataset.grid.origin_x + double(available.key.column) * dataset.grid.width;
      header.lower_left_y =
          dataset.grid.origin_y - double(available.key.row + 1) * dataset.grid.width;
      const TerrainTileTransform whole = make_terrain_tile_transform(header, frame);
      const double dx = render_observer.x < whole.bounds[0]   ? whole.bounds[0] - render_observer.x
                        : render_observer.x > whole.bounds[2] ? render_observer.x - whole.bounds[2]
                                                              : 0.0;
      const double dy = render_observer.y < whole.bounds[1]   ? whole.bounds[1] - render_observer.y
                        : render_observer.y > whole.bounds[3] ? render_observer.y - whole.bounds[3]
                                                              : 0.0;
      const double distance = std::hypot(dx, dy);
      if (distance > max_distance && available.key != observer_key)
        continue;
      TerrainSource source = available;
      source.vertical_offset_metres = dataset.config.vertical_offset_metres;
      if (source.maximum_elevation.has_value())
        *source.maximum_elevation += static_cast<float>(source.vertical_offset_metres);
      if (source.minimum_elevation.has_value())
        *source.minimum_elevation += static_cast<float>(source.vertical_offset_metres);
      // Geometry uses sufficiently small affine regions that independently
      // transformed neighbouring tiles differ by less than half a metre at a
      // shared edge. Coverage uses a coarser shared-vertex tessellation: it
      // proves source continuity but never supplies a rendered surface.
      constexpr double geometry_residual_metres = 0.25;
      const std::vector<TerrainTransformPatch> geometry_patches =
          make_terrain_transform_patches(header, frame, geometry_residual_metres);
      std::vector<TerrainTransformPatch> coverage_patches;
      if (source.valid_cells) {
        for (const auto &rect : source.valid_cells->rectangles)
          coverage_patches.push_back({rect.column, rect.row, rect.width, rect.height, whole});
      } else {
        coverage_patches.push_back({0U, 0U, header.cell_count, header.cell_count, whole});
      }
      const auto valid_geometry = intersect_owned_rectangles(geometry_patches, coverage_patches);
      source.transform_patches = owned_transform_patches(
          header,
          valid_geometry,
          std::span<const TerrainDataset>(datasets).first(dataset.index)
      );
      const std::vector<TerrainTransformPatch> coverage_ownership = owned_transform_patches(
          header,
          coverage_patches,
          std::span<const TerrainDataset>(datasets).first(dataset.index)
      );
      if (source.transform_patches.empty() || coverage_ownership.empty())
        continue;
      // Physical coverage includes all usable cells, independent of ownership.
      // Adjacent fallback cells can overlap at a priority edge without opening
      // an artificial gap in this continuity hierarchy.
      const auto boundary_junctions = coverage_boundary_junctions(dataset, available.key);
      source.coverage_polygons =
          make_terrain_coverage_polygons(header, frame, coverage_patches, 256U, boundary_junctions);
      uint64_t owned_cells = 0U;
      for (const TerrainTransformPatch &patch : source.transform_patches)
        owned_cells += uint64_t(patch.cell_width) * patch.cell_height;
      // A coarse sample can straddle an ownership edge. Keep boundary sources
      // at native resolution so lower-priority geometry never expands back
      // into a region removed above.
      if (owned_cells != uint64_t(header.cell_count) * header.cell_count) {
        source.lod_count = 1U;
        source.ownership_polygons = make_terrain_coverage_polygons(
            header,
            frame,
            coverage_ownership,
            256U,
            boundary_junctions
        );
      }
      source.effective_cell_size_metres = whole.maximum_cell_size_metres();
      const bool contains_observer =
          available.key == observer_key &&
          (!source.valid_cells || source.valid_cells->contains(
                                      (native_observer.x - header.lower_left_x) / header.cell_size,
                                      (native_observer.y - header.lower_left_y) / header.cell_size
                                  ));
      candidates.push_back({std::move(source), distance, contains_observer});
    }
  }
  if (candidates.empty())
    throw std::runtime_error("No prepared terrain lies within the configured range");
  const auto origin = std::find_if(candidates.begin(), candidates.end(), [](const Candidate &item) {
    return item.contains_observer;
  });
  if (origin == candidates.end() && !allow_observer_fallback)
    throw std::runtime_error("No prepared terrain tile contains the observer");
  if (origin == candidates.end()) {
    const auto &candidate = candidates.front().source;
    const auto &dataset = datasets[candidate.dataset_index];
    double column = 0.5, row = 0.5;
    if (candidate.valid_cells) {
      const auto &rect = candidate.valid_cells->rectangles.front();
      column = (rect.column + 0.5 * rect.width) / candidate.valid_cells->cell_count;
      row = (rect.row + 0.5 * rect.height) / candidate.valid_cells->cell_count;
    }
    const std::array<Coord, 1> point = {
        {{dataset.grid.origin_x + (double(candidate.key.column) + column) * dataset.grid.width,
          dataset.grid.origin_y - (double(candidate.key.row + 1) - row) * dataset.grid.width}}};
    const Coord navigation =
        transform_coordinates(dataset.epsg_code, datasets.front().epsg_code, point).front();
    return discover(
        configs,
        {navigation.x, navigation.y, observer.elevation},
        max_distance,
        max_tile_count,
        false
    );
  }
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const Candidate &left, const Candidate &right) {
        if (left.contains_observer != right.contains_observer)
          return left.contains_observer;
        if (left.distance != right.distance)
          return left.distance < right.distance;
        if (left.source.dataset_index != right.source.dataset_index)
          return left.source.dataset_index < right.source.dataset_index;
        return left.source.key < right.source.key;
      }
  );
  if (max_tile_count != 0U && candidates.size() > max_tile_count)
    candidates.resize(max_tile_count);
  std::vector<TerrainSource> sources;
  sources.reserve(candidates.size());
  for (Candidate &candidate : candidates)
    sources.push_back(std::move(candidate.source));
  std::vector<TileKey> primary_coverage;
  primary_coverage.reserve(datasets.front().sources.size());
  for (const TerrainSource &source : datasets.front().sources)
    primary_coverage.push_back(source.key);
  return TerrainCatalogue(
      datasets.front().grid,
      std::move(sources),
      observer,
      std::move(primary_coverage),
      std::move(datasets),
      frame
  );
}

TerrainCatalogue TerrainCatalogue::discover(
    const std::filesystem::path &tile_dir,
    const ObserverLocation &observer,
    float max_distance,
    uint32_t max_tile_count,
    bool allow_observer_fallback
) {
  std::vector<TerrainDataset> datasets =
      discover_terrain_datasets(std::array<TerrainDatasetConfig, 1>{{{tile_dir, 0.0}}});
  if (std::any_of(
          datasets.front().sources.begin(),
          datasets.front().sources.end(),
          [](const TerrainSource &source) { return bool(source.valid_cells); }
      ))
    return discover(
        std::array<TerrainDatasetConfig, 1>{{{tile_dir, 0.0}}},
        observer,
        max_distance,
        max_tile_count,
        allow_observer_fallback
    );
  const TileGrid grid = datasets.front().grid;
  std::vector<TerrainSource> available_sources = std::move(datasets.front().sources);
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
  return find_source(0U, key);
}

std::optional<uint32_t> TerrainCatalogue::find_source(uint32_t dataset_index, TileKey key) const {
  const auto found = source_index_by_key_.find({dataset_index, key});
  if (found == source_index_by_key_.end()) {
    return std::nullopt;
  }
  return found->second;
}

std::optional<TerrainLocation> TerrainCatalogue::locate_source(Coord navigation_coordinate) const {
  if (datasets_.empty()) {
    const TileKey key = tile_key_at(grid_, navigation_coordinate.x, navigation_coordinate.y);
    const auto source = find_source(key);
    return source.has_value() ? std::optional<TerrainLocation>{{*source, navigation_coordinate}}
                              : std::nullopt;
  }
  const std::array<Coord, 1> input = {navigation_coordinate};
  for (size_t dataset_index = 0; dataset_index < datasets_.size(); ++dataset_index) {
    const Coord native = navigation_to_dataset_[dataset_index]->apply(input).front();
    const TileKey key = tile_key_at(datasets_[dataset_index].grid, native.x, native.y);
    if (const auto source = find_source(static_cast<uint32_t>(dataset_index), key)) {
      const auto &coverage = sources_[*source].valid_cells;
      const auto &grid = datasets_[dataset_index].grid;
      if (coverage && !coverage->contains(
                          (native.x - grid.origin_x - double(key.column) * grid.width) *
                              coverage->cell_count / grid.width,
                          (native.y - grid.origin_y + double(key.row + 1) * grid.width) *
                              coverage->cell_count / grid.width
                      ))
        continue;
      return TerrainLocation{*source, native};
    }
  }
  return std::nullopt;
}

const std::vector<TerrainDataset> &TerrainCatalogue::datasets() const { return datasets_; }

const TerrainRenderFrame &TerrainCatalogue::render_frame() const {
  if (!render_frame_.has_value())
    throw std::logic_error("Legacy terrain catalogue has no explicit render frame");
  return *render_frame_;
}

Coord TerrainCatalogue::render_coordinate(Coord navigation_coordinate) const {
  if (!render_frame_.has_value() || datasets_.empty())
    return navigation_coordinate;
  const std::array<Coord, 1> coordinate = {navigation_coordinate};
  return render_frame_->project(datasets_.front().epsg_code, coordinate).front();
}

std::array<Coord, 2> TerrainCatalogue::render_basis(Coord navigation_coordinate) const {
  if (!render_frame_.has_value() || datasets_.empty())
    return {{{1.0, 0.0}, {0.0, 1.0}}};
  const double step = epsg_uses_projected_metres(datasets_.front().epsg_code) ? 10.0 : 1e-4;
  const std::array<Coord, 3> points = {
      navigation_coordinate,
      Coord{navigation_coordinate.x + step, navigation_coordinate.y},
      Coord{navigation_coordinate.x, navigation_coordinate.y + step},
  };
  const auto rendered = render_frame_->project(datasets_.front().epsg_code, points);
  std::array<Coord, 2> result;
  for (size_t axis = 0; axis < result.size(); ++axis) {
    const double dx = rendered[axis + 1U].x - rendered[0].x;
    const double dy = rendered[axis + 1U].y - rendered[0].y;
    const double length = std::hypot(dx, dy);
    if (!(length > 0.0) || !std::isfinite(length))
      throw std::runtime_error("Terrain render-frame basis is invalid");
    result[axis] = {dx / length, dy / length};
  }
  return result;
}

} // namespace panorama
