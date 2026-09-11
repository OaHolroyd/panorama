#pragma once

#include "metal_tile.h"
#include "raytrace_config.h"
#include "terrain_transform.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace panorama {

/// Key identifying one rechunked tile in the global row/column grid.
struct TileKey {
  int64_t row;
  int64_t column;

  /// Order keys by row and then column for associative containers.
  [[nodiscard]] bool operator<(const TileKey &other) const;

  /// Return whether two keys refer to the same rechunked tile.
  [[nodiscard]] bool operator==(const TileKey &other) const;
};

/// Global rechunk-grid origin and fixed physical width of each square tile.
struct TileGrid {
  double origin_x;
  double origin_y;
  double width;
};

/// Complete prepared-data footprint, independent of one observer's trace radius.
struct TerrainCoverage {
  TileGrid grid;
  std::vector<TileKey> tiles;
};

/// One available prepared-terrain file, grid location, and optional culling bound.
struct TerrainSource {
  TileKey key;
  std::filesystem::path path;
  std::optional<float> maximum_elevation;
  /// Independently loadable terrain representations, including LOD 1.
  uint32_t lod_count = 1U;
  std::optional<float> minimum_elevation = std::nullopt;
  /// Position of the owning dataset in the configured priority order.
  uint32_t dataset_index = 0U;
  /// Tile-local logical-cell transforms into the selected render frame.
  std::vector<TerrainTransformPatch> transform_patches;
  /// Full source footprint with shared edges, used to prove uninterrupted
  /// physical data coverage independently of cell-rounded ownership.
  std::vector<TerrainCoveragePolygon> coverage_polygons;
  /// Cell-rounded ownership regions for bounded streaming. Empty means the
  /// complete source footprint is owned. These must not define data coverage:
  /// rounding between different source grids can leave sub-cell slivers.
  std::vector<TerrainCoveragePolygon> ownership_polygons;
  /// Largest transformed LOD-1 cell axis, used by per-source LOD policy.
  double effective_cell_size_metres = 0.0;
  double vertical_offset_metres = 0.0;
  /// Null for legacy tiles; otherwise also identifies the reserved-zero encoding.
  std::shared_ptr<const TerrainCellCoverage> valid_cells;
};

/// One independently gridded prepared dataset before observer filtering and
/// cross-dataset ownership resolution.
struct TerrainDataset {
  TerrainDatasetConfig config;
  uint32_t index;
  TileGrid grid;
  uint32_t epsg_code;
  uint32_t cell_count;
  uint32_t level_count;
  uint32_t lod_count;
  MetalTileSampleType sample_type;
  MetalTileCompression compression;
  std::vector<TerrainSource> sources;
};

/// A navigation-frame point resolved into one source dataset's native grid.
struct TerrainLocation {
  uint32_t source_index;
  Coord native_coordinate;
};

/// Discover an ordered stack and enforce the payload/atlas layout shared by
/// all datasets. Native CRS and cell spacing may differ.
[[nodiscard]] std::vector<TerrainDataset>
discover_terrain_datasets(std::span<const TerrainDatasetConfig> configs);

/// Select the native projected frame for one metre dataset, otherwise centre
/// an AEQD metre frame on the observer expressed in the first dataset's CRS.
[[nodiscard]] TerrainRenderFrame
select_terrain_render_frame(std::span<const TerrainDataset> datasets, Coord navigation_observer);

/// A finite, indexed catalogue of terrain sources relevant to one render.
///
/// The catalogue scans a prepared-tile directory once, derives its physical
/// grid from one `.ptile` header, attaches any maxima
/// published in the directory manifest, retains sources within the configured
/// horizontal range, and maps their stable grid keys to source indices. It is
/// immutable thereafter, so foreground scheduling and worker threads can
/// safely share its source vector without synchronisation.
class TerrainCatalogue {
public:
  /// Infer the prepared grid, discover sources, and put the observer tile first.
  [[nodiscard]] static TerrainCatalogue discover(
      const std::filesystem::path &tile_dir,
      const ObserverLocation &observer,
      float max_distance,
      uint32_t max_tile_count,
      bool allow_observer_fallback = false
  );

  /// Discover, transform, and observer-filter an ordered terrain stack.
  [[nodiscard]] static TerrainCatalogue discover(
      std::span<const TerrainDatasetConfig> datasets,
      const ObserverLocation &observer,
      float max_distance,
      uint32_t max_tile_count,
      bool allow_observer_fallback = false
  );

  /// Return the shared grid used to locate all catalogue source tiles.
  [[nodiscard]] const TileGrid &grid() const;

  /// Return the source containing the observer, which always occupies index zero.
  [[nodiscard]] const TerrainSource &origin() const;

  /// Return the requested observer, or the dataset-derived fallback selected during discovery.
  [[nodiscard]] const ObserverLocation &observer() const;

  /// Return every available tile footprint, including tiles outside the trace radius.
  [[nodiscard]] const TerrainCoverage &coverage() const;

  /// Return all retained sources in stable scheduler order.
  [[nodiscard]] const std::vector<TerrainSource> &sources() const;

  /// Return a conservative upper bound for the retained terrain, when every
  /// source published a manifest maximum.
  [[nodiscard]] std::optional<float> maximum_elevation() const;

  /// Return a source index for a grid key, or no value when coverage is absent.
  [[nodiscard]] std::optional<uint32_t> find_source(TileKey key) const;
  [[nodiscard]] std::optional<uint32_t> find_source(uint32_t dataset_index, TileKey key) const;
  /// Resolve a point in the first dataset's navigation CRS, respecting the
  /// configured dataset priority.
  [[nodiscard]] std::optional<TerrainLocation> locate_source(Coord navigation_coordinate) const;
  [[nodiscard]] const std::vector<TerrainDataset> &datasets() const;
  [[nodiscard]] const TerrainRenderFrame &render_frame() const;
  [[nodiscard]] Coord render_coordinate(Coord navigation_coordinate) const;
  /// Unit render-frame directions corresponding to the navigation CRS's
  /// positive easting and northing axes at a point.
  [[nodiscard]] std::array<Coord, 2> render_basis(Coord navigation_coordinate) const;

private:
  /// Construct an already validated, indexable catalogue.
  TerrainCatalogue(
      TileGrid grid,
      std::vector<TerrainSource> sources,
      ObserverLocation observer,
      std::vector<TileKey> coverage_tiles,
      std::vector<TerrainDataset> datasets = {},
      std::optional<TerrainRenderFrame> render_frame = std::nullopt
  );

  struct SourceKey {
    uint32_t dataset;
    TileKey tile;
    [[nodiscard]] bool operator<(const SourceKey &other) const;
  };

  TileGrid grid_;
  std::vector<TerrainSource> sources_;
  ObserverLocation observer_;
  TerrainCoverage coverage_;
  std::map<SourceKey, uint32_t> source_index_by_key_;
  std::optional<float> maximum_elevation_;
  std::vector<TerrainDataset> datasets_;
  std::vector<std::unique_ptr<CoordinateTransform>> navigation_to_dataset_;
  std::optional<TerrainRenderFrame> render_frame_;
};

/// Return the global rechunked tile key containing one projected coordinate.
[[nodiscard]] TileKey tile_key_at(const TileGrid &grid, double easting, double northing);

/// Return the shortest horizontal distance from an observer to one tile square.
[[nodiscard]] double
tile_minimum_distance(const TileGrid &grid, TileKey key, const ObserverLocation &observer);

} // namespace panorama
