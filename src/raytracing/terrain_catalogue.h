#pragma once

#include "raytrace_setup.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
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

/// One available prepared-terrain file, grid location, and optional culling bound.
struct TerrainSource {
  TileKey key;
  std::filesystem::path path;
  std::optional<float> maximum_elevation;
};

/// A finite, indexed catalogue of terrain sources relevant to one render.
///
/// The catalogue scans a prepared-tile directory once, derives its physical
/// grid from GeoTIFF metadata or the custom tile header, attaches any maxima
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
      uint32_t max_tile_count
  );

  /// Return the shared grid used to locate all catalogue source tiles.
  [[nodiscard]] const TileGrid &grid() const;

  /// Return the source containing the observer, which always occupies index zero.
  [[nodiscard]] const TerrainSource &origin() const;

  /// Return all retained sources in stable scheduler order.
  [[nodiscard]] const std::vector<TerrainSource> &sources() const;

  /// Return a conservative upper bound for the retained terrain, when every
  /// source published a manifest maximum.
  [[nodiscard]] std::optional<float> maximum_elevation() const;

  /// Return a source index for a grid key, or no value when coverage is absent.
  [[nodiscard]] std::optional<uint32_t> find_source(TileKey key) const;

private:
  /// Construct an already validated, indexable catalogue.
  TerrainCatalogue(TileGrid grid, std::vector<TerrainSource> sources);

  TileGrid grid_;
  std::vector<TerrainSource> sources_;
  std::map<TileKey, uint32_t> source_index_by_key_;
  std::optional<float> maximum_elevation_;
};

/// Return the global rechunked tile key containing one projected coordinate.
[[nodiscard]] TileKey tile_key_at(const TileGrid &grid, double easting, double northing);

/// Return the shortest horizontal distance from an observer to one tile square.
[[nodiscard]] double
tile_minimum_distance(const TileGrid &grid, TileKey key, const ObserverLocation &observer);

} // namespace panorama
