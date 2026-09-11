#pragma once

#include "crs.h"
#include "metal_tile.h"

#include <array>
#include <cstdint>
#include <vector>

namespace panorama {

/// Metric coordinate system used by terrain BVHs. A single projected-metre
/// dataset can retain its native CRS; geographic and mixed stacks use AEQD.
struct TerrainRenderFrame {
  enum class Kind : uint32_t { FixedEpsg, LocalAzimuthalEquidistant };

  Kind kind;
  uint32_t fixed_epsg;
  LatLon anchor;

  [[nodiscard]] static TerrainRenderFrame fixed(uint32_t epsg_code);
  [[nodiscard]] static TerrainRenderFrame local_aeqd(LatLon anchor);
  [[nodiscard]] std::vector<Coord>
  project(uint32_t source_epsg, std::span<const Coord> coordinates) const;
  [[nodiscard]] std::vector<Coord>
  unproject(uint32_t destination_epsg, std::span<const Coord> coordinates) const;
  [[nodiscard]] CoordinateTransform projector(uint32_t source_epsg) const;
  [[nodiscard]] CoordinateTransform unprojector(uint32_t destination_epsg) const;
};

/// Affine map from a tile's logical `(column, row)` sample coordinates to the
/// navigation CRS. Coefficients are expressed per logical cell, which keeps
/// geographic source coordinates out of Float32 GPU geometry.
struct TerrainTileTransform {
  Coord origin;
  Coord column_step;
  Coord row_step;
  /// Largest difference from the exact CRS transform at validation samples.
  double maximum_residual_metres;
  /// Conservative axis-aligned bounds of the affine tile footprint.
  std::array<double, 4> bounds;

  [[nodiscard]] Coord apply(double column, double row) const;
  [[nodiscard]] Coord inverse(Coord coordinate) const;
  [[nodiscard]] double determinant() const;
  [[nodiscard]] double maximum_cell_size_metres() const;
};

/// Linearise one prepared tile's native CRS into `destination_epsg`. The fit
/// is centred on the tile and the residual is measured at its corners and edge
/// midpoints, ready for later subdivision when a whole tile is too nonlinear.
[[nodiscard]] TerrainTileTransform
make_terrain_tile_transform(const MetalTileHeader &header, const TerrainRenderFrame &frame);

/// One independently linearised rectangular portion of a source tile.
struct TerrainTransformPatch {
  uint32_t minimum_column;
  uint32_t minimum_row;
  uint32_t cell_width;
  uint32_t cell_height;
  TerrainTileTransform transform;
};

/// Subdivide a tile until every local affine approximation meets the requested
/// horizontal residual. Patch edges remain on logical cell boundaries.
[[nodiscard]] std::vector<TerrainTransformPatch> make_terrain_transform_patches(
    const MetalTileHeader &header,
    const TerrainRenderFrame &frame,
    double maximum_residual_metres = 1.0
);

} // namespace panorama
