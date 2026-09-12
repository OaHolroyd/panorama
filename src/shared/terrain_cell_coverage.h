#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace panorama {

/// Disjoint rectangles of usable cells, in the tile's south-to-north grid.
/// Disjoint usable-cell rectangle in the native grid, counted from south-west.
struct TerrainCoverageRect {
  uint32_t column, row, width, height;
  bool operator==(const TerrainCoverageRect &) const = default;
};

/// Full coverage is one N-by-N rectangle; an empty vector has no usable cells.
struct TerrainCellCoverage {
  uint32_t cell_count = 0;
  std::vector<TerrainCoverageRect> rectangles;
  [[nodiscard]] bool full() const;
  [[nodiscard]] bool contains(double column, double row) const;
  [[nodiscard]] uint64_t covered_area(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) const;
  void validate() const;
  bool operator==(const TerrainCellCoverage &) const = default;
};

/// Merge equal runs on adjacent rows; complete tiles need just one rectangle.
[[nodiscard]] TerrainCellCoverage
make_cell_coverage(uint32_t cells, std::span<const uint8_t> usable_cells);

} // namespace panorama
