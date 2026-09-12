#include "terrain_cell_coverage.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <stdexcept>

namespace panorama {
bool TerrainCellCoverage::full() const {
  return rectangles.size() == 1 &&
         rectangles.front() == TerrainCoverageRect{0, 0, cell_count, cell_count};
}
bool TerrainCellCoverage::contains(double column, double row) const {
  if (!std::isfinite(column) || !std::isfinite(row) || column < 0 || row < 0 ||
      column >= cell_count || row >= cell_count)
    return false;
  const auto x = uint32_t(column), y = uint32_t(row);
  for (const auto &r : rectangles)
    if (x >= r.column && x - r.column < r.width && y >= r.row && y - r.row < r.height)
      return true;
  return false;
}
uint64_t
TerrainCellCoverage::covered_area(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) const {
  uint64_t area = 0;
  for (const auto &r : rectangles) {
    const uint32_t ax = std::max(x0, r.column), ay = std::max(y0, r.row);
    const uint32_t bx = std::min(x1, r.column + r.width), by = std::min(y1, r.row + r.height);
    if (ax < bx && ay < by)
      area += uint64_t(bx - ax) * (by - ay);
  }
  return area;
}
void TerrainCellCoverage::validate() const {
  if (!std::has_single_bit(cell_count) || rectangles.size() > uint64_t(cell_count) * cell_count)
    throw std::runtime_error("Invalid terrain coverage dimensions");
  if (full())
    return;
  std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> rows;
  for (const auto &r : rectangles) {
    if (!r.width || !r.height || r.column >= cell_count || r.row >= cell_count ||
        r.width > cell_count - r.column || r.height > cell_count - r.row)
      throw std::runtime_error("Terrain coverage rectangle is outside its tile");
    for (uint32_t y = r.row; y < r.row + r.height; ++y)
      rows[y].emplace_back(r.column, r.column + r.width);
  }
  for (auto &[row, runs] : rows) {
    (void)row;
    std::sort(runs.begin(), runs.end());
    for (size_t i = 1; i < runs.size(); ++i)
      if (runs[i].first < runs[i - 1].second)
        throw std::runtime_error("Terrain coverage rectangles overlap");
  }
}
TerrainCellCoverage make_cell_coverage(uint32_t cells, std::span<const uint8_t> usable) {
  if (usable.size() != uint64_t(cells) * cells)
    throw std::invalid_argument("Terrain coverage mask has the wrong size");
  TerrainCellCoverage result{cells, {}};
  std::map<std::pair<uint32_t, uint32_t>, size_t> active;
  for (uint32_t y = 0; y < cells; ++y) {
    std::map<std::pair<uint32_t, uint32_t>, size_t> next;
    for (uint32_t x = 0; x < cells;) {
      if (!usable[size_t(y) * cells + x]) {
        ++x;
        continue;
      }
      const uint32_t begin = x++;
      while (x < cells && usable[size_t(y) * cells + x])
        ++x;
      const auto run = std::pair{begin, x};
      if (const auto prior = active.find(run); prior != active.end()) {
        ++result.rectangles[prior->second].height;
        next.emplace(run, prior->second);
      } else {
        next.emplace(run, result.rectangles.size());
        result.rectangles.push_back({begin, y, x - begin, 1});
      }
    }
    active = std::move(next);
  }
  return result;
}
} // namespace panorama
