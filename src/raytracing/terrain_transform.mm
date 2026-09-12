#include "terrain_transform.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace panorama {
namespace {

[[nodiscard]] Coord native_coordinate(const MetalTileHeader &header, double column, double row) {
  return {header.lower_left_x + column * header.cell_size,
          header.lower_left_y + row * header.cell_size};
}

[[nodiscard]] TerrainTileTransform make_transform(
    const MetalTileHeader &header,
    const CoordinateTransform &projector,
    double minimum_column,
    double minimum_row,
    double maximum_column,
    double maximum_row
) {
  const double centre_column = 0.5 * (minimum_column + maximum_column);
  const double centre_row = 0.5 * (minimum_row + maximum_row);
  const std::array<Coord, 3> fit_native = {
      native_coordinate(header, centre_column, centre_row),
      native_coordinate(header, centre_column + 1.0, centre_row),
      native_coordinate(header, centre_column, centre_row + 1.0),
  };
  const auto fit = projector.apply(fit_native);
  const Coord column_step = {fit[1].x - fit[0].x, fit[1].y - fit[0].y};
  const Coord row_step = {fit[2].x - fit[0].x, fit[2].y - fit[0].y};
  TerrainTileTransform result = {
      {fit[0].x - centre_column * column_step.x - centre_row * row_step.x,
       fit[0].y - centre_column * column_step.y - centre_row * row_step.y},
      column_step,
      row_step,
      0.0,
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity()},
  };
  if (!std::isfinite(result.determinant()) || std::abs(result.determinant()) < 1e-12)
    throw std::runtime_error("CRS produced a singular terrain tile transform");

  const std::array<Coord, 8> logical = {
      Coord{minimum_column, minimum_row},
      Coord{maximum_column, minimum_row},
      Coord{minimum_column, maximum_row},
      Coord{maximum_column, maximum_row},
      Coord{centre_column, minimum_row},
      Coord{centre_column, maximum_row},
      Coord{minimum_column, centre_row},
      Coord{maximum_column, centre_row},
  };
  std::vector<Coord> native;
  native.reserve(logical.size());
  for (const Coord point : logical)
    native.push_back(native_coordinate(header, point.x, point.y));
  const auto exact = projector.apply(native);
  for (size_t index = 0; index < logical.size(); ++index) {
    const Coord approximate = result.apply(logical[index].x, logical[index].y);
    result.maximum_residual_metres = std::max(
        result.maximum_residual_metres,
        std::hypot(approximate.x - exact[index].x, approximate.y - exact[index].y)
    );
    result.bounds[0] = std::min(result.bounds[0], approximate.x);
    result.bounds[1] = std::min(result.bounds[1], approximate.y);
    result.bounds[2] = std::max(result.bounds[2], approximate.x);
    result.bounds[3] = std::max(result.bounds[3], approximate.y);
  }
  return result;
}

} // namespace

TerrainRenderFrame TerrainRenderFrame::fixed(uint32_t epsg_code) {
  if (!epsg_uses_projected_metres(epsg_code))
    throw std::invalid_argument("Fixed terrain render frame must be a projected metre CRS");
  return {Kind::FixedEpsg, epsg_code, {0.0, 0.0}};
}

TerrainRenderFrame TerrainRenderFrame::local_aeqd(LatLon anchor) {
  const std::array<Coord, 1> centre = {{{anchor.lon, anchor.lat}}};
  (void)transform_coordinates_to_local_aeqd(4326U, anchor, centre);
  return {Kind::LocalAzimuthalEquidistant, 0U, anchor};
}

std::vector<Coord>
TerrainRenderFrame::project(uint32_t source_epsg, std::span<const Coord> coordinates) const {
  if (kind == Kind::FixedEpsg)
    return transform_coordinates(source_epsg, fixed_epsg, coordinates);
  return transform_coordinates_to_local_aeqd(source_epsg, anchor, coordinates);
}

std::vector<Coord>
TerrainRenderFrame::unproject(uint32_t destination_epsg, std::span<const Coord> coordinates) const {
  if (kind == Kind::FixedEpsg)
    return transform_coordinates(fixed_epsg, destination_epsg, coordinates);
  return transform_coordinates_from_local_aeqd(destination_epsg, anchor, coordinates);
}

CoordinateTransform TerrainRenderFrame::projector(uint32_t source_epsg) const {
  return kind == Kind::FixedEpsg ? CoordinateTransform(source_epsg, fixed_epsg)
                                 : CoordinateTransform(source_epsg, anchor);
}

CoordinateTransform TerrainRenderFrame::unprojector(uint32_t destination_epsg) const {
  return kind == Kind::FixedEpsg ? CoordinateTransform(fixed_epsg, destination_epsg)
                                 : CoordinateTransform(anchor, destination_epsg);
}

Coord TerrainTileTransform::apply(double column, double row) const {
  return {origin.x + column * column_step.x + row * row_step.x,
          origin.y + column * column_step.y + row * row_step.y};
}

double TerrainTileTransform::determinant() const {
  return column_step.x * row_step.y - column_step.y * row_step.x;
}

Coord TerrainTileTransform::inverse(Coord coordinate) const {
  const double det = determinant();
  if (!std::isfinite(det) || std::abs(det) <= std::numeric_limits<double>::min())
    throw std::runtime_error("Terrain tile transform is singular");
  const double x = coordinate.x - origin.x;
  const double y = coordinate.y - origin.y;
  return {(x * row_step.y - y * row_step.x) / det, (y * column_step.x - x * column_step.y) / det};
}

double TerrainTileTransform::maximum_cell_size_metres() const {
  return std::max(std::hypot(column_step.x, column_step.y), std::hypot(row_step.x, row_step.y));
}

TerrainTileTransform
make_terrain_tile_transform(const MetalTileHeader &header, const TerrainRenderFrame &frame) {
  if (header.epsg_code == 0U || header.cell_count == 0U || !std::isfinite(header.cell_size) ||
      header.cell_size <= 0.0)
    throw std::invalid_argument("Cannot transform invalid terrain tile geometry");
  const CoordinateTransform projector = frame.projector(header.epsg_code);
  return make_transform(header, projector, 0.0, 0.0, header.cell_count, header.cell_count);
}

std::vector<TerrainTransformPatch> make_terrain_transform_patches(
    const MetalTileHeader &header,
    const TerrainRenderFrame &frame,
    double maximum_residual_metres
) {
  if (!std::isfinite(maximum_residual_metres) || maximum_residual_metres <= 0.0)
    throw std::invalid_argument("Terrain transform residual bound must be positive and finite");
  if (header.epsg_code == 0U || header.cell_count == 0U || !std::isfinite(header.cell_size) ||
      header.cell_size <= 0.0)
    throw std::invalid_argument("Cannot transform invalid terrain tile geometry");
  const CoordinateTransform projector = frame.projector(header.epsg_code);
  std::vector<std::array<uint32_t, 4>> pending = {{0U, 0U, header.cell_count, header.cell_count}};
  std::vector<TerrainTransformPatch> result;
  while (!pending.empty()) {
    const auto region = pending.back();
    pending.pop_back();
    TerrainTileTransform transform = make_transform(
        header,
        projector,
        region[0],
        region[1],
        region[0] + region[2],
        region[1] + region[3]
    );
    if (transform.maximum_residual_metres <= maximum_residual_metres ||
        (region[2] == 1U && region[3] == 1U)) {
      result.push_back({region[0], region[1], region[2], region[3], std::move(transform)});
      continue;
    }
    const uint32_t left = region[2] / 2U;
    const uint32_t right = region[2] - left;
    const uint32_t bottom = region[3] / 2U;
    const uint32_t top = region[3] - bottom;
    pending.push_back({region[0], region[1], left, bottom});
    pending.push_back({region[0] + left, region[1], right, bottom});
    pending.push_back({region[0], region[1] + bottom, left, top});
    pending.push_back({region[0] + left, region[1] + bottom, right, top});
  }
  return result;
}

std::vector<TerrainCoveragePolygon> make_terrain_coverage_polygons(
    const MetalTileHeader &header,
    const TerrainRenderFrame &frame,
    std::span<const TerrainTransformPatch> ownership,
    uint32_t maximum_cells_per_side,
    std::span<const std::array<uint32_t, 2>> boundary_junctions
) {
  if (maximum_cells_per_side == 0U)
    throw std::invalid_argument("Coverage triangle side must be positive");
  if (header.epsg_code == 0U || header.cell_count == 0U || !std::isfinite(header.cell_size) ||
      header.cell_size <= 0.0)
    throw std::invalid_argument("Cannot transform invalid terrain tile geometry");

  const CoordinateTransform projector = frame.projector(header.epsg_code);
  std::vector<TerrainCoveragePolygon> result;

  // A geographic grid line becomes curved after projection. If one side of
  // a shared edge includes a T-junction but the other uses a longer chord,
  // the two polygons leave a false gap. Split both at the same grid points.
  std::map<uint32_t, std::vector<uint32_t>> horizontal, vertical;
  const auto add_junction = [&](uint32_t column, uint32_t row) {
    horizontal[row].push_back(column);
    vertical[column].push_back(row);
  };
  for (const auto &region : ownership) {
    for (uint32_t column : {region.minimum_column, region.minimum_column + region.cell_width})
      for (uint32_t row : {region.minimum_row, region.minimum_row + region.cell_height})
        add_junction(column, row);
  }
  for (const auto &point : boundary_junctions)
    add_junction(point[0], point[1]);
  for (auto *edges : {&horizontal, &vertical})
    for (auto &[line, cuts] : *edges) {
      (void)line;
      std::sort(cuts.begin(), cuts.end());
      cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    }
  // Adjacent coverage rectangles share most boundary vertices. Project each
  // grid point once, in one batch, then reuse it in the original perimeters.
  std::unordered_map<uint64_t, size_t> vertex_indices;
  std::vector<Coord> native;
  std::vector<std::vector<size_t>> perimeters;
  perimeters.reserve(ownership.size());
  for (const TerrainTransformPatch &region : ownership) {
    const uint32_t x0 = region.minimum_column, x1 = x0 + region.cell_width;
    const uint32_t y0 = region.minimum_row, y1 = y0 + region.cell_height;
    std::vector<size_t> perimeter;
    const auto append_edge =
        [&](bool is_horizontal, uint32_t fixed, uint32_t minimum, uint32_t maximum, bool reverse) {
          std::vector<uint32_t> coordinates = {minimum};
          for (uint32_t value = minimum; value < maximum;) {
            value =
                std::min(maximum, ((value / maximum_cells_per_side) + 1U) * maximum_cells_per_side);
            coordinates.push_back(value);
          }
          const auto &cuts = (is_horizontal ? horizontal : vertical).at(fixed);
          coordinates.insert(
              coordinates.end(),
              std::lower_bound(cuts.begin(), cuts.end(), minimum),
              std::upper_bound(cuts.begin(), cuts.end(), maximum)
          );
          std::sort(coordinates.begin(), coordinates.end());
          coordinates.erase(std::unique(coordinates.begin(), coordinates.end()), coordinates.end());
          if (reverse)
            std::reverse(coordinates.begin(), coordinates.end());
          for (uint32_t value : coordinates) {
            const uint32_t column = is_horizontal ? value : fixed;
            const uint32_t row = is_horizontal ? fixed : value;
            const Coord point = native_coordinate(header, column, row);
            const auto [vertex, inserted] =
                vertex_indices.try_emplace((uint64_t(column) << 32U) | row, native.size());
            if (inserted)
              native.push_back(point);
            if (perimeter.empty() || native[perimeter.back()].x != point.x ||
                native[perimeter.back()].y != point.y)
              perimeter.push_back(vertex->second);
          }
        };
    append_edge(true, y0, x0, x1, false);
    append_edge(false, x1, y0, y1, false);
    append_edge(true, y1, x0, x1, true);
    append_edge(false, x0, y0, y1, true);
    perimeter.pop_back(); // The last edge closes at the first vertex.
    perimeters.push_back(std::move(perimeter));
  }
  if (native.empty())
    return result;
  const std::vector<Coord> vertices = projector.apply(native);
  for (const auto &perimeter : perimeters) {
    std::vector<Coord> projected;
    projected.reserve(perimeter.size());
    for (size_t index : perimeter)
      projected.push_back(vertices[index]);
    double signed_area = 0.0;
    for (size_t index = 0; index < projected.size(); ++index) {
      const Coord a = projected[index];
      const Coord b = projected[(index + 1U) % projected.size()];
      signed_area += a.x * b.y - a.y * b.x;
    }
    if (std::abs(signed_area) > std::numeric_limits<double>::epsilon()) {
      if (signed_area < 0.0)
        std::reverse(projected.begin(), projected.end());
      result.push_back({std::move(projected)});
    }
  }
  return result;
}

} // namespace panorama
