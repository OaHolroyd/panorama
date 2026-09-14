#include "visibility_mask.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace panorama::app {
namespace {
constexpr double world = 268435456.0;
constexpr double circumference = 2.0 * std::numbers::pi * 6378137.0;
} // namespace

VisibilityProjectionGrid make_visibility_projection_grid(
    const VisibilityMapRegion &region,
    uint32_t width,
    uint32_t height
) {
  auto inverse = region.frame.projector(3857U);
  auto forward = region.frame.unprojector(3857U);
  const Coord observer = region.frame.project(region.observer);
  const auto transform =
      [](const CoordinateTransform &projection, std::vector<double> &x, std::vector<double> &y) {
        std::vector<Coord> input;
        input.reserve(x.size());
        for (size_t i = 0; i < x.size(); ++i)
          input.push_back({x[i], y[i]});
        const auto output = projection.apply(input);
        for (size_t i = 0; i < x.size(); ++i) {
          x[i] = output[i].x;
          y[i] = output[i].y;
        }
      };

  // Sample the displayed rectangle in one CRS call. Pad its terrain-space
  // envelope for curved edges, and restrict it to the trace's possible hits.
  std::vector<double> x, y;
  for (int row = 0; row <= 8; ++row) {
    for (int col = 0; col <= 8; ++col) {
      x.push_back(((region.x + region.width * col / 8.0) / world - 0.5) * circumference);
      y.push_back((0.5 - (region.y + region.height * row / 8.0) / world) * circumference);
    }
  }
  double left = observer.x - region.max_distance;
  double right = observer.x + region.max_distance;
  double bottom = observer.y - region.max_distance;
  double top = observer.y + region.max_distance;
  try {
    transform(inverse, x, y);
    const auto [xmin, xmax] = std::minmax_element(x.begin(), x.end());
    const auto [ymin, ymax] = std::minmax_element(y.begin(), y.end());
    const double dx = (*xmax - *xmin) * 0.05 + 1, dy = (*ymax - *ymin) * 0.05 + 1;
    left = std::max(left, *xmin - dx);
    right = std::min(right, *xmax + dx);
    bottom = std::max(bottom, *ymin - dy);
    top = std::min(top, *ymax + dy);
  } catch (const std::runtime_error &) {
    // A map spanning the opposite hemisphere cannot bound this local frame.
  }
  if (right <= left || top <= bottom) {
    // The map is outside the observer's trace radius. A grid outside that
    // radius rejects all points before reading these transparent coordinates.
    return {region.max_distance * 2,
            region.max_distance * 2,
            1,
            1,
            2,
            std::vector<std::array<float, 2>>(4, {-10, -10})};
  }

  // Sample a fine grid and compare every midpoint against the coarser
  // bilinear grid. Normally eight cells suffice; refinement remains bounded.
  for (uint32_t cells = 8; cells <= 128; cells *= 2) {
    const uint32_t fine = cells * 2 + 1;
    x.clear();
    y.clear();
    for (uint32_t row = 0; row < fine; ++row) {
      for (uint32_t col = 0; col < fine; ++col) {
        x.push_back(left + (right - left) * col / (fine - 1));
        y.push_back(bottom + (top - bottom) * row / (fine - 1));
      }
    }
    transform(forward, x, y);
    for (size_t i = 0; i < x.size(); ++i) {
      double map_x = (x[i] / circumference + 0.5) * world;
      map_x += std::round((region.x + 0.5 * region.width - map_x) / world) * world;
      x[i] = (map_x - region.x) * width / region.width;
      y[i] = ((0.5 - y[i] / circumference) * world - region.y) * height / region.height;
    }
    double error = 0;
    for (uint32_t row = 0; row < fine; ++row) {
      for (uint32_t col = 0; col < fine; ++col) {
        const uint32_t r0 = std::min(row / 2, cells - 1) * 2;
        const uint32_t c0 = std::min(col / 2, cells - 1) * 2;
        const double tx = (col - c0) * 0.5, ty = (row - r0) * 0.5;
        const auto interpolate = [&](const std::vector<double> &v) {
          return std::lerp(
              std::lerp(v[r0 * fine + c0], v[r0 * fine + c0 + 2], tx),
              std::lerp(v[(r0 + 2) * fine + c0], v[(r0 + 2) * fine + c0 + 2], tx),
              ty
          );
        };
        error = std::max(
            error,
            std::hypot(interpolate(x) - x[row * fine + col], interpolate(y) - y[row * fine + col])
        );
      }
    }
    if (error > 0.25 && cells < 128)
      continue;
    if (!std::isfinite(error) || error > 0.5)
      throw std::runtime_error("Minimap projection exceeds half-pixel error budget");
    VisibilityProjectionGrid grid = {left - observer.x,
                                     bottom - observer.y,
                                     (right - left) / cells,
                                     (top - bottom) / cells,
                                     cells + 1,
                                     {}};
    for (uint32_t row = 0; row < fine; row += 2)
      for (uint32_t col = 0; col < fine; col += 2)
        grid.pixels.push_back({float(x[row * fine + col]), float(y[row * fine + col])});
    return grid;
  }
  throw std::logic_error("Unreachable minimap grid refinement");
}
} // namespace panorama::app
