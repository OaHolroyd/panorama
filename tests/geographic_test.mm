#include "coordinate_input.h"
#include "solar_position.h"
#include "terrain_transform.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace panorama;
using namespace panorama::app;
namespace {
void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
void check_position(LatLon a, LatLon b, double tolerance) {
  const auto delta = geographic_offset(a, b);
  require(std::hypot(delta.x, delta.y) < tolerance, "Geographic position differs");
}
} // namespace
int main() {
  try {
    for (const LatLon origin : {LatLon{46.0, 7.0},
                                LatLon{43.17, 16.44},
                                LatLon{72.0, 179.999},
                                LatLon{-45.0, -179.999}}) {
      const auto frame = TerrainRenderFrame::local_aeqd(origin);
      for (const Coord movement : {Coord{100, 0}, Coord{0, 100}, Coord{-3000, 4000}}) {
        const auto moved = offset_position(origin, movement.x, movement.y);
        const auto delta = geographic_offset(origin, moved);
        require(
            std::hypot(delta.x - movement.x, delta.y - movement.y) < 1e-5,
            "Movement lost metre units or true-north direction"
        );
        const auto xy = frame.project(moved);
        require(
            std::hypot(xy.x - movement.x, xy.y - movement.y) < 1e-5,
            "AEQD axes disagree with geographic movement"
        );
        check_position(frame.unproject(xy), moved, 1e-5);
      }
      // The frame survives movement; true-north axes must rotate at the new observer.
      const auto moved = offset_position(origin, 100000, 200000);
      const auto north = offset_position(moved, 0, 100);
      const auto delta = frame.offset(moved, north);
      require(
          delta.y > 99 && std::abs(delta.x) < .01,
          "True north changed after observer relocation"
      );
    }
    require(
        offset_position({0, 179.999}, 1000, 0).lon < -179,
        "Eastward movement did not cross the date line"
    );
    require(
        offset_position({0, -179.999}, -1000, 0).lon > 179,
        "Westward movement did not cross the date line"
    );
    require(
        !valid_lat_lon({91, 0}) && !valid_lat_lon({0, 181}) &&
            !valid_lat_lon({std::numeric_limits<double>::quiet_NaN(), 0}),
        "Invalid geographic coordinates accepted"
    );
    const auto swiss = Crs(CrsId::SwissLv95);
    const auto global = Crs(CrsId::Wgs84);
    const auto lv95 =
        parse_coordinate_input("2621451.3, 1105565.9", global, CoordinateInputSystem::SwissLv95);
    check_position(lv95.geographic, {46.1012605320838, 7.71604367731172}, .01);
    const auto british = parse_coordinate_input(
        "NG 90716 59877",
        global,
        CoordinateInputSystem::BritishNationalGrid
    );
    check_position(
        british.geographic,
        Crs(CrsId::BritishNationalGrid).to_lat_lon({190716.5, 859877.5}),
        1
    );
    // Explicit global input must work even if the active terrain uses a regional CRS.
    const auto far = parse_coordinate_input("-45, 179.9", swiss, CoordinateInputSystem::Wgs84);
    check_position(far.geographic, {-45, 179.9}, 1e-5);
    check_position(
        parse_coordinate_input("43.17, 16.44", global, CoordinateInputSystem::Terrain).geographic,
        {43.17, 16.44},
        1e-5
    );
    const auto noon = solar_position({40, 0}, {2026, 3, 20, 12, 0});
    require(
        noon.azimuth > 3.0 && noon.azimuth < 3.2 && noon.elevation > .8,
        "Solar azimuth is not relative to true north"
    );
    require(
        daylight_times({80, 0}, {2026, 6, 21, 0, 0}).state == DaylightState::PolarDay &&
            daylight_times({80, 0}, {2026, 12, 21, 0, 0}).state == DaylightState::PolarNight,
        "Polar daylight calculation lost geographic latitude"
    );
    std::puts("Geographic movement, date line, input conversion and solar direction passed.");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
