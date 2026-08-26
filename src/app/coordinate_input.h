#pragma once

#include "crs.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace panorama::app {

/// Coordinate systems currently offered by the Position inspector. The
/// dataset option means the projected CRS declared by the loaded terrain.
enum class CoordinateInputSystem : uint8_t {
  Wgs84,
  SwissLv95,
  BritishNationalGrid,
  Terrain,
};

/// One user-entered location transformed into both WGS 84 and the active
/// terrain grid. `source_name` describes the interpretation shown in the UI.
struct ParsedCoordinateInput {
  Coord projected;
  LatLon geographic;
  std::string source_name;
  CoordinateInputSystem system = CoordinateInputSystem::Wgs84;
};

/// Parse input as one explicitly selected system. Recognised textual prefixes
/// may be retained when they agree with `system`; a conflicting prefix is an
/// error rather than being silently ignored.
[[nodiscard]] ParsedCoordinateInput parse_coordinate_input(
    std::string_view input,
    const Crs &terrain_crs,
    CoordinateInputSystem system
);

/// Return every plausible interpretation for Auto mode. Distinctive syntax or
/// an explicit prefix produces one candidate; a bare numeric pair may produce
/// several candidates for the inspector to disambiguate using terrain coverage
/// or a user selection.
[[nodiscard]] std::vector<ParsedCoordinateInput>
detect_coordinate_inputs(std::string_view input, const Crs &terrain_crs);

} // namespace panorama::app
