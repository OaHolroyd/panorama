#pragma once

#include "coordinate_input.h"
#include "peak_catalogue.h"

namespace panorama::app {

/// Incomplete or invalid coordinate syntax stays in the coordinate branch,
/// rather than starting a place-name search while the user is still typing.
[[nodiscard]] bool is_coordinate_search(std::string_view input);
/// Ordered fallback: WGS84 first, then the other recognised coordinate formats.
[[nodiscard]] ParsedCoordinateInput
parse_search_coordinate(std::string_view input, const Crs &terrain_crs);
/// Case- and accent-insensitive matches, ordered by exact name, prefix,
/// substring, then distance. IDs refer to the original loaded catalogue.
[[nodiscard]] std::vector<uint32_t> search_peaks(
    const PeakCatalogue &catalogue,
    std::string_view query,
    LatLon observer,
    size_t limit = 6
);

} // namespace panorama::app
