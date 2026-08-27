#include "coordinate_input.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace panorama::app {
namespace {

[[nodiscard]] std::string trim_copy(std::string_view input) {
  const auto whitespace = [](unsigned char character) { return std::isspace(character) != 0; };
  const auto first = std::find_if_not(input.begin(), input.end(), whitespace);
  const auto last = std::find_if_not(input.rbegin(), input.rend(), whitespace).base();
  return first < last ? std::string(first, last) : std::string();
}

[[nodiscard]] std::string uppercase(std::string input) {
  std::transform(input.begin(), input.end(), input.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  return input;
}

/// Remove a case-normalised prefix followed by whitespace, a colon, or the end
/// of the string. Requiring that boundary prevents names such as `OSGB` from
/// being mistaken for the shorter `OS` prefix.
[[nodiscard]] bool strip_prefix(std::string &input, std::string_view prefix) {
  if (!input.starts_with(prefix)) {
    return false;
  }
  if (input.size() > prefix.size() && input[prefix.size()] != ':' &&
      std::isspace(static_cast<unsigned char>(input[prefix.size()])) == 0) {
    return false;
  }
  size_t offset = prefix.size();
  if (offset < input.size() && input[offset] == ':') {
    offset++;
  }
  input = trim_copy(std::string_view(input).substr(offset));
  return true;
}

[[nodiscard]] std::optional<std::array<double, 2>> parse_number_pair(std::string_view input) {
  std::string normalised(input);
  std::replace(normalised.begin(), normalised.end(), ',', ' ');
  std::replace(normalised.begin(), normalised.end(), ';', ' ');
  const char *cursor = normalised.c_str();
  const auto skip_whitespace = [&cursor] {
    while (*cursor != '\0' && std::isspace(static_cast<unsigned char>(*cursor)) != 0) {
      cursor++;
    }
  };
  const auto parse_number = [&cursor, &skip_whitespace]() -> std::optional<double> {
    skip_whitespace();
    errno = 0;
    char *end = nullptr;
    const double value = std::strtod(cursor, &end);
    if (end == cursor || errno == ERANGE || !std::isfinite(value)) {
      return std::nullopt;
    }
    cursor = end;
    return value;
  };

  const std::optional<double> first = parse_number();
  const std::optional<double> second = parse_number();
  skip_whitespace();
  if (!first.has_value() || !second.has_value() || *cursor != '\0') {
    return std::nullopt;
  }
  return std::array<double, 2>{*first, *second};
}

[[nodiscard]] int national_grid_letter_index(char letter) {
  if (letter < 'A' || letter > 'Z' || letter == 'I') {
    return -1;
  }
  int index = letter - 'A';
  if (letter > 'I') {
    index--;
  }
  return index;
}

/// Decode conventional two-letter OS references. Removing separators first
/// makes spaced `NG 90716 59877` and compact `NG907598` forms equivalent.
[[nodiscard]] std::optional<Coord> parse_national_grid_reference(std::string_view input) {
  std::string compact;
  compact.reserve(input.size());
  for (unsigned char character : input) {
    if (std::isspace(character) == 0 && character != ',') {
      compact.push_back(static_cast<char>(character));
    }
  }
  if (compact.size() < 4U || std::isalpha(static_cast<unsigned char>(compact[0])) == 0 ||
      std::isalpha(static_cast<unsigned char>(compact[1])) == 0) {
    return std::nullopt;
  }
  const size_t digit_count = compact.size() - 2U;
  if (digit_count > 10U || digit_count % 2U != 0U ||
      !std::all_of(compact.begin() + 2, compact.end(), [](unsigned char character) {
        return std::isdigit(character) != 0;
      })) {
    throw std::invalid_argument(
        "OS grid references need two letters followed by an even number of digits (up to ten)"
    );
  }

  const int first = national_grid_letter_index(compact[0]);
  const int second = national_grid_letter_index(compact[1]);
  if (first < 0 || second < 0) {
    throw std::invalid_argument("OS grid-reference letters must be A-Z, excluding I");
  }
  const int hundred_kilometre_easting = ((first - 2 + 10) % 5) * 5 + second % 5;
  const int hundred_kilometre_northing = 19 - (first / 5) * 5 - second / 5;
  if (hundred_kilometre_easting < 0 || hundred_kilometre_easting > 6 ||
      hundred_kilometre_northing < 0 || hundred_kilometre_northing > 12) {
    throw std::invalid_argument("OS grid reference lies outside the National Grid");
  }

  const size_t digits_per_axis = digit_count / 2U;
  const auto parse_digits = [&](size_t offset) {
    uint32_t value = 0U;
    for (size_t index = 0U; index < digits_per_axis; index++) {
      value = value * 10U + static_cast<uint32_t>(compact[offset + index] - '0');
    }
    for (size_t index = digits_per_axis; index < 5U; index++) {
      value *= 10U;
    }
    return value;
  };
  return Coord{
      static_cast<double>(hundred_kilometre_easting * 100'000 + parse_digits(2U)),
      static_cast<double>(
          hundred_kilometre_northing * 100'000 + parse_digits(2U + digits_per_axis)
      ),
  };
}

[[nodiscard]] ParsedCoordinateInput
from_geographic(LatLon coordinate, const Crs &terrain_crs, std::string source_name) {
  if (coordinate.lat < -90.0 || coordinate.lat > 90.0 || coordinate.lon < -180.0 ||
      coordinate.lon > 180.0) {
    throw std::invalid_argument("Latitude must be within ±90° and longitude within ±180°");
  }
  return {
      terrain_crs.from_lat_lon(coordinate),
      coordinate,
      std::move(source_name),
  };
}

[[nodiscard]] ParsedCoordinateInput from_projected(
    Coord coordinate,
    const Crs &source_crs,
    const Crs &terrain_crs,
    std::string source_name
) {
  return from_geographic(source_crs.to_lat_lon(coordinate), terrain_crs, std::move(source_name));
}

[[nodiscard]] bool is_lv95(Coord coordinate) {
  return coordinate.x >= 2'000'000.0 && coordinate.x <= 3'000'000.0 &&
         coordinate.y >= 1'000'000.0 && coordinate.y <= 1'500'000.0;
}

[[nodiscard]] bool is_british_national_grid(Coord coordinate) {
  return coordinate.x >= 0.0 && coordinate.x <= 700'000.0 && coordinate.y >= 0.0 &&
         coordinate.y <= 1'300'000.0;
}

[[nodiscard]] bool is_french_lambert93(Coord coordinate) {
  return coordinate.x >= -380'000.0 && coordinate.x <= 1'330'000.0 && coordinate.y >= 6'000'000.0 &&
         coordinate.y <= 7'250'000.0;
}

struct NormalisedInput {
  std::string text;
  std::optional<CoordinateInputSystem> prefix;
};

[[nodiscard]] NormalisedInput normalise_input(std::string_view input) {
  std::string text = uppercase(trim_copy(input));
  if (text.empty()) {
    throw std::invalid_argument("Enter a coordinate");
  }

  std::optional<CoordinateInputSystem> prefix;
  if (strip_prefix(text, "LATLON") || strip_prefix(text, "WGS84") || strip_prefix(text, "WGS 84")) {
    prefix = CoordinateInputSystem::Wgs84;
  } else if (strip_prefix(text, "LV95")) {
    prefix = CoordinateInputSystem::SwissLv95;
  } else if (strip_prefix(text, "OSGB") || strip_prefix(text, "BNG") || strip_prefix(text, "OS")) {
    prefix = CoordinateInputSystem::BritishNationalGrid;
  } else if (strip_prefix(text, "DATASET") || strip_prefix(text, "PROJECTED")) {
    prefix = CoordinateInputSystem::Terrain;
  }
  if (text.empty()) {
    throw std::invalid_argument("Enter a coordinate after the coordinate-system prefix");
  }
  return {std::move(text), prefix};
}

[[nodiscard]] const char *system_name(CoordinateInputSystem system, const Crs &terrain_crs) {
  switch (system) {
  case CoordinateInputSystem::Wgs84:
    return "WGS 84 latitude/longitude";
  case CoordinateInputSystem::SwissLv95:
    return "Swiss LV95";
  case CoordinateInputSystem::BritishNationalGrid:
    return "OS National Grid";
  case CoordinateInputSystem::Terrain:
    return terrain_crs.name();
  }
  throw std::logic_error("Unknown coordinate input system");
}

[[nodiscard]] ParsedCoordinateInput parse_normalised_input(
    std::string_view text,
    const Crs &terrain_crs,
    CoordinateInputSystem system
) {
  if (system == CoordinateInputSystem::BritishNationalGrid) {
    if (const std::optional<Coord> grid = parse_national_grid_reference(text)) {
      ParsedCoordinateInput parsed = from_projected(
          *grid,
          Crs(CrsId::BritishNationalGrid),
          terrain_crs,
          system_name(system, terrain_crs)
      );
      parsed.system = system;
      return parsed;
    }
  }

  const std::optional<std::array<double, 2>> pair = parse_number_pair(text);
  if (!pair.has_value()) {
    throw std::invalid_argument(
        system == CoordinateInputSystem::BritishNationalGrid
            ? "Enter an OS grid reference or easting, northing"
            : "Enter two numeric coordinates separated by a comma or space"
    );
  }
  const Coord coordinate = {(*pair)[0], (*pair)[1]};
  ParsedCoordinateInput parsed;
  switch (system) {
  case CoordinateInputSystem::Wgs84:
    parsed = from_geographic(
        {coordinate.x, coordinate.y},
        terrain_crs,
        system_name(system, terrain_crs)
    );
    break;
  case CoordinateInputSystem::SwissLv95:
    parsed = from_projected(
        coordinate,
        Crs(CrsId::SwissLv95),
        terrain_crs,
        system_name(system, terrain_crs)
    );
    break;
  case CoordinateInputSystem::BritishNationalGrid:
    parsed = from_projected(
        coordinate,
        Crs(CrsId::BritishNationalGrid),
        terrain_crs,
        system_name(system, terrain_crs)
    );
    break;
  case CoordinateInputSystem::Terrain:
    parsed = from_projected(coordinate, terrain_crs, terrain_crs, system_name(system, terrain_crs));
    break;
  }
  parsed.system = system;
  return parsed;
}

void append_candidate(
    std::vector<ParsedCoordinateInput> &candidates,
    std::string_view text,
    const Crs &terrain_crs,
    CoordinateInputSystem system
) {
  try {
    ParsedCoordinateInput candidate = parse_normalised_input(text, terrain_crs, system);
    const bool duplicate =
        std::any_of(candidates.begin(), candidates.end(), [&](const auto &other) {
          return std::abs(other.projected.x - candidate.projected.x) < 0.01 &&
                 std::abs(other.projected.y - candidate.projected.y) < 0.01;
        });
    if (!duplicate) {
      candidates.push_back(std::move(candidate));
    }
  } catch (const std::exception &) {
    // A candidate inferred only from its numeric range is optional. An invalid
    // transformation should not hide another viable interpretation.
  }
}

} // namespace

ParsedCoordinateInput parse_coordinate_input(
    std::string_view input,
    const Crs &terrain_crs,
    CoordinateInputSystem system
) {
  NormalisedInput normalised = normalise_input(input);
  if (normalised.prefix.has_value() && *normalised.prefix != system) {
    throw std::invalid_argument(
        "The coordinate prefix does not match the selected coordinate system"
    );
  }
  return parse_normalised_input(normalised.text, terrain_crs, system);
}

std::vector<ParsedCoordinateInput>
detect_coordinate_inputs(std::string_view input, const Crs &terrain_crs) {
  NormalisedInput normalised = normalise_input(input);
  if (normalised.prefix.has_value()) {
    return {parse_normalised_input(normalised.text, terrain_crs, *normalised.prefix)};
  }

  if (parse_national_grid_reference(normalised.text).has_value()) {
    return {parse_normalised_input(
        normalised.text,
        terrain_crs,
        CoordinateInputSystem::BritishNationalGrid
    )};
  }

  const std::optional<std::array<double, 2>> pair = parse_number_pair(normalised.text);
  if (!pair.has_value()) {
    throw std::invalid_argument(
        "Enter a numeric pair, an OS grid reference, or select a coordinate system"
    );
  }
  const Coord coordinate = {(*pair)[0], (*pair)[1]};
  std::vector<ParsedCoordinateInput> candidates;
  if (std::abs(coordinate.x) <= 90.0 && std::abs(coordinate.y) <= 180.0) {
    append_candidate(candidates, normalised.text, terrain_crs, CoordinateInputSystem::Wgs84);
  }
  if (is_lv95(coordinate)) {
    append_candidate(candidates, normalised.text, terrain_crs, CoordinateInputSystem::SwissLv95);
  }
  if (is_british_national_grid(coordinate)) {
    append_candidate(
        candidates,
        normalised.text,
        terrain_crs,
        CoordinateInputSystem::BritishNationalGrid
    );
  }

  const bool active_grid_matched =
      (terrain_crs.id() == CrsId::SwissLv95 && is_lv95(coordinate)) ||
      (terrain_crs.id() == CrsId::BritishNationalGrid && is_british_national_grid(coordinate)) ||
      (terrain_crs.id() == CrsId::FrenchLambert93 && is_french_lambert93(coordinate));
  if (active_grid_matched || candidates.empty()) {
    append_candidate(candidates, normalised.text, terrain_crs, CoordinateInputSystem::Terrain);
  }
  if (candidates.empty()) {
    throw std::invalid_argument("Coordinate is outside the supported ranges");
  }
  return candidates;
}

} // namespace panorama::app
