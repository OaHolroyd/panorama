#pragma once

#include "crs.h"

#include <optional>
#include <string_view>

namespace panorama::app {

/// One Gregorian date and UTC clock time used for astronomical lighting.
struct UtcDateTime {
  int year;
  int month;
  int day;
  int hour;
  int minute;
};

/// Sun direction in the renderer's projected grid frame.
struct SolarPosition {
  double azimuth;
  double elevation;
};

enum class DaylightState {
  Normal,
  PolarDay,
  PolarNight,
};

/// Geometric horizon crossings in UTC minutes after midnight.
struct DaylightTimes {
  DaylightState state;
  double sunrise_minutes;
  double sunset_minutes;
};

/// Parse strict DD-MM-YYYY and HH:MM input, rejecting invalid calendar dates.
[[nodiscard]] std::optional<UtcDateTime>
parse_utc_date_time(std::string_view date, std::string_view time);

/// Derive geometric solar angles at a projected observer coordinate.
/// Azimuth is clockwise from grid north and elevation is above the horizon.
[[nodiscard]] SolarPosition solar_position(const Crs &crs, Coord observer, UtcDateTime utc);

/// Derive geometric sunrise and sunset for the supplied UTC calendar date.
/// The clock fields in `date` are ignored.
[[nodiscard]] DaylightTimes daylight_times(const Crs &crs, Coord observer, UtcDateTime date);

} // namespace panorama::app
