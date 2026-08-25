#pragma once

#include "crs.h"

#include <optional>
#include <string_view>

namespace panorama::app {

/// Gregorian calendar fields. Callers choose their time-zone interpretation;
/// `solar_position` specifically expects these fields to describe UTC.
struct CalendarDateTime {
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

/// Geometric horizon crossings represented relative to UTC midnight.
struct DaylightTimes {
  DaylightState state;
  /// UTC minutes relative to midnight on the supplied date. Values are not
  /// wrapped so callers can preserve the correct day in every time zone.
  double sunrise_minutes;
  double sunset_minutes;
};

/// Parse strict DD-MM-YYYY and HH:MM input, rejecting invalid calendar dates.
[[nodiscard]] std::optional<CalendarDateTime>
parse_date_time(std::string_view date, std::string_view time);

/// Derive geometric solar angles at a projected observer coordinate.
/// Azimuth is clockwise from grid north and elevation is above the horizon.
[[nodiscard]] SolarPosition solar_position(const Crs &crs, Coord observer, CalendarDateTime utc);

/// Derive geometric sunrise and sunset for the supplied Gregorian date.
/// The clock fields in `date` are ignored.
[[nodiscard]] DaylightTimes daylight_times(const Crs &crs, Coord observer, CalendarDateTime date);

} // namespace panorama::app
