#include "solar_position.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace panorama::app {
namespace {

constexpr double kDegreesToRadians = std::numbers::pi / 180.0;
constexpr double kRadiansToDegrees = 180.0 / std::numbers::pi;
constexpr double kEarthRadiusMetres = 6'378'137.0;

[[nodiscard]] bool leap_year(int year) {
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

[[nodiscard]] int days_in_month(int year, int month) {
  constexpr int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  return month == 2 && leap_year(year) ? 29 : days[month - 1];
}

[[nodiscard]] std::optional<int> parse_digits(std::string_view text) {
  int value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] double normalised_degrees(double value) {
  value = std::fmod(value, 360.0);
  return value < 0.0 ? value + 360.0 : value;
}

[[nodiscard]] double julian_day(CalendarDateTime utc) {
  int year = utc.year;
  int month = utc.month;
  if (month <= 2) {
    year--;
    month += 12;
  }
  const int century = year / 100;
  const int correction = 2 - century + century / 4;
  const double day_fraction =
      (static_cast<double>(utc.hour) + static_cast<double>(utc.minute) / 60.0) / 24.0;
  return std::floor(365.25 * static_cast<double>(year + 4716)) +
         std::floor(30.6001 * static_cast<double>(month + 1)) + static_cast<double>(utc.day) +
         day_fraction + static_cast<double>(correction) - 1524.5;
}

struct SolarEphemeris {
  double declination;
  double equation_of_time;
};

[[nodiscard]] SolarEphemeris solar_ephemeris(CalendarDateTime utc) {
  const double centuries = (julian_day(utc) - 2'451'545.0) / 36'525.0;
  const double mean_longitude =
      normalised_degrees(280.46646 + centuries * (36'000.76983 + 0.0003032 * centuries));
  const double mean_anomaly =
      (357.52911 + centuries * (35'999.05029 - 0.0001537 * centuries)) * kDegreesToRadians;
  const double eccentricity = 0.016708634 - centuries * (0.000042037 + 0.0000001267 * centuries);
  const double equation_of_centre =
      std::sin(mean_anomaly) * (1.914602 - centuries * (0.004817 + 0.000014 * centuries)) +
      std::sin(2.0 * mean_anomaly) * (0.019993 - 0.000101 * centuries) +
      std::sin(3.0 * mean_anomaly) * 0.000289;
  const double true_longitude = mean_longitude + equation_of_centre;
  const double omega = (125.04 - 1934.136 * centuries) * kDegreesToRadians;
  const double apparent_longitude =
      (true_longitude - 0.00569 - 0.00478 * std::sin(omega)) * kDegreesToRadians;
  const double seconds =
      21.448 - centuries * (46.815 + centuries * (0.00059 - centuries * 0.001813));
  const double mean_obliquity = (23.0 + (26.0 + seconds / 60.0) / 60.0) * kDegreesToRadians;
  const double obliquity = mean_obliquity + 0.00256 * kDegreesToRadians * std::cos(omega);
  const double declination = std::asin(std::sin(obliquity) * std::sin(apparent_longitude));
  const double y = std::tan(obliquity / 2.0) * std::tan(obliquity / 2.0);
  const double longitude_radians = mean_longitude * kDegreesToRadians;
  const double equation_of_time =
      4.0 * kRadiansToDegrees *
      (y * std::sin(2.0 * longitude_radians) - 2.0 * eccentricity * std::sin(mean_anomaly) +
       4.0 * eccentricity * y * std::sin(mean_anomaly) * std::cos(2.0 * longitude_radians) -
       0.5 * y * y * std::sin(4.0 * longitude_radians) -
       1.25 * eccentricity * eccentricity * std::sin(2.0 * mean_anomaly));
  return {declination, equation_of_time};
}

} // namespace

std::optional<CalendarDateTime> parse_date_time(std::string_view date, std::string_view time) {
  if (date.size() != 10U || date[2] != '-' || date[5] != '-' || time.size() != 5U ||
      time[2] != ':') {
    return std::nullopt;
  }
  const std::optional<int> day = parse_digits(date.substr(0U, 2U));
  const std::optional<int> month = parse_digits(date.substr(3U, 2U));
  const std::optional<int> year = parse_digits(date.substr(6U, 4U));
  const std::optional<int> hour = parse_digits(time.substr(0U, 2U));
  const std::optional<int> minute = parse_digits(time.substr(3U, 2U));
  if (!day || !month || !year || !hour || !minute || *year < 1583 || *year > 9999 || *month < 1 ||
      *month > 12 || *day < 1 || *day > days_in_month(*year, *month) || *hour < 0 || *hour > 23 ||
      *minute < 0 || *minute > 59) {
    return std::nullopt;
  }
  return CalendarDateTime{*year, *month, *day, *hour, *minute};
}

SolarPosition solar_position(const Crs &crs, Coord observer, CalendarDateTime utc) {
  const LatLon geographic = crs.to_lat_lon(observer);
  const double latitude = geographic.lat * kDegreesToRadians;
  const double longitude = geographic.lon;
  const SolarEphemeris ephemeris = solar_ephemeris(utc);
  double solar_minutes = static_cast<double>(utc.hour * 60 + utc.minute) +
                         ephemeris.equation_of_time + 4.0 * longitude;
  solar_minutes = std::fmod(solar_minutes, 1440.0);
  if (solar_minutes < 0.0) {
    solar_minutes += 1440.0;
  }
  const double hour_angle = (solar_minutes / 4.0 - 180.0) * kDegreesToRadians;
  const double sin_elevation = std::clamp(
      std::sin(latitude) * std::sin(ephemeris.declination) +
          std::cos(latitude) * std::cos(ephemeris.declination) * std::cos(hour_angle),
      -1.0,
      1.0
  );
  const double elevation = std::asin(sin_elevation);
  const double true_azimuth = normalised_degrees(
      std::atan2(
          std::sin(hour_angle),
          std::cos(hour_angle) * std::sin(latitude) -
              std::tan(ephemeris.declination) * std::cos(latitude)
      ) * kRadiansToDegrees +
      180.0
  );

  // Astronomical azimuth is relative to true north. Project short true-east
  // and true-north basis vectors through the terrain CRS so grid convergence
  // is included in the direction sent to the renderer.
  constexpr double basis_metres = 100.0;
  const double latitude_step = basis_metres / kEarthRadiusMetres * kRadiansToDegrees;
  const double longitude_step = latitude_step / std::cos(latitude);
  const Coord projected_east = crs.from_lat_lon({geographic.lat, geographic.lon + longitude_step});
  const Coord projected_north = crs.from_lat_lon({geographic.lat + latitude_step, geographic.lon});
  const double azimuth = true_azimuth * kDegreesToRadians;
  const double projected_x = (projected_east.x - observer.x) * std::sin(azimuth) +
                             (projected_north.x - observer.x) * std::cos(azimuth);
  const double projected_y = (projected_east.y - observer.y) * std::sin(azimuth) +
                             (projected_north.y - observer.y) * std::cos(azimuth);
  if (!std::isfinite(projected_x) || !std::isfinite(projected_y) ||
      std::hypot(projected_x, projected_y) == 0.0) {
    throw std::runtime_error("Could not project the astronomical sun direction");
  }
  double grid_azimuth = std::atan2(projected_x, projected_y);
  if (grid_azimuth < 0.0) {
    grid_azimuth += 2.0 * std::numbers::pi;
  }
  return {grid_azimuth, elevation};
}

DaylightTimes daylight_times(const Crs &crs, Coord observer, CalendarDateTime date) {
  const LatLon geographic = crs.to_lat_lon(observer);
  const double latitude = geographic.lat * kDegreesToRadians;
  date.hour = 12;
  date.minute = 0;
  const SolarEphemeris ephemeris = solar_ephemeris(date);
  const double cosine_hour_angle = -std::tan(latitude) * std::tan(ephemeris.declination);
  if (cosine_hour_angle <= -1.0) {
    return {DaylightState::PolarDay, 0.0, 0.0};
  }
  if (cosine_hour_angle >= 1.0) {
    return {DaylightState::PolarNight, 0.0, 0.0};
  }
  const double hour_angle_degrees = std::acos(cosine_hour_angle) * kRadiansToDegrees;
  const double solar_noon = 720.0 - 4.0 * geographic.lon - ephemeris.equation_of_time;
  return {
      DaylightState::Normal,
      solar_noon - 4.0 * hour_angle_degrees,
      solar_noon + 4.0 * hour_angle_degrees,
  };
}

} // namespace panorama::app
