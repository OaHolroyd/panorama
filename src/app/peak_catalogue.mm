#include "peak_catalogue.h"

#import <Foundation/Foundation.h>

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace panorama::app {
namespace {

[[nodiscard]] double parse_number(std::string_view input, size_t line, const char *field) {
  double result = 0.0;
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), result);
  if (error != std::errc() || end != input.data() + input.size() || !std::isfinite(result)) {
    throw std::runtime_error(
        "Invalid " + std::string(field) + " on peak gazetteer line " + std::to_string(line)
    );
  }
  return result;
}

/// Parse ordinary unquoted fields and RFC 4180-style quoted fields. A quote
/// inside an unquoted DMS field is literal, as it is in the supplied dataset.
[[nodiscard]] std::vector<std::string> parse_csv_line(std::string_view line, size_t line_number) {
  std::vector<std::string> fields;
  size_t offset = 0;
  while (offset <= line.size()) {
    std::string field;
    if (offset < line.size() && line[offset] == '"') {
      ++offset;
      bool closed = false;
      while (offset < line.size()) {
        if (line[offset] != '"') {
          field.push_back(line[offset++]);
        } else if (offset + 1 < line.size() && line[offset + 1] == '"') {
          field.push_back('"');
          offset += 2;
        } else {
          ++offset;
          closed = true;
          break;
        }
      }
      if (!closed || (offset < line.size() && line[offset] != ','))
        throw std::runtime_error("Malformed peak gazetteer line " + std::to_string(line_number));
    } else {
      const size_t end = line.find(',', offset);
      field.assign(
          line.substr(offset, end == std::string_view::npos ? line.size() - offset : end - offset)
      );
      offset = end == std::string_view::npos ? line.size() : end;
    }
    fields.push_back(std::move(field));
    if (offset == line.size())
      break;
    ++offset;
  }
  return fields;
}

} // namespace

PeakCatalogue PeakCatalogue::load(const std::filesystem::path &path, const Crs &crs) {
  NSString *file = [NSString stringWithUTF8String:path.c_str()];
  NSError *error = nil;
  NSString *contents = [NSString stringWithContentsOfFile:file
                                                 encoding:NSWindowsCP1252StringEncoding
                                                    error:&error];
  if (contents == nil)
    throw std::runtime_error(
        "Could not read peak gazetteer: " + std::string(error.localizedDescription.UTF8String)
    );
  NSArray<NSString *> *lines =
      [contents componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet];
  if (lines.count == 0 ||
      ![lines[0] isEqualToString:@"Lat,Long,Elevation,Prom,Name,Lat_dec,Long_dec,Icon"])
    throw std::runtime_error("Peak gazetteer has an unexpected header");

  PeakCatalogue result;
  result.peaks_.reserve(lines.count - 1);
  std::vector<LatLon> coordinates;
  coordinates.reserve(lines.count - 1);
  for (NSUInteger index = 1; index < lines.count; ++index) {
    NSString *line = lines[index];
    if (line.length == 0)
      continue;
    const char *utf8 = line.UTF8String;
    if (utf8 == nullptr)
      throw std::runtime_error("Peak gazetteer contains undecodable text");
    const std::vector<std::string> columns = parse_csv_line(utf8, index + 1);
    if (columns.size() != 8U)
      throw std::runtime_error("Malformed peak gazetteer line " + std::to_string(index + 1));
    const double latitude = parse_number(columns[5], index + 1, "latitude");
    const double longitude = parse_number(columns[6], index + 1, "longitude");
    const double elevation = parse_number(columns[2], index + 1, "elevation");
    const double prominence = parse_number(columns[3], index + 1, "prominence");
    if (latitude < -90.0 || latitude > 90.0 || longitude < -180.0 || longitude > 180.0)
      throw std::runtime_error(
          "Invalid coordinate on peak gazetteer line " + std::to_string(index + 1)
      );
    if (columns[4].empty())
      throw std::runtime_error("Missing name on peak gazetteer line " + std::to_string(index + 1));
    coordinates.push_back({latitude, longitude});
    result.peaks_.push_back(
        {static_cast<uint32_t>(result.peaks_.size()),
         0.0,
         0.0,
         static_cast<float>(elevation),
         static_cast<float>(prominence),
         columns[4]}
    );
  }
  const std::vector<Coord> projected = crs.from_lat_lon(coordinates);
  for (size_t index = 0; index < projected.size(); ++index) {
    result.peaks_[index].easting = projected[index].x;
    result.peaks_[index].northing = projected[index].y;
  }
  return result;
}

} // namespace panorama::app
