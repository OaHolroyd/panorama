#include "location_search.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cctype>
#include <cmath>

namespace panorama::app {
namespace {
NSString *normalised_name(std::string_view input) {
  NSString *text = [[NSString alloc] initWithBytes:input.data()
                                            length:input.size()
                                          encoding:NSUTF8StringEncoding];
  return [[text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]
      stringByFoldingWithOptions:NSCaseInsensitiveSearch | NSDiacriticInsensitiveSearch
                          locale:[NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"]];
}
} // namespace

bool is_coordinate_search(std::string_view input) {
  NSString *text = [normalised_name(input) uppercaseString];
  if (text.length == 0) {
    return false;
  }
  for (NSString *prefix in @[
         @"LATLON",
         @"WGS84",
         @"WGS 84",
         @"LV95",
         @"OSGB",
         @"BNG",
         @"OS",
         @"DATASET",
         @"PROJECTED"
       ]) {
    if ([text isEqualToString:prefix] ||
        ([text hasPrefix:prefix] && text.length > prefix.length &&
         ([text characterAtIndex:prefix.length] == ':' ||
          [NSCharacterSet.whitespaceAndNewlineCharacterSet
              characterIsMember:[text characterAtIndex:prefix.length]]))) {
      return true;
    }
  }
  // Decimal pairs, including partially entered exponents and separators.
  const std::string numeric(text.UTF8String);
  if (std::all_of(
          numeric.begin(),
          numeric.end(),
          [](unsigned char c) {
            return std::isdigit(c) || std::isspace(c) || c == '+' || c == '-' || c == '.' ||
                   c == ',' || c == ';' || c == 'E';
          }
      ) &&
      numeric.find_first_of("0123456789+-.,;") != std::string::npos) {
    return true;
  }
  // OS references start with two letters and digits; ordinary place names
  // starting with two letters must still reach autocomplete.
  NSMutableCharacterSet *separators = [NSCharacterSet.whitespaceAndNewlineCharacterSet mutableCopy];
  [separators addCharactersInString:@","];
  NSString *compact =
      [[text componentsSeparatedByCharactersInSet:separators] componentsJoinedByString:@""];
  if (compact.length > 2) {
    const auto letter_index = [](unichar c) {
      return c >= 'A' && c <= 'Z' && c != 'I' ? int(c - 'A' - (c > 'I' ? 1 : 0)) : -1;
    };
    const int first = letter_index([compact characterAtIndex:0]);
    const int second = letter_index([compact characterAtIndex:1]);
    if (first < 0 || second < 0) {
      return false;
    }
    const int easting = ((first - 2 + 10) % 5) * 5 + second % 5;
    const int northing = 19 - (first / 5) * 5 - second / 5;
    if (easting > 6 || northing < 0 || northing > 12) {
      return false;
    }
    // Postcodes such as SW1A 1AA are place queries, not OS references.
    NSString *digits = [compact substringFromIndex:2];
    return [digits rangeOfCharacterFromSet:NSCharacterSet.decimalDigitCharacterSet.invertedSet]
               .location == NSNotFound;
  }
  return false;
}

ParsedCoordinateInput parse_search_coordinate(std::string_view input, const Crs &terrain_crs) {
  // The detector already orders geographic coordinates before inferred grids,
  // and honours explicit prefixes before considering numeric ranges.
  return detect_coordinate_inputs(input, terrain_crs).front();
}

std::vector<uint32_t> search_peaks(
    const PeakCatalogue &catalogue,
    std::string_view query,
    LatLon observer,
    size_t limit
) {
  NSString *needle = normalised_name(query);
  if (needle.length == 0) {
    return {};
  }
  struct Match {
    uint32_t id;
    int rank;
    double distance;
  };
  std::vector<Match> matches;
  for (const PeakRecord &peak : catalogue.peaks()) {
    NSString *name = normalised_name(peak.name);
    if ([name rangeOfString:needle].location == NSNotFound) {
      continue;
    }
    const auto offset = geographic_offset(observer, peak.position);
    matches.push_back(
        {peak.id,
         [name isEqualToString:needle] ? 0
         : [name hasPrefix:needle]     ? 1
                                       : 2,
         std::hypot(offset.x, offset.y)}
    );
  }
  const auto order = [](const Match &a, const Match &b) {
    if (a.rank != b.rank) {
      return a.rank < b.rank;
    }
    if (a.distance != b.distance) {
      return a.distance < b.distance;
    }
    return a.id < b.id;
  };
  const size_t count = std::min(limit, matches.size());
  std::partial_sort(matches.begin(), matches.begin() + count, matches.end(), order);
  std::vector<uint32_t> result;
  for (size_t index = 0; index < count; ++index) {
    result.push_back(matches[index].id);
  }
  return result;
}
} // namespace panorama::app
