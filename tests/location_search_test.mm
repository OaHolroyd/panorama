#include "location_search.h"

#import <Foundation/Foundation.h>

#include <cstdio>
#include <stdexcept>

using namespace panorama;
using namespace panorama::app;

namespace {
void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
} // namespace

int main() {
  @autoreleasepool {
    try {
      for (const char *input : {"46.5, 7.2",
                                "-",
                                "46.",
                                "46.5,",
                                "4.6e",
                                "WGS84:",
                                "LV95 2600000, 1200000",
                                "LV95\t2600000, 1200000",
                                "NG 90716 59877",
                                "NG,90716,59877",
                                "NG9",
                                "OSGB:",
                                "dataset 400000 300000"}) {
        require(is_coordinate_search(input), "Coordinate text started place autocomplete");
      }
      for (const char *input : {"",
                                "Ben Nevis",
                                "Be",
                                "Mont Blanc",
                                "Hochkönig",
                                "London",
                                "SW1A 1AA",
                                "AB12 3CD",
                                "NG9 Hotel",
                                "123 Main Street",
                                "NG Hotel",
                                "Wgsborough"}) {
        require(!is_coordinate_search(input), "Place name suppressed autocomplete");
      }
      const auto global = Crs(CrsId::Wgs84);
      const auto latlon = parse_search_coordinate("46.5, 7.2", Crs(CrsId::SwissLv95));
      require(
          latlon.system == CoordinateInputSystem::Wgs84 && latlon.geographic.lat == 46.5 &&
              latlon.geographic.lon == 7.2,
          "Lat/lon did not take priority over grids"
      );
      require(
          parse_search_coordinate("2600000, 1200000", global).system ==
              CoordinateInputSystem::SwissLv95,
          "LV95 fallback failed"
      );
      require(
          parse_search_coordinate("NG 90716 59877", global).system ==
              CoordinateInputSystem::BritishNationalGrid,
          "OS reference fallback failed"
      );
      bool rejected = false;
      try {
        (void)parse_search_coordinate("WGS84: 91, 181", global);
      } catch (const std::exception &) {
        rejected = true;
      }
      require(rejected, "Invalid explicit coordinates were accepted");

      NSString *path =
          [NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString];
      NSString *fixture = @"Lat,Lon,Elevation,Prom,Name\n"
                           "45.8,6.8,4808,4695,MONT BLANC\n"
                           "46.0,7.0,2000,500,MONT BLANC\n"
                           "46.1,7.1,1900,400,MONT BLANC DE CHEILON\n"
                           "46.0,7.0,1800,300,PETIT MONT BLANC\n"
                           "47.4,13.0,2941,2000,HOCHKÖNIG\n";
      require(
          [fixture writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil],
          "Could not write peak fixture"
      );
      const auto catalogue = PeakCatalogue::load(path.fileSystemRepresentation);
      [NSFileManager.defaultManager removeItemAtPath:path error:nil];
      require(
          search_peaks(catalogue, "  mont blanc  ", {46, 7}) == std::vector<uint32_t>({1, 0, 2, 3}),
          "Peak exact/prefix/substring/distance ordering failed"
      );
      require(
          search_peaks(catalogue, "hochkonig", {46, 7}) == std::vector<uint32_t>({4}),
          "Accent-insensitive peak search failed"
      );
      require(search_peaks(catalogue, "mont", {46, 7}, 2).size() == 2, "Peak result limit failed");
      require(
          search_peaks(catalogue, "", {46, 7}).empty() &&
              search_peaks(catalogue, "London", {46, 7}).empty(),
          "Empty or unmatched input did not allow Apple fallback"
      );
      std::puts("Location search coordinate routing and peak matching passed.");
    } catch (const std::exception &error) {
      std::fprintf(stderr, "Location search test failed: %s\n", error.what());
      return 1;
    }
  }
}
