#include "peak_catalogue.h"

#import <Foundation/Foundation.h>

#include <cstdio>
#include <exception>
#include <string>

int main() {
  try {
    @autoreleasepool {
      NSString *path =
          [NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString];
      NSString *fixture = @"Lat,Lon,Elevation,Prom,Name\n"
                           "45.8325,6.86444,4808,4695,MONT BLANC\n"
                           "47.4204,13.0625,2941,2000,HOCHKÖNIG\n";
      NSError *error = nil;
      if (![fixture writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:&error]) {
        throw std::runtime_error("Could not write peak catalogue fixture");
      }
      const panorama::app::PeakCatalogue catalogue =
          panorama::app::PeakCatalogue::load(path.fileSystemRepresentation);
      [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
      if (catalogue.peaks().size() != 2U) {
        throw std::runtime_error("Unexpected peak count");
      }
      const auto &montBlanc = catalogue.at(0U);
      if (montBlanc.name != "MONT BLANC" || montBlanc.elevation != 4808.0F ||
          montBlanc.prominence != 4695.0F) {
        throw std::runtime_error("Mont Blanc record was parsed incorrectly");
      }
      if (catalogue.at(1U).name != "HOCHKÖNIG") {
        throw std::runtime_error("UTF-8 peak name was not decoded");
      }
    }
    std::printf("Peak catalogue tests passed.\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Peak catalogue test failed: %s\n", error.what());
    return 1;
  }
}
