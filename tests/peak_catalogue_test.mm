#include "peak_catalogue.h"

#include <cstdio>
#include <exception>
#include <string>

int main() {
  try {
    const panorama::app::PeakCatalogue catalogue = panorama::app::PeakCatalogue::load(
        "data/gazetteers/Alps589.csv",
        panorama::Crs(panorama::CrsId::SwissLv95)
    );
    if (catalogue.peaks().size() != 633U)
      throw std::runtime_error("Unexpected Alpine peak count");
    const auto &montBlanc = catalogue.at(0U);
    if (montBlanc.name != "MONT BLANC" || montBlanc.elevation != 4808.0F ||
        montBlanc.prominence != 4695.0F)
      throw std::runtime_error("Mont Blanc record was parsed incorrectly");
    if (catalogue.at(5U).name != "HOCHKÖNIG")
      throw std::runtime_error("Windows-1252 peak name was not decoded");
    std::printf("Peak catalogue tests passed.\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Peak catalogue test failed: %s\n", error.what());
    return 1;
  }
}
