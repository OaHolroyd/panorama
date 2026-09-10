#pragma once

#include "crs.h"
#include "ray_projection.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace panorama::app {

enum class PeakLabelMode : uint8_t { Off, On, NearPointer };

struct PeakRecord {
  uint32_t id;
  double easting;
  double northing;
  float elevation;
  float prominence;
  std::string name;
};

struct VisiblePeak {
  uint32_t peak_id;
  double pixel_x;
  double pixel_y;
  double distance;
  float prominence;
  float elevation;
  std::string name;
};

struct PeakLabelFrame {
  uint64_t revision;
  ImageSize output_image;
  std::vector<VisiblePeak> peaks;
};

class PeakCatalogue {
public:
  [[nodiscard]] static PeakCatalogue load(const std::filesystem::path &path, const Crs &crs);
  [[nodiscard]] const std::vector<PeakRecord> &peaks() const { return peaks_; }
  [[nodiscard]] const PeakRecord &at(uint32_t id) const { return peaks_.at(id); }

private:
  std::vector<PeakRecord> peaks_;
};

} // namespace panorama::app
