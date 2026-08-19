#pragma once

#include <cpl_error.h>
#include <gdal_priv.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

namespace panorama {

/// Give one GDAL dataset unique ownership and close it on every return path.
struct GdalDatasetCloser {
  void operator()(GDALDataset *dataset) const { GDALClose(dataset); }
};

using GdalDatasetPointer = std::unique_ptr<GDALDataset, GdalDatasetCloser>;

/// Register GDAL's built-in raster drivers exactly once per process.
inline void register_gdal_drivers() {
  static std::once_flag registration_once;
  std::call_once(registration_once, [] { GDALAllRegister(); });
}

/// Append GDAL's thread-local diagnostic to an operation-level error.
[[nodiscard]] inline std::string gdal_error(const std::string &context) {
  const char *detail = CPLGetLastErrorMsg();
  return context + (detail != nullptr && detail[0] != '\0' ? ": " + std::string(detail) : "");
}

/// Compare floating-point grid metadata at the precision at which it is stored.
[[nodiscard]] inline bool approximately_equal(double left, double right) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <= 1e-10 * scale;
}

} // namespace panorama
