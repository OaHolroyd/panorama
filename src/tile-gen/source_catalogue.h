#pragma once

#include "terrain_types.h"

#include <filesystem>
#include <vector>

namespace panorama::terrain {

/// Metadata-only catalogue of compatible GeoTIFF elevation sources.
///
/// Discovery recursively opens candidate TIFF files, validates that they are
/// single-band north-up rasters on one shared CRS and sample grid, and closes
/// them without reading their elevation arrays. Rechunking opens only the
/// sources which overlap the output tile currently being constructed.
class SourceCatalogue {
public:
  /// Recursively inspect GeoTIFFs below `input_directory` in stable path order.
  [[nodiscard]] static SourceCatalogue discover(
      const std::filesystem::path &input_directory,
      const std::filesystem::path &excluded_directory
  );

  /// Return the input directory from which all sources were discovered.
  [[nodiscard]] const std::filesystem::path &input_directory() const;

  /// Return the CRS, resolution, registration, and unit common to all sources.
  [[nodiscard]] const SourceGrid &grid() const;

  /// Return every recursively discovered raster in stable path order.
  [[nodiscard]] const std::vector<SourceRaster> &sources() const;

private:
  SourceCatalogue(
      std::filesystem::path input_directory,
      SourceGrid grid,
      std::vector<SourceRaster> sources
  );

  std::filesystem::path input_directory_;
  SourceGrid grid_;
  std::vector<SourceRaster> sources_;
};

} // namespace panorama::terrain
