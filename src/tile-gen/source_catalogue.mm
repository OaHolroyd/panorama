#include "source_catalogue.h"

#include "gdal_utils.h"

#include <cpl_conv.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace panorama::terrain {
namespace {

/// Return whether a path has one of the supported DEM filename extensions.
[[nodiscard]] bool has_source_extension(const std::filesystem::path &path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  return extension == ".tif" || extension == ".tiff" || extension == ".hgt";
}

/// Return an absolute, lexically normal path without requiring it to exist.
[[nodiscard]] std::filesystem::path normal_path(const std::filesystem::path &path) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    throw std::runtime_error("Could not resolve path " + path.string() + ": " + error.message());
  }
  return absolute.lexically_normal();
}

/// Return whether `candidate` is equal to or below `directory`.
[[nodiscard]] bool
is_within(const std::filesystem::path &candidate, const std::filesystem::path &directory) {
  auto candidate_part = candidate.begin();
  for (auto directory_part = directory.begin(); directory_part != directory.end();
       directory_part++) {
    if (candidate_part == candidate.end() || *candidate_part != *directory_part) {
      return false;
    }
    candidate_part++;
  }
  return true;
}

/// Convert a positive GDAL raster dimension to the tool's fixed-width type.
[[nodiscard]] uint32_t checked_dimension(int value, const std::filesystem::path &path) {
  if (value <= 0 || static_cast<uintmax_t>(value) > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DEM source has an invalid dimension: " + path.string());
  }
  return static_cast<uint32_t>(value);
}

/// Require a coordinate to occupy an integer location on a reference sample grid.
void validate_grid_alignment(
    double coordinate,
    double reference,
    double resolution,
    const char *axis,
    const std::filesystem::path &path
) {
  const double index = (coordinate - reference) / resolution;
  if (!approximately_equal(index, std::round(index))) {
    throw std::runtime_error(
        "DEM source " + path.string() + " is not aligned with the shared " + axis + " sample grid"
    );
  }
}

/// Read the source's PixelIsArea or PixelIsPoint registration declaration.
[[nodiscard]] RasterRegistration registration_from_dataset(GDALDataset &dataset) {
  const char *area_or_point = dataset.GetMetadataItem(GDALMD_AREA_OR_POINT);
  if (area_or_point != nullptr && std::string(area_or_point) == "Point") {
    return RasterRegistration::PixelIsPoint;
  }
  return RasterRegistration::PixelIsArea;
}

/// Compare CRS definitions independently of GDAL's dataset axis-to-SRS mapping.
///
/// SRTMHGT exposes conventional raster X/Y as longitude/latitude while EPSG:4326
/// declares latitude/longitude authority order. Parsing its WKT alone therefore
/// produces a different data-axis mapping even though the underlying CRS is the
/// same. Terrain arrays consistently use conventional GIS X/Y, so normalise both
/// sides before comparison.
[[nodiscard]] bool
same_spatial_reference(const OGRSpatialReference &candidate, const std::string &shared_wkt) {
  OGRSpatialReference shared;
  if (shared.SetFromUserInput(shared_wkt.c_str()) != OGRERR_NONE) {
    throw std::runtime_error("Could not parse the shared DEM coordinate reference system");
  }
  OGRSpatialReference normalised_candidate(candidate);
  normalised_candidate.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  shared.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  return normalised_candidate.IsSame(&shared) != 0;
}

/// Open one candidate through the explicitly supported GDAL raster drivers.
[[nodiscard]] GdalDatasetPointer open_source(const std::filesystem::path &path) {
  const char *drivers[] = {"GTiff", "SRTMHGT", nullptr};
  GDALDataset *dataset = static_cast<GDALDataset *>(GDALOpenEx(
      path.string().c_str(),
      GDAL_OF_RASTER | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR,
      drivers,
      nullptr,
      nullptr
  ));
  if (dataset == nullptr) {
    throw std::runtime_error(gdal_error("Could not open DEM source " + path.string()));
  }
  return GdalDatasetPointer(dataset);
}

/// Parse and validate one DEM source without reading its elevation pixels.
[[nodiscard]] SourceRaster
inspect_source(const std::filesystem::path &path, SourceGrid *shared_grid, bool first_source) {
  GdalDatasetPointer dataset = open_source(path);
  if (dataset->GetRasterCount() != 1) {
    throw std::runtime_error("DEM source must have exactly one raster band: " + path.string());
  }

  std::array<double, 6> transform = {};
  if (dataset->GetGeoTransform(transform.data()) != CE_None) {
    throw std::runtime_error("DEM source has no affine geotransform: " + path.string());
  }
  if (transform[1] <= 0.0 || transform[5] >= 0.0 || transform[2] != 0.0 || transform[4] != 0.0 ||
      !approximately_equal(transform[1], -transform[5])) {
    throw std::runtime_error(
        "DEM source must be a north-up, unrotated grid with square pixels: " + path.string()
    );
  }

  const RasterRegistration registration = registration_from_dataset(*dataset);
  if (registration == RasterRegistration::PixelIsPoint) {
    // GDAL always reports an outer pixel-corner geotransform, including for
    // point-registered rasters. Rechunking reasons about sample positions, so
    // move the origin to the centre of sample (0, 0). For N46E007.hgt this
    // converts (6.999861..., 47.000138...) to the stated (7, 47) HGT bounds.
    transform[0] += 0.5 * transform[1] + 0.5 * transform[2];
    transform[3] += 0.5 * transform[4] + 0.5 * transform[5];
  }

  const OGRSpatialReference *reference = dataset->GetSpatialRef();
  if (reference == nullptr || (reference->IsProjected() == 0 && reference->IsGeographic() == 0)) {
    throw std::runtime_error(
        "DEM source must declare a projected or geographic CRS: " + path.string()
    );
  }
  char *projection_text = nullptr;
  if (reference->exportToWkt(&projection_text) != OGRERR_NONE || projection_text == nullptr) {
    throw std::runtime_error("Could not serialise DEM source CRS: " + path.string());
  }
  const std::string projection_wkt(projection_text);
  CPLFree(projection_text);

  GDALRasterBand *band = dataset->GetRasterBand(1);
  const char *unit_text = band->GetUnitType();
  const std::string elevation_unit = unit_text == nullptr ? "" : unit_text;

  if (first_source) {
    shared_grid->projection_wkt = projection_wkt;
    shared_grid->coordinate_system_name =
        reference->GetName() == nullptr ? "unknown" : reference->GetName();
    shared_grid->elevation_unit = elevation_unit;
    shared_grid->x_resolution = transform[1];
    shared_grid->y_resolution = -transform[5];
    shared_grid->reference_x = transform[0];
    shared_grid->reference_y = transform[3];
    shared_grid->registration = registration;
  } else {
    if (!same_spatial_reference(*reference, shared_grid->projection_wkt)) {
      throw std::runtime_error("DEM source CRS does not match the first source: " + path.string());
    }
    if (!approximately_equal(transform[1], shared_grid->x_resolution) ||
        !approximately_equal(-transform[5], shared_grid->y_resolution)) {
      throw std::runtime_error(
          "DEM source resolution does not match the first source: " + path.string()
      );
    }
    if (registration != shared_grid->registration) {
      throw std::runtime_error(
          "DEM source pixel registration does not match the first source: " + path.string()
      );
    }
    if (elevation_unit != shared_grid->elevation_unit) {
      throw std::runtime_error(
          "DEM source elevation unit does not match the first source: " + path.string()
      );
    }
    validate_grid_alignment(
        transform[0],
        shared_grid->reference_x,
        shared_grid->x_resolution,
        "X",
        path
    );
    validate_grid_alignment(
        transform[3],
        shared_grid->reference_y,
        shared_grid->y_resolution,
        "Y",
        path
    );
  }

  int has_no_data = false;
  const double no_data_value = band->GetNoDataValue(&has_no_data);
  int has_scale = false;
  const double scale = band->GetScale(&has_scale);
  int has_offset = false;
  const double offset = band->GetOffset(&has_offset);

  return {
      path,
      transform,
      checked_dimension(dataset->GetRasterXSize(), path),
      checked_dimension(dataset->GetRasterYSize(), path),
      static_cast<int>(band->GetRasterDataType()),
      has_no_data != 0 ? std::optional<double>(no_data_value) : std::nullopt,
      has_scale != 0 ? scale : 1.0,
      has_offset != 0 ? offset : 0.0,
      band->GetMaskFlags(),
      registration,
  };
}

} // namespace

SourceCatalogue SourceCatalogue::discover(
    const std::filesystem::path &input_directory,
    const std::filesystem::path &excluded_directory
) {
  register_gdal_drivers();
  const std::filesystem::path input = normal_path(input_directory);
  const std::filesystem::path excluded =
      excluded_directory.empty() ? std::filesystem::path() : normal_path(excluded_directory);
  if (!std::filesystem::is_directory(input)) {
    throw std::invalid_argument("Input path is not a directory: " + input.string());
  }

  std::vector<std::filesystem::path> paths;
  std::error_code iteration_error;
  std::filesystem::recursive_directory_iterator iterator(
      input,
      std::filesystem::directory_options::skip_permission_denied,
      iteration_error
  );
  const std::filesystem::recursive_directory_iterator end;
  while (iterator != end) {
    if (iteration_error) {
      throw std::runtime_error(
          "Could not scan input directory " + input.string() + ": " + iteration_error.message()
      );
    }

    const std::filesystem::directory_entry &entry = *iterator;
    const std::filesystem::path candidate = normal_path(entry.path());
    if (!excluded.empty() && entry.is_directory() && is_within(candidate, excluded)) {
      iterator.disable_recursion_pending();
    } else if (entry.is_regular_file() && has_source_extension(candidate)) {
      paths.push_back(candidate);
    }
    iterator.increment(iteration_error);
  }
  if (iteration_error) {
    throw std::runtime_error(
        "Could not scan input directory " + input.string() + ": " + iteration_error.message()
    );
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    throw std::runtime_error("No GeoTIFF or HGT files were found below " + input.string());
  }

  SourceGrid grid = {};
  std::vector<SourceRaster> sources;
  sources.reserve(paths.size());
  for (const std::filesystem::path &path : paths) {
    sources.push_back(inspect_source(path, &grid, sources.empty()));
  }
  return SourceCatalogue(input, std::move(grid), std::move(sources));
}

SourceCatalogue::SourceCatalogue(
    std::filesystem::path input_directory,
    SourceGrid grid,
    std::vector<SourceRaster> sources
)
    : input_directory_(std::move(input_directory)), grid_(std::move(grid)),
      sources_(std::move(sources)) {}

const std::filesystem::path &SourceCatalogue::input_directory() const { return input_directory_; }

const SourceGrid &SourceCatalogue::grid() const { return grid_; }

const std::vector<SourceRaster> &SourceCatalogue::sources() const { return sources_; }

} // namespace panorama::terrain
