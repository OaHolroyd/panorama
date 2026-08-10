#include "loaded_tile.h"

#include <cpl_error.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

struct DatasetCloser {
  /// Give GDALDataset unique_ptr ownership semantics and close the file once.
  void operator()(GDALDataset *dataset) const { GDALClose(dataset); }
};

/// Append GDAL's thread-local error text to an operation-level error message.
[[nodiscard]] std::string gdal_error(const std::string &context) {
  const char *detail = CPLGetLastErrorMsg();
  return context + (detail != nullptr && detail[0] != '\0'
                        ? ": " + std::string(detail)
                        : "");
}

/// Validate a positive GDAL raster dimension before narrowing it to uint32_t.
[[nodiscard]] uint32_t checked_dimension(int value, const char *name) {
  if (value <= 0 ||
      static_cast<uintmax_t>(value) > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string("Invalid GeoTIFF ") + name);
  }
  return static_cast<uint32_t>(value);
}

/// Return whether a non-zero dimension can be repeatedly halved to one.
[[nodiscard]] bool is_power_of_two(uint32_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

/// Extract the GeoTIFF's authority code and restrict it to the supported CRSs.
[[nodiscard]] Crs crs_from_dataset(GDALDataset &dataset) {
  const char *projection = dataset.GetProjectionRef();
  if (projection == nullptr || projection[0] == '\0') {
    throw std::runtime_error("GeoTIFF has no projected CRS");
  }

  OGRSpatialReference reference;
  if (reference.SetFromUserInput(projection) != OGRERR_NONE) {
    throw std::runtime_error("Could not parse the GeoTIFF CRS");
  }
  // A GeoTIFF may store WKT rather than a literal EPSG tag. Ask GDAL to map
  // that WKT back to its authority code before applying our restricted set.
  reference.AutoIdentifyEPSG();
  const char *authority_code = reference.GetAuthorityCode(nullptr);
  if (authority_code == nullptr) {
    throw std::runtime_error("GeoTIFF CRS has no identifiable EPSG code");
  }

  char *end = nullptr;
  const unsigned long epsg_code = std::strtoul(authority_code, &end, 10);
  if (end == authority_code || *end != '\0' ||
      epsg_code > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("GeoTIFF CRS has an invalid EPSG code");
  }
  return Crs::from_epsg(static_cast<uint32_t>(epsg_code));
}

// GeoTIFF registration affects how the loader translates its affine transform
// into LoadedTile's canonical origin, but has no meaning after that conversion.
enum class RasterRegistration {
  PixelIsArea,
  PixelIsPoint,
};

/// Read the GeoTIFF PixelIsArea/PixelIsPoint registration convention.
[[nodiscard]] RasterRegistration
registration_from_dataset(GDALDataset &dataset) {
  const char *area_or_point = dataset.GetMetadataItem(GDALMD_AREA_OR_POINT);
  if (area_or_point != nullptr && std::string(area_or_point) == "Point") {
    return RasterRegistration::PixelIsPoint;
  }
  return RasterRegistration::PixelIsArea;
}

/// Build the required level-1 maximum field from a square vertex grid.
[[nodiscard]] std::vector<float>
make_level_1_cells(const std::vector<float> &vertices, uint32_t size) {
  std::vector<float> cells(static_cast<size_t>(size) * size);
  const uint32_t vertex_size = size + 1U;
  for (uint32_t y = 0; y < size; ++y) {
    for (uint32_t x = 0; x < size; ++x) {
      const size_t lower_left = static_cast<size_t>(y) * vertex_size + x;
      cells[static_cast<size_t>(y) * size + x] = std::max(
          std::max(vertices[lower_left], vertices[lower_left + 1U]),
          std::max(
              vertices[lower_left + vertex_size],
              vertices[lower_left + vertex_size + 1U]
          )
      );
    }
  }
  return cells;
}

} // namespace

LoadedTile LoadedTile::load_tif(
    const std::filesystem::path &path,
    bool level_0_collisions
) {
  // Keep the initial file-format contract narrow. Other loaders can later
  // prepare exactly the same LoadedTile representation from other sources.
  if (path.extension() != ".tif") {
    throw std::invalid_argument(
        "Only .tif terrain tiles are supported: " + path.string()
    );
  }

  // Register GDAL's built-in raster drivers before opening a dataset. GDALOpen
  // returns an opaque pointer owned by GDAL, so immediately wrap it in a
  // unique_ptr to close the file on every success or exception path.
  GDALAllRegister();
  GDALDataset *raw_dataset = static_cast<GDALDataset *>(
      GDALOpen(path.string().c_str(), GDALAccess::GA_ReadOnly)
  );
  if (raw_dataset == nullptr) {
    throw std::runtime_error(
        gdal_error("Could not open GeoTIFF " + path.string())
    );
  }
  std::unique_ptr<GDALDataset, DatasetCloser> dataset(raw_dataset);

  // Terrain is one scalar elevation field, not a multi-band image.
  if (dataset->GetRasterCount() != 1) {
    throw std::runtime_error(
        "Terrain GeoTIFF must have exactly one raster band"
    );
  }

  const uint32_t width = checked_dimension(dataset->GetRasterXSize(), "width");
  const uint32_t height =
      checked_dimension(dataset->GetRasterYSize(), "height");
  if (width != height) {
    throw std::runtime_error("Terrain GeoTIFF must be square");
  }
  // A level-0 raster has one more vertex than cells along each side. A level-1
  // raster has exactly one value per cell. `size` always means cell count.
  if (level_0_collisions && width < 2U) {
    throw std::runtime_error(
        "Level-0 terrain GeoTIFF needs at least a 2 by 2 vertex grid"
    );
  }
  const uint32_t source_size = width;
  const uint32_t size = level_0_collisions ? source_size - 1U : source_size;
  // The implicit maximum mipmap repeatedly combines 2×2 cells. Requiring a
  // power-of-two cell count means every level remains square down to one cell.
  if (!is_power_of_two(size)) {
    throw std::runtime_error(
        "Terrain GeoTIFF level-1 cell count must be a power of two"
    );
  }
  if (static_cast<size_t>(source_size) >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(source_size)) {
    throw std::runtime_error("Terrain GeoTIFF is too large to load");
  }

  // GDAL's affine transform maps a source pixel corner (column, row) to:
  // x = transform[0] + column * transform[1] + row * transform[2]
  // y = transform[3] + column * transform[4] + row * transform[5]
  double transform[6] = {};
  if (dataset->GetGeoTransform(transform) != CE_None) {
    throw std::runtime_error("GeoTIFF has no affine geotransform");
  }
  // The tracer only supports regular north-up grids. GeoTIFF rows normally
  // run north-to-south, so transform[5] must be negative before we flip them.
  if (transform[1] <= 0.0 || transform[5] >= 0.0 || transform[2] != 0.0 ||
      transform[4] != 0.0) {
    throw std::runtime_error(
        "Terrain GeoTIFF must be north-up with square, unrotated pixels"
    );
  }
  const double delta = transform[1];
  if (delta != -transform[5]) {
    throw std::runtime_error("Terrain GeoTIFF must have square pixels");
  }

  GDALRasterBand *band = dataset->GetRasterBand(1);
  int has_no_data = false;
  const double no_data = band->GetNoDataValue(&has_no_data);
  const size_t sample_count = static_cast<size_t>(source_size) * source_size;
  // RasterIO converts the source's native sample format directly to float32.
  // This is the only terrain precision the eventual Metal tracer supports.
  std::vector<float> source_heights(sample_count);
  if (band->RasterIO(
          GF_Read,
          0,
          0,
          static_cast<int>(source_size),
          static_cast<int>(source_size),
          source_heights.data(),
          static_cast<int>(source_size),
          static_cast<int>(source_size),
          GDT_Float32,
          0,
          0,
          nullptr
      ) != CE_None) {
    throw std::runtime_error(gdal_error("Could not read terrain heights"));
  }

  // GeoTIFF source row zero is normally northernmost. Reverse the rows so the
  // returned row zero is southernmost and tracer Y increases northward.
  std::vector<float> oriented_samples(sample_count);
  float maximum_elevation = -std::numeric_limits<float>::infinity();
  for (uint32_t source_row = 0; source_row < source_size; ++source_row) {
    const uint32_t trace_row = source_size - 1U - source_row;
    for (uint32_t x = 0; x < source_size; ++x) {
      const size_t source_index =
          static_cast<size_t>(source_row) * source_size + x;
      const float elevation = source_heights[source_index];
      // A hole in the elevation field cannot be treated as sea level: it would
      // create false terrain hits. Reject no-data tiles until coverage handling
      // is implemented at the multi-tile traversal layer.
      if (!std::isfinite(elevation) ||
          (has_no_data != 0 && elevation == static_cast<float>(no_data))) {
        throw std::runtime_error(
            "Terrain GeoTIFF contains no-data or non-finite "
            "elevations"
        );
      }
      oriented_samples[static_cast<size_t>(trace_row) * source_size + x] =
          elevation;
      maximum_elevation = std::max(maximum_elevation, elevation);
    }
  }

  std::unique_ptr<std::vector<float>> level_0_vertices;
  std::vector<float> level_1_cells;
  if (level_0_collisions) {
    // Preserve the original vertices for future exact bilinear collision and
    // derive the first conservative maximum level used by all traversal modes.
    level_0_vertices =
        std::make_unique<std::vector<float>>(std::move(oriented_samples));
    level_1_cells = make_level_1_cells(*level_0_vertices, size);
  } else {
    // The source is already the N×N first maximum-mipmap level.
    level_1_cells = std::move(oriented_samples);
  }

  const RasterRegistration registration = registration_from_dataset(*dataset);
  double lower_left_x = transform[0];
  double lower_left_y = transform[3] - static_cast<double>(size) * delta;
  if (level_0_collisions) {
    // Prepared level-0 terrain records vertices at the affine grid positions,
    // despite carrying the common GeoTIFF PixelIsArea metadata. Match the
    // Python reference: transform[0] is the west vertex and the last source
    // row is the south vertex.
    lower_left_y = transform[3] - static_cast<double>(size) * delta;
  } else if (registration != RasterRegistration::PixelIsArea) {
    throw std::runtime_error(
        "Level-1 terrain GeoTIFF must use PixelIsArea registration"
    );
  }

  return {level_0_collisions,
          crs_from_dataset(*dataset),
          maximum_elevation,
          size,
          lower_left_x,
          lower_left_y,
          delta,
          std::move(level_0_vertices),
          std::move(level_1_cells)};
}

} // namespace panorama
