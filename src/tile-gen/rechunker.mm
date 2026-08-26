#include "rechunker.h"

#include "gdal_utils.h"

#include <gdal_priv.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace panorama::terrain {
namespace {

/// Convert an aligned coordinate difference to an exact signed sample index.
[[nodiscard]] int64_t
aligned_index(double coordinate, double origin, double resolution, const char *axis) {
  const double index = (coordinate - origin) / resolution;
  const double rounded = std::round(index);
  if (!approximately_equal(index, rounded) ||
      rounded < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      rounded > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument(
        std::string("Destination origin is not aligned with the source ") + axis + " sample grid"
    );
  }
  return static_cast<int64_t>(rounded);
}

/// Divide signed grid coordinates using mathematical floor rather than truncation.
[[nodiscard]] int64_t floor_divide(int64_t numerator, int64_t denominator) {
  int64_t quotient = numerator / denominator;
  if (numerator % denominator < 0) {
    quotient--;
  }
  return quotient;
}

/// Open a supported source raster for the pixel-reading pass.
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

/// Return whether one decoded source value represents usable terrain.
[[nodiscard]] bool valid_value(float value, const SourceRaster &source) {
  if (!std::isfinite(value)) {
    return false;
  }
  if (!source.no_data.has_value()) {
    return true;
  }
  const float no_data = static_cast<float>(*source.no_data);
  return std::isnan(no_data) ? !std::isnan(value) : value != no_data;
}

} // namespace

uint32_t sample_side(const DestinationGrid &grid) {
  if (grid.tile_cell_count == 0U) {
    throw std::invalid_argument("Destination tile cell count must be positive");
  }
  if (grid.layout == RasterLayout::Level0) {
    if (grid.tile_cell_count == std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error("Destination level-0 sample side is too large");
    }
    return grid.tile_cell_count + 1U;
  }
  return grid.tile_cell_count;
}

RechunkPlan
make_rechunk_plan(const SourceCatalogue &catalogue, const DestinationGrid &destination) {
  if (!std::isfinite(destination.origin_x) || !std::isfinite(destination.origin_y) ||
      !std::isfinite(destination.resolution) || destination.resolution <= 0.0) {
    throw std::invalid_argument("Destination grid coordinates and resolution must be finite");
  }
  const SourceGrid &source_grid = catalogue.grid();
  if (!approximately_equal(destination.resolution, source_grid.x_resolution) ||
      !approximately_equal(destination.resolution, source_grid.y_resolution)) {
    throw std::invalid_argument("Rechunking requires matching source and destination resolutions");
  }

  // Rechunking copies aligned source windows without resampling. An arbitrary
  // output origin is supported only when it lies on the source sample grid;
  // reprojection and sub-pixel origins are not supported.
  const int64_t destination_x_alignment =
      aligned_index(destination.origin_x, source_grid.reference_x, destination.resolution, "X");
  const int64_t destination_y_alignment =
      aligned_index(destination.origin_y, source_grid.reference_y, destination.resolution, "Y");
  (void)destination_x_alignment;
  (void)destination_y_alignment;

  const uint32_t side = sample_side(destination);
  const int64_t stride = static_cast<int64_t>(destination.tile_cell_count);
  RechunkPlan plan = {destination, {}, {}};
  plan.sources.reserve(catalogue.sources().size());

  for (uint32_t source_index = 0U; source_index < catalogue.sources().size(); source_index++) {
    const SourceRaster &source = catalogue.sources()[source_index];
    const int64_t source_column =
        aligned_index(source.transform[0], destination.origin_x, destination.resolution, "X");
    const int64_t source_row =
        aligned_index(source.transform[3], destination.origin_y, -destination.resolution, "Y");
    const uint32_t indexed_source = static_cast<uint32_t>(plan.sources.size());
    plan.sources.push_back({source_index, source_row, source_column});

    // A level-0 chunk overlaps its neighbours by one sample. Include every
    // chunk whose complete sample window intersects this source rectangle.
    const int64_t chunk_column_start =
        floor_divide(source_column - static_cast<int64_t>(side - 1U), stride);
    const int64_t chunk_column_end =
        floor_divide(source_column + static_cast<int64_t>(source.width) - 1, stride);
    const int64_t chunk_row_start =
        floor_divide(source_row - static_cast<int64_t>(side - 1U), stride);
    const int64_t chunk_row_end =
        floor_divide(source_row + static_cast<int64_t>(source.height) - 1, stride);

    for (int64_t row = chunk_row_start; row <= chunk_row_end; row++) {
      for (int64_t column = chunk_column_start; column <= chunk_column_end; column++) {
        plan.contributors[{row, column}].push_back(indexed_source);
      }
    }
  }
  return plan;
}

TerrainChunk build_chunk(
    const SourceCatalogue &catalogue,
    const RechunkPlan &plan,
    ChunkKey key,
    std::span<const uint32_t> contributor_indices
) {
  const uint32_t side = sample_side(plan.grid);
  if (static_cast<size_t>(side) > std::numeric_limits<size_t>::max() / side) {
    throw std::overflow_error("Destination terrain chunk is too large");
  }
  const size_t sample_count = static_cast<size_t>(side) * side;
  TerrainChunk chunk = {
      side,
      std::vector<float>(sample_count, plan.grid.no_data),
      std::vector<uint8_t>(sample_count, 0U),
  };

  const int64_t stride = static_cast<int64_t>(plan.grid.tile_cell_count);
  const int64_t chunk_column_start = key.column * stride;
  const int64_t chunk_row_start = key.row * stride;
  const int64_t chunk_column_end = chunk_column_start + static_cast<int64_t>(side);
  const int64_t chunk_row_end = chunk_row_start + static_cast<int64_t>(side);

  // Contributors retain catalogue path order. As in the Python reference,
  // the first valid source sample wins where source rasters overlap.
  for (uint32_t contributor_index : contributor_indices) {
    if (contributor_index >= plan.sources.size()) {
      throw std::logic_error("Rechunk plan contains an invalid source index");
    }
    const IndexedSource &indexed = plan.sources[contributor_index];
    const SourceRaster &source = catalogue.sources()[indexed.source_index];
    const int64_t source_column_end = indexed.column + static_cast<int64_t>(source.width);
    const int64_t source_row_end = indexed.row + static_cast<int64_t>(source.height);
    const int64_t column_start = std::max(chunk_column_start, indexed.column);
    const int64_t column_end = std::min(chunk_column_end, source_column_end);
    const int64_t row_start = std::max(chunk_row_start, indexed.row);
    const int64_t row_end = std::min(chunk_row_end, source_row_end);
    if (column_start >= column_end || row_start >= row_end) {
      continue;
    }

    const uint32_t width = static_cast<uint32_t>(column_end - column_start);
    const uint32_t height = static_cast<uint32_t>(row_end - row_start);
    const size_t window_count = static_cast<size_t>(width) * height;
    std::vector<float> source_values(window_count);
    GdalDatasetPointer dataset = open_source(source.path);
    GDALRasterBand *band = dataset->GetRasterBand(1);
    if (band->RasterIO(
            GF_Read,
            static_cast<int>(column_start - indexed.column),
            static_cast<int>(row_start - indexed.row),
            static_cast<int>(width),
            static_cast<int>(height),
            source_values.data(),
            static_cast<int>(width),
            static_cast<int>(height),
            GDT_Float32,
            0,
            0,
            nullptr
        ) != CE_None) {
      throw std::runtime_error(gdal_error("Could not read terrain from " + source.path.string()));
    }

    std::vector<uint8_t> valid_mask;
    if ((source.mask_flags & GMF_ALL_VALID) == 0) {
      valid_mask.resize(window_count);
      GDALRasterBand *mask_band = band->GetMaskBand();
      if (mask_band == nullptr || mask_band->RasterIO(
                                      GF_Read,
                                      static_cast<int>(column_start - indexed.column),
                                      static_cast<int>(row_start - indexed.row),
                                      static_cast<int>(width),
                                      static_cast<int>(height),
                                      valid_mask.data(),
                                      static_cast<int>(width),
                                      static_cast<int>(height),
                                      GDT_Byte,
                                      0,
                                      0,
                                      nullptr
                                  ) != CE_None) {
        throw std::runtime_error(
            gdal_error("Could not read terrain mask from " + source.path.string())
        );
      }
    }

    const size_t destination_row_start = static_cast<size_t>(row_start - chunk_row_start);
    const size_t destination_column_start = static_cast<size_t>(column_start - chunk_column_start);
    for (uint32_t row = 0U; row < height; row++) {
      for (uint32_t column = 0U; column < width; column++) {
        const size_t source_offset = static_cast<size_t>(row) * width + column;
        const size_t destination_offset =
            (destination_row_start + row) * side + destination_column_start + column;
        const bool mask_valid = valid_mask.empty() || valid_mask[source_offset] != 0U;
        if (chunk.covered[destination_offset] != 0U || !mask_valid ||
            !valid_value(source_values[source_offset], source)) {
          continue;
        }

        const double elevation =
            static_cast<double>(source_values[source_offset]) * source.scale + source.offset;
        if (!std::isfinite(elevation) ||
            elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
            elevation > static_cast<double>(std::numeric_limits<float>::max())) {
          throw std::runtime_error(
              "Scaled terrain elevation cannot be represented as Float32 in " + source.path.string()
          );
        }
        chunk.elevations[destination_offset] = static_cast<float>(elevation);
        chunk.covered[destination_offset] = 1U;
      }
    }
  }
  return chunk;
}

} // namespace panorama::terrain
