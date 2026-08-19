#include "geotiff_writer.h"

#include "gdal_utils.h"

#include <cpl_string.h>
#include <gdal_priv.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace panorama::terrain {
namespace {

/// Return the on-disk name for a sample interpretation.
[[nodiscard]] const char *layout_name(RasterLayout layout) {
  return layout == RasterLayout::Level0 ? "level-0" : "level-1";
}

/// Describe a power-of-two cell count compatibly with the existing filenames.
[[nodiscard]] std::string size_name(uint32_t tile_cell_count) {
  if (std::has_single_bit(tile_cell_count)) {
    return "p" + std::to_string(std::countr_zero(tile_cell_count));
  }
  return "n" + std::to_string(tile_cell_count);
}

/// Choose a valid tiled-TIFF block size no larger than the usual 256 samples.
[[nodiscard]] uint32_t geotiff_block_size(uint32_t sample_count) {
  if (sample_count >= 256U) {
    return 256U;
  }
  // TIFF tile dimensions must be divisible by 16. A tile may extend past the
  // raster edge, so round small outputs up rather than falling back to strips.
  return std::max(16U, ((sample_count + 15U) / 16U) * 16U);
}

} // namespace

std::filesystem::path geotiff_chunk_path(
    const std::filesystem::path &output_directory,
    const std::string &dataset_name,
    const DestinationGrid &grid,
    ChunkKey key
) {
  return output_directory / (terrain_chunk_stem(dataset_name, grid, key) + ".tif");
}

std::string
terrain_chunk_stem(const std::string &dataset_name, const DestinationGrid &grid, ChunkKey key) {
  return dataset_name + "_" + layout_name(grid.layout) + "_" + size_name(grid.tile_cell_count) +
         "_r" + std::to_string(key.row) + "_c" + std::to_string(key.column);
}

void write_geotiff_chunk(
    const std::filesystem::path &path,
    const TerrainChunk &chunk,
    const DestinationGrid &grid,
    ChunkKey key,
    const SourceGrid &source_grid
) {
  GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
  if (driver == nullptr) {
    throw std::runtime_error("GDAL GeoTIFF writer is unavailable");
  }

  const uint32_t block_size = geotiff_block_size(chunk.sample_side);
  char **options = nullptr;
  options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
  options = CSLSetNameValue(options, "PREDICTOR", "3");
  options = CSLSetNameValue(options, "TILED", "YES");
  const std::string block_size_text = std::to_string(block_size);
  options = CSLSetNameValue(options, "BLOCKXSIZE", block_size_text.c_str());
  options = CSLSetNameValue(options, "BLOCKYSIZE", block_size_text.c_str());

  GDALDataset *raw_dataset = driver->Create(
      path.string().c_str(),
      static_cast<int>(chunk.sample_side),
      static_cast<int>(chunk.sample_side),
      1,
      GDT_Float32,
      options
  );
  CSLDestroy(options);
  if (raw_dataset == nullptr) {
    throw std::runtime_error(gdal_error("Could not create GeoTIFF " + path.string()));
  }
  GdalDatasetPointer dataset(raw_dataset);

  // Prepared GeoTIFFs retain the conventional north-up raster layout: row
  // zero begins at `top` and subsequent rows move south by one resolution.
  const double stride = static_cast<double>(grid.tile_cell_count) * grid.resolution;
  double transform[6] = {
      grid.origin_x + static_cast<double>(key.column) * stride,
      grid.resolution,
      0.0,
      grid.origin_y - static_cast<double>(key.row) * stride,
      0.0,
      -grid.resolution,
  };
  if (dataset->SetGeoTransform(transform) != CE_None ||
      dataset->SetProjection(source_grid.projection_wkt.c_str()) != CE_None) {
    throw std::runtime_error(gdal_error("Could not georeference GeoTIFF " + path.string()));
  }
  dataset->SetMetadataItem(GDALMD_AREA_OR_POINT, "Area");

  GDALRasterBand *band = dataset->GetRasterBand(1);
  if (band->SetNoDataValue(static_cast<double>(grid.no_data)) != CE_None) {
    throw std::runtime_error(gdal_error("Could not set GeoTIFF no-data value " + path.string()));
  }
  if (!source_grid.elevation_unit.empty() &&
      band->SetUnitType(source_grid.elevation_unit.c_str()) != CE_None) {
    throw std::runtime_error(gdal_error("Could not set GeoTIFF elevation unit " + path.string()));
  }
  if (band->RasterIO(
          GF_Write,
          0,
          0,
          static_cast<int>(chunk.sample_side),
          static_cast<int>(chunk.sample_side),
          const_cast<float *>(chunk.elevations.data()),
          static_cast<int>(chunk.sample_side),
          static_cast<int>(chunk.sample_side),
          GDT_Float32,
          0,
          0,
          nullptr
      ) != CE_None) {
    throw std::runtime_error(gdal_error("Could not write terrain to " + path.string()));
  }
}

} // namespace panorama::terrain
