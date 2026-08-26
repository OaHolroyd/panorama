#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace panorama::terrain {

/// Interpretation of the samples written to one prepared terrain tile.
enum class RasterLayout {
  // An N-cell tile stores an overlapping (N + 1) by (N + 1) vertex grid.
  Level0,
  // An N-cell tile stores N by N cell-centred elevation values.
  Level1,
};

/// Pixel registration declared by a source raster's AREA_OR_POINT metadata.
enum class RasterRegistration {
  PixelIsArea,
  PixelIsPoint,
};

/// Metadata collected without reading the elevation pixels from one DEM source.
struct SourceRaster {
  std::filesystem::path path;
  // North-up affine transform whose origin is the centre of source sample
  // (0, 0). GDAL's corner-based transform is normalised during discovery for
  // PixelIsPoint rasters such as SRTM HGT.
  std::array<double, 6> transform;
  uint32_t width;
  uint32_t height;
  int data_type;
  std::optional<double> no_data;
  double scale;
  double offset;
  int mask_flags;
  RasterRegistration registration;
};

/// Common grid metadata shared by every accepted input raster.
struct SourceGrid {
  std::string projection_wkt;
  std::string coordinate_system_name;
  std::string elevation_unit;
  double x_resolution;
  double y_resolution;
  double reference_x;
  double reference_y;
  RasterRegistration registration;
};

/// User-selected regular grid on which prepared terrain tiles are written.
struct DestinationGrid {
  double origin_x;
  double origin_y;
  double resolution;
  uint32_t tile_cell_count;
  RasterLayout layout;
  float no_data;
};

/// Signed row and column of one tile on the destination grid.
struct ChunkKey {
  int64_t row;
  int64_t column;

  /// Order chunks deterministically by row and then by column.
  [[nodiscard]] bool operator<(const ChunkKey &other) const {
    if (row != other.row) {
      return row < other.row;
    }
    return column < other.column;
  }
};

/// One input raster expressed as integer sample offsets on the destination grid.
struct IndexedSource {
  uint32_t source_index;
  int64_t row;
  int64_t column;
};

/// Complete mapping from output chunks to their potentially overlapping inputs.
struct RechunkPlan {
  DestinationGrid grid;
  std::vector<IndexedSource> sources;
  std::map<ChunkKey, std::vector<uint32_t>> contributors;
};

/// Float32 output samples and a byte-per-sample valid-coverage mask.
struct TerrainChunk {
  uint32_t sample_side;
  std::vector<float> elevations;
  std::vector<uint8_t> covered;
};

} // namespace panorama::terrain
