#include "metal_tile_writer.h"

#include "geotiff_writer.h"

#include <ogr_spatialref.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace panorama::terrain {
namespace {

/// Extract the projected EPSG authority code recorded in the source WKT.
[[nodiscard]] uint32_t epsg_code(const SourceGrid &source_grid) {
  OGRSpatialReference reference;
  if (reference.importFromWkt(source_grid.projection_wkt.c_str()) != OGRERR_NONE) {
    throw std::runtime_error("Could not parse the source CRS for a Metal tile");
  }
  reference.AutoIdentifyEPSG();
  const char *code = reference.GetAuthorityCode(nullptr);
  if (code == nullptr) {
    code = reference.GetAuthorityCode("PROJCS");
  }
  if (code == nullptr) {
    throw std::runtime_error("Metal tiles require a CRS with an EPSG authority code");
  }
  const unsigned long parsed = std::stoul(code);
  if (parsed == 0UL || parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Metal tile EPSG authority code is out of range");
  }
  return static_cast<uint32_t>(parsed);
}

} // namespace

std::filesystem::path metal_tile_chunk_path(
    const std::filesystem::path &output_directory,
    const std::string &dataset_name,
    const DestinationGrid &grid,
    ChunkKey key,
    MetalTileCompression compression
) {
  return output_directory /
         (terrain_chunk_stem(dataset_name, grid, key) + metal_tile_suffix(compression));
}

void write_metal_tile_chunk(
    const std::filesystem::path &path,
    const TerrainChunk &chunk,
    const DestinationGrid &grid,
    ChunkKey key,
    const SourceGrid &source_grid,
    MetalTileCompression compression
) {
  if (grid.layout != RasterLayout::Level0 || chunk.sample_side != grid.tile_cell_count + 1U ||
      !std::has_single_bit(grid.tile_cell_count)) {
    throw std::invalid_argument("Metal tiles require a power-of-two level-0 destination grid");
  }

  // Reverse whole rows into the raytracer's south-to-north convention. No
  // runtime swizzle or temporary decode buffer is needed after this conversion.
  std::vector<float> vertices(chunk.elevations.size());
  for (uint32_t source_row = 0U; source_row < chunk.sample_side; source_row++) {
    const uint32_t destination_row = chunk.sample_side - 1U - source_row;
    std::copy_n(
        chunk.elevations.begin() + static_cast<size_t>(source_row) * chunk.sample_side,
        chunk.sample_side,
        vertices.begin() + static_cast<size_t>(destination_row) * chunk.sample_side
    );
  }
  if (!std::all_of(vertices.begin(), vertices.end(), [](float value) {
        return std::isfinite(value);
      })) {
    throw std::runtime_error("Metal tiles cannot contain non-finite elevations");
  }
  const float maximum = *std::max_element(vertices.begin(), vertices.end());
  const double tile_width = static_cast<double>(grid.tile_cell_count) * grid.resolution;

  const MetalTileHeader header = {
      kMetalTileMagic,
      kMetalTileVersion,
      static_cast<uint32_t>(sizeof(MetalTileHeader)),
      compression,
      epsg_code(source_grid),
      grid.tile_cell_count,
      std::countr_zero(grid.tile_cell_count) + 1U,
      maximum,
      MetalTileSampleType::Float32,
      key.row,
      key.column,
      grid.origin_x + static_cast<double>(key.column) * tile_width,
      grid.origin_y - static_cast<double>(key.row + 1) * tile_width,
      grid.resolution,
      sizeof(MetalTileHeader),
      static_cast<uint64_t>(vertices.size()) * sizeof(float),
  };
  write_metal_tile(path, header, vertices);
}

} // namespace panorama::terrain
