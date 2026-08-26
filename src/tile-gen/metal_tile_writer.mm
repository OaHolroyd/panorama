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

/// Quantize one finite elevation onto the format's global decimetre lattice.
[[nodiscard]] int32_t elevation_decimeters(float elevation) {
  const double scaled = static_cast<double>(elevation) * 10.0;
  if (!std::isfinite(scaled) ||
      scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()) - 0.5 ||
      scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()) + 0.5) {
    throw std::overflow_error("Terrain elevation is outside the fixed-point tile range");
  }
  return static_cast<int32_t>(std::llround(scaled));
}

/// Extract the projected EPSG authority code recorded in the source WKT.
[[nodiscard]] uint32_t epsg_code(const SourceGrid &source_grid) {
  if (source_grid.epsg_code.has_value()) {
    return *source_grid.epsg_code;
  }
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

float write_metal_tile_chunk(
    const std::filesystem::path &path,
    const TerrainChunk &chunk,
    const DestinationGrid &grid,
    ChunkKey key,
    const SourceGrid &source_grid,
    MetalTileCompression compression,
    MetalTileSampleType sample_type
) {
  if (grid.layout != RasterLayout::Level0 || chunk.sample_side != grid.tile_cell_count + 1U ||
      !std::has_single_bit(grid.tile_cell_count)) {
    throw std::invalid_argument("Metal tiles require a power-of-two level-0 destination grid");
  }

  // Reverse whole rows into the raytracer's south-to-north convention so both
  // stored sample representations already match atlas order.
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
  const auto [minimum_vertex, maximum_vertex] =
      std::minmax_element(vertices.begin(), vertices.end());
  const double tile_width = static_cast<double>(grid.tile_cell_count) * grid.resolution;
  if (sample_type == MetalTileSampleType::Float32) {
    const MetalTileHeader header = {
        kMetalTileFloat32Magic,
        kMetalTileFloat32Version,
        kMetalTileFloat32HeaderSize,
        compression,
        epsg_code(source_grid),
        grid.tile_cell_count,
        std::countr_zero(grid.tile_cell_count) + 1U,
        *maximum_vertex,
        MetalTileSampleType::Float32,
        0,
        0U,
        key.row,
        key.column,
        grid.origin_x + static_cast<double>(key.column) * tile_width,
        grid.origin_y - static_cast<double>(key.row + 1) * tile_width,
        grid.resolution,
        kMetalTileFloat32HeaderSize,
        static_cast<uint64_t>(vertices.size()) * sizeof(float),
    };
    write_metal_tile(path, header, vertices);
    return header.maximum_elevation;
  }
  if (sample_type != MetalTileSampleType::Uint16Decimeters) {
    throw std::invalid_argument("Unsupported Metal tile sample type");
  }

  const int32_t minimum_decimeters = elevation_decimeters(*minimum_vertex);
  const int32_t maximum_decimeters = elevation_decimeters(*maximum_vertex);
  const int64_t quantized_range =
      static_cast<int64_t>(maximum_decimeters) - static_cast<int64_t>(minimum_decimeters);
  if (quantized_range > static_cast<int64_t>(std::numeric_limits<uint16_t>::max())) {
    throw std::runtime_error("Metal tile elevation range exceeds 6553.5 metres");
  }

  std::vector<uint16_t> quantized_vertices;
  quantized_vertices.reserve(vertices.size());
  for (float elevation : vertices) {
    const int64_t code = static_cast<int64_t>(elevation_decimeters(elevation)) - minimum_decimeters;
    quantized_vertices.push_back(static_cast<uint16_t>(code));
  }
  const float maximum = static_cast<float>(static_cast<double>(maximum_decimeters) / 10.0);

  const MetalTileHeader header = {
      kMetalTileUint16Magic,
      kMetalTileUint16Version,
      kMetalTileUint16HeaderSize,
      compression,
      epsg_code(source_grid),
      grid.tile_cell_count,
      std::countr_zero(grid.tile_cell_count) + 1U,
      maximum,
      MetalTileSampleType::Uint16Decimeters,
      minimum_decimeters,
      0U,
      key.row,
      key.column,
      grid.origin_x + static_cast<double>(key.column) * tile_width,
      grid.origin_y - static_cast<double>(key.row + 1) * tile_width,
      grid.resolution,
      kMetalTileUint16HeaderSize,
      static_cast<uint64_t>(quantized_vertices.size()) * sizeof(uint16_t),
  };
  write_metal_tile(path, header, quantized_vertices);
  return header.maximum_elevation;
}

} // namespace panorama::terrain
