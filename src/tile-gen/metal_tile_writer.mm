#include "metal_tile_writer.h"

#include <ogr_spatialref.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
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

/// Round a logical stream offset up so a compressed Metal I/O request can
/// begin at a compression-block boundary without inflating earlier variants.
[[nodiscard]] uint64_t align_up(uint64_t value, size_t alignment) {
  if (alignment == 0U || value > std::numeric_limits<uint64_t>::max() - (alignment - 1U)) {
    throw std::overflow_error("Metal tile LOD payload offset overflows");
  }
  const uint64_t divisor = static_cast<uint64_t>(alignment);
  return ((value + divisor - 1U) / divisor) * divisor;
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
  if (grid.layout != RasterLayout::Level0 || !std::has_single_bit(grid.tile_cell_count))
    throw std::invalid_argument("Metal tiles require a power-of-two level-0 destination grid");
  return output_directory /
         (dataset_name + "_level-0_p" + std::to_string(std::countr_zero(grid.tile_cell_count)) +
          "_r" + std::to_string(key.row) + "_c" + std::to_string(key.column) +
          metal_tile_suffix(compression));
}

TerrainElevationRange write_metal_tile_chunk(
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

  struct VariantVertices {
    uint32_t cell_count;
    std::vector<float> values;
  };
  std::vector<VariantVertices> variants;
  variants.push_back({grid.tile_cell_count, std::move(vertices)});
  for (const TerrainChunk::LodVariant &variant : chunk.lod_variants) {
    std::vector<float> values(variant.elevations.size());
    for (uint32_t source_row = 0U; source_row < variant.sample_side; source_row++) {
      const uint32_t destination_row = variant.sample_side - 1U - source_row;
      std::copy_n(
          variant.elevations.begin() + static_cast<size_t>(source_row) * variant.sample_side,
          variant.sample_side,
          values.begin() + static_cast<size_t>(destination_row) * variant.sample_side
      );
    }
    if (!std::all_of(values.begin(), values.end(), [](float value) {
          return std::isfinite(value);
        })) {
      throw std::runtime_error("Metal tile LOD cannot contain non-finite elevations");
    }
    variants.push_back({variant.sample_side - 1U, std::move(values)});
  }
  const uint64_t table_bytes = static_cast<uint64_t>(variants.size()) * sizeof(MetalTileLod);
  const uint64_t metadata_bytes = kMetalTileLodHeaderSize + table_bytes;
  const uint64_t first_payload = align_up(
      metadata_bytes,
      compression == MetalTileCompression::None ? 1U : metal_tile_compression_chunk_size()
  );
  const double tile_width = static_cast<double>(grid.tile_cell_count) * grid.resolution;
  std::vector<MetalTileLod> lods;
  lods.reserve(variants.size());
  std::vector<std::byte> payload(static_cast<size_t>(first_payload - metadata_bytes));
  uint64_t offset = first_payload;
  TerrainElevationRange range{std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()};
  for (uint32_t index = 0U; index < variants.size(); index++) {
    const VariantVertices &variant = variants[index];
    const auto [minimum, maximum] =
        std::minmax_element(variant.values.begin(), variant.values.end());
    const uint64_t byte_count = static_cast<uint64_t>(variant.values.size()) * sizeof(uint16_t);
    const int32_t base = elevation_decimeters(*minimum);
    const float stored_maximum = static_cast<float>(elevation_decimeters(*maximum)) / 10.0F;
    const float stored_minimum = float(base) * 0.1F;
    const float guard = 4.0F * std::numeric_limits<float>::epsilon() *
                        std::max({1.0F, std::abs(stored_minimum), std::abs(stored_maximum)});
    range.minimum = std::min(range.minimum, stored_minimum - guard);
    range.maximum = std::max(range.maximum, stored_maximum + guard);
    if (static_cast<int64_t>(elevation_decimeters(*maximum)) - base >
        static_cast<int64_t>(std::numeric_limits<uint16_t>::max())) {
      throw std::runtime_error("Terrain LOD elevation range exceeds 6553.5 metres");
    }
    lods.push_back(
        {index + 1U,
         variant.cell_count,
         std::countr_zero(variant.cell_count) + 1U,
         base,
         stored_maximum,
         0U,
         offset,
         byte_count}
    );
    const size_t old_size = payload.size();
    payload.resize(old_size + static_cast<size_t>(byte_count));
    auto *destination = reinterpret_cast<uint16_t *>(payload.data() + old_size);
    for (size_t sample = 0U; sample < variant.values.size(); sample++) {
      destination[sample] = static_cast<uint16_t>(
          static_cast<int64_t>(elevation_decimeters(variant.values[sample])) - base
      );
    }
    offset += byte_count;
    if (index + 1U < variants.size()) {
      offset = align_up(
          offset,
          compression == MetalTileCompression::None ? 1U : metal_tile_compression_chunk_size()
      );
      payload.resize(static_cast<size_t>(offset - metadata_bytes));
    }
  }
  const MetalTileLod &base = lods.front();
  const MetalTileHeader header = {
      kMetalTileLodMagic,
      kMetalTileLodVersion,
      kMetalTileLodHeaderSize,
      compression,
      epsg_code(source_grid),
      base.cell_count,
      base.level_count,
      base.maximum_elevation,
      MetalTileSampleType::Uint16Decimeters,
      base.elevation_base_decimeters,
      0U,
      key.row,
      key.column,
      grid.origin_x + static_cast<double>(key.column) * tile_width,
      grid.origin_y - static_cast<double>(key.row + 1) * tile_width,
      grid.resolution,
      base.vertex_offset,
      base.vertex_byte_count,
      static_cast<uint32_t>(lods.size()),
      static_cast<uint32_t>(sizeof(MetalTileLod)),
      kMetalTileLodHeaderSize,
      table_bytes,
  };
  write_metal_tile_lods(path, header, lods, payload);
  return range;
}

TerrainElevationRange read_metal_tile_elevation_range(
    const std::filesystem::path &path,
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue
) {
  const auto header = read_metal_tile_header(path);
  const auto lods = read_metal_tile_lods(path, header);
  TerrainElevationRange range{std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()};
  id<MTLIOFileHandle> file = open_metal_tile_file(device, path);
  for (const auto &lod : lods) {
    @autoreleasepool {
      if (lod.vertex_byte_count > device.maxBufferLength)
        throw std::runtime_error("Tile payload exceeds Metal device limit");
      id<MTLBuffer> data = [device newBufferWithLength:lod.vertex_byte_count
                                               options:MTLResourceStorageModeShared];
      if (data == nil)
        throw std::runtime_error("Could not allocate manifest scan buffer");
      const MetalTileBufferLoad load{path, 0U, file, lod.vertex_offset, lod.vertex_byte_count};
      load_metal_tiles_into_buffer(device, queue, std::span(&load, 1), data, data.length);
      const uint64_t count = (uint64_t(lod.cell_count) + 1U) * (uint64_t(lod.cell_count) + 1U);
      for (uint64_t i = 0; i < count; ++i) {
        const float value = (float(lod.elevation_base_decimeters) +
                             float(static_cast<const uint16_t *>(data.contents)[i])) *
                            0.1F;
        if (!std::isfinite(value))
          throw std::runtime_error("Non-finite terrain sample in manifest scan");
        range.minimum = std::min(range.minimum, value);
        range.maximum = std::max(range.maximum, value);
      }
    }
  }
  const float guard = 4.0F * std::numeric_limits<float>::epsilon() *
                      std::max({1.0F, std::abs(range.minimum), std::abs(range.maximum)});
  return {range.minimum - guard, range.maximum + guard};
}

} // namespace panorama::terrain
