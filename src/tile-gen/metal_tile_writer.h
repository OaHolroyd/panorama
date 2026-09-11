#pragma once

#include "metal_tile.h"
#include "terrain_types.h"

#include <filesystem>
#include <string>

namespace panorama::terrain {

struct TerrainElevationRange {
  float minimum;
  float maximum;
  std::optional<TerrainCellCoverage> coverage = std::nullopt;
};

[[nodiscard]] TerrainCellCoverage terrain_chunk_coverage(const TerrainChunk &chunk);

/// Scan existing payloads when upgrading an old manifest. The caller reuses
/// one I/O queue across files; no source rasters need to be regenerated.
[[nodiscard]] TerrainElevationRange read_metal_tile_elevation_range(
    const std::filesystem::path &path,
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue
);

/// Return the stable path for one raw or compressed Metal terrain tile.
[[nodiscard]] std::filesystem::path metal_tile_chunk_path(
    const std::filesystem::path &output_directory,
    const std::string &dataset_name,
    const DestinationGrid &grid,
    ChunkKey key,
    MetalTileCompression compression
);

/// Convert one level-0 chunk into a uint16 fixed-point Metal terrain tile.
///
/// The source chunk is north-to-south for conventional GIS writers. This
/// writer flips it once into atlas order. Uint16 output quantizes onto a global
/// decimetre lattice and stores offsets from a per-tile integer base. The
/// renderer can expand Uint16 during atlas installation; by default the
/// renderer instead retains it through tracing.
/// Returns conservative elevation bounds enclosing every stored LOD.
[[nodiscard]] TerrainElevationRange write_metal_tile_chunk(
    const std::filesystem::path &path,
    const TerrainChunk &chunk,
    const DestinationGrid &grid,
    ChunkKey key,
    const SourceGrid &source_grid,
    MetalTileCompression compression
);

} // namespace panorama::terrain
