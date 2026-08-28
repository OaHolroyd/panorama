#pragma once

#include "metal_tile.h"
#include "terrain_types.h"

#include <filesystem>
#include <string>

namespace panorama::terrain {

/// Return the stable path for one raw or compressed Metal terrain tile.
[[nodiscard]] std::filesystem::path metal_tile_chunk_path(
    const std::filesystem::path &output_directory,
    const std::string &dataset_name,
    const DestinationGrid &grid,
    ChunkKey key,
    MetalTileCompression compression
);

/// Convert one level-0 chunk into a Float32 or fixed-point Metal terrain tile.
///
/// The source chunk is north-to-south for conventional GIS writers. This
/// writer flips it once into atlas order. Uint16 output quantizes onto a global
/// decimetre lattice and stores offsets from a per-tile integer base. The
/// renderer can expand Uint16 during atlas installation; by default the
/// renderer instead retains it through tracing.
/// Returns the exact maximum elevation recorded in the tile header.
[[nodiscard]] float write_metal_tile_chunk(
    const std::filesystem::path &path,
    const TerrainChunk &chunk,
    const DestinationGrid &grid,
    ChunkKey key,
    const SourceGrid &source_grid,
    MetalTileCompression compression,
    MetalTileSampleType sample_type
);

} // namespace panorama::terrain
