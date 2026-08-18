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

/// Convert one level-0 chunk into the exact two arrays used by the raytracer.
///
/// The source chunk is north-to-south for conventional GIS writers. This
/// writer flips it once and stores the vertices in their final GPU atlas order
/// before applying Metal compression. The renderer builds mipmaps on the GPU.
void write_metal_tile_chunk(
    const std::filesystem::path &path,
    const TerrainChunk &chunk,
    const DestinationGrid &grid,
    ChunkKey key,
    const SourceGrid &source_grid,
    MetalTileCompression compression
);

} // namespace panorama::terrain
