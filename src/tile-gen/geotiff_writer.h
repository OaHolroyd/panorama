#pragma once

#include "terrain_types.h"

#include <filesystem>
#include <string>

namespace panorama::terrain {

/// Return the stable filename for one prepared GeoTIFF terrain chunk.
[[nodiscard]] std::filesystem::path geotiff_chunk_path(
    const std::filesystem::path &output_directory,
    const std::string &dataset_name,
    const DestinationGrid &grid,
    ChunkKey key
);

/// Write one north-up Float32 chunk with standard CRS and no-data metadata.
void write_geotiff_chunk(
    const std::filesystem::path &path,
    const TerrainChunk &chunk,
    const DestinationGrid &grid,
    ChunkKey key,
    const SourceGrid &source_grid
);

} // namespace panorama::terrain
