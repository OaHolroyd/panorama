#pragma once

#include "source_catalogue.h"
#include "terrain_types.h"

#include <cstdint>
#include <span>

namespace panorama::terrain {

/// Return the output sample side for the selected cell count and layout.
[[nodiscard]] uint32_t sample_side(const DestinationGrid &grid);

/// Validate an aligned destination grid and index every overlapping source.
[[nodiscard]] RechunkPlan
make_rechunk_plan(const SourceCatalogue &catalogue, const DestinationGrid &destination);

/// Read and mosaic the source windows contributing to one output chunk.
[[nodiscard]] TerrainChunk build_chunk(
    const SourceCatalogue &catalogue,
    const RechunkPlan &plan,
    ChunkKey key,
    std::span<const uint32_t> contributor_indices
);

} // namespace panorama::terrain
