#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace panorama {

/// Stable filename of the optional directory-level prepared-terrain index.
inline constexpr const char *kTerrainManifestFilename = "panorama-terrain-manifest.bin";

/// The acceleration metadata needed to reject a tile before opening it.
struct TerrainManifestEntry {
  int64_t row;
  int64_t column;
  float maximum_elevation;
};

/// Return the manifest sidecar belonging to a prepared-terrain directory.
[[nodiscard]] std::filesystem::path terrain_manifest_path(const std::filesystem::path &directory);

/// Read and validate a versioned terrain manifest.
[[nodiscard]] std::vector<TerrainManifestEntry>
read_terrain_manifest(const std::filesystem::path &path);

/// Atomically replace a terrain manifest with the supplied entries.
void write_terrain_manifest(
    const std::filesystem::path &path,
    std::span<const TerrainManifestEntry> entries
);

} // namespace panorama
