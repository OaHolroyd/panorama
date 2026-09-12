#include "terrain_manifest.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <type_traits>

namespace panorama {
namespace {

inline constexpr std::array<char, 8> kTerrainManifestMagic =
    {'P', 'N', 'M', 'A', 'N', '0', '0', '1'};
inline constexpr uint32_t kTerrainManifestVersion = 3U;

struct TerrainManifestHeader {
  std::array<char, 8> magic;
  uint32_t version;
  uint32_t header_size;
  uint32_t entry_size;
  uint32_t entry_count;
};

struct TerrainManifestDiskEntry {
  int64_t row;
  int64_t column;
  float maximum_elevation;
  // Version 1 used these four bytes as a zero reserved word.
  float minimum_elevation;
};

struct TerrainManifestCoverageEntry {
  TerrainManifestDiskEntry elevation;
  uint32_t cell_count;      // Zero denotes a legacy tile without validity metadata.
  uint32_t rectangle_count; // Zero denotes full coverage for a nonzero cell count.
  uint64_t coverage_offset;
};
static_assert(sizeof(TerrainManifestCoverageEntry) == 40U);

static_assert(std::endian::native == std::endian::little);
static_assert(std::is_trivially_copyable_v<TerrainManifestHeader>);
static_assert(std::is_trivially_copyable_v<TerrainManifestDiskEntry>);
static_assert(sizeof(TerrainManifestHeader) == 24U);
static_assert(sizeof(TerrainManifestDiskEntry) == 24U);

/// Publish a completed temporary file without exposing a partial manifest.
void publish_manifest(
    const std::filesystem::path &temporary,
    const std::filesystem::path &destination
) {
  std::error_code error;
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error(
        "Could not publish terrain manifest " + destination.string() + ": " + error.message()
    );
  }
}

} // namespace

std::filesystem::path terrain_manifest_path(const std::filesystem::path &directory) {
  return directory / kTerrainManifestFilename;
}

std::vector<TerrainManifestEntry> read_terrain_manifest(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  TerrainManifestHeader header = {};
  if (!stream.read(reinterpret_cast<char *>(&header), sizeof(header)) ||
      header.magic != kTerrainManifestMagic ||
      (header.version < 1U || header.version > kTerrainManifestVersion) ||
      header.header_size != sizeof(TerrainManifestHeader) ||
      header.entry_size != (header.version < 3U ? sizeof(TerrainManifestDiskEntry)
                                                : sizeof(TerrainManifestCoverageEntry))) {
    throw std::runtime_error("Terrain manifest has an unsupported header: " + path.string());
  }
  uintmax_t expected_size = sizeof(TerrainManifestHeader) +
                            static_cast<uintmax_t>(header.entry_count) * header.entry_size;
  const auto file_size = std::filesystem::file_size(path);
  if (file_size < expected_size) {
    throw std::runtime_error("Terrain manifest has an invalid size: " + path.string());
  }

  std::vector<TerrainManifestEntry> entries;
  entries.reserve(header.entry_count);
  for (uint32_t index = 0U; index < header.entry_count; index++) {
    stream.seekg(sizeof(TerrainManifestHeader) + uint64_t(index) * header.entry_size);
    TerrainManifestDiskEntry disk = {};
    if (!stream.read(reinterpret_cast<char *>(&disk), sizeof(disk)) ||
        !std::isfinite(disk.maximum_elevation) ||
        (header.version == 1U ? std::bit_cast<uint32_t>(disk.minimum_elevation) != 0U
                              : !std::isfinite(disk.minimum_elevation) ||
                                    disk.minimum_elevation > disk.maximum_elevation)) {
      throw std::runtime_error("Terrain manifest contains an invalid entry: " + path.string());
    }
    entries.push_back(
        {disk.row,
         disk.column,
         disk.maximum_elevation,
         header.version == 1U ? std::nullopt : std::optional<float>(disk.minimum_elevation)}
    );
    if (header.version == 3U) {
      uint32_t cells = 0, count = 0;
      uint64_t offset = 0;
      stream.read(reinterpret_cast<char *>(&cells), sizeof(cells));
      stream.read(reinterpret_cast<char *>(&count), sizeof(count));
      stream.read(reinterpret_cast<char *>(&offset), sizeof(offset));
      const uint64_t bytes = uint64_t(count) * sizeof(TerrainCoverageRect);
      if (!stream || offset != expected_size || bytes > file_size - expected_size ||
          (cells == 0U && count != 0U) || (cells && count > uint64_t(cells) * cells))
        throw std::runtime_error("Terrain manifest has invalid coverage offsets");
      expected_size += bytes;
      if (cells) {
        TerrainCellCoverage coverage{cells, {}};
        if (count == 0U)
          coverage.rectangles.push_back({0, 0, cells, cells});
        else {
          coverage.rectangles.resize(count);
          stream.seekg(static_cast<std::streamoff>(offset));
          if (!stream.read(
                  reinterpret_cast<char *>(coverage.rectangles.data()),
                  static_cast<std::streamsize>(bytes)
              ))
            throw std::runtime_error("Could not read manifest coverage");
        }
        coverage.validate();
        entries.back().coverage = std::move(coverage);
      }
    }
  }
  if (file_size != expected_size)
    throw std::runtime_error("Terrain manifest has an invalid size: " + path.string());
  return entries;
}

void write_terrain_manifest(
    const std::filesystem::path &path,
    std::span<const TerrainManifestEntry> entries
) {
  if (entries.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::overflow_error("Terrain manifest contains too many entries");
  }
  const bool with_coverage = std::any_of(entries.begin(), entries.end(), [](const auto &entry) {
    return entry.coverage.has_value();
  });
  const uint32_t entry_size =
      with_coverage ? sizeof(TerrainManifestCoverageEntry) : sizeof(TerrainManifestDiskEntry);
  uint64_t coverage_offset = sizeof(TerrainManifestHeader) + uint64_t(entries.size()) * entry_size;
  for (const auto &entry : entries) {
    if (entry.coverage) {
      entry.coverage->validate();
      if (entry.coverage->rectangles.empty())
        throw std::invalid_argument("Empty terrain tiles must not be published");
    }
  }
  const std::filesystem::path temporary = path.string() + ".tmp";
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  const TerrainManifestHeader header = {
      kTerrainManifestMagic,
      with_coverage ? kTerrainManifestVersion : 2U,
      sizeof(TerrainManifestHeader),
      entry_size,
      static_cast<uint32_t>(entries.size()),
  };
  stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
  for (const TerrainManifestEntry &entry : entries) {
    if (!std::isfinite(entry.maximum_elevation) || !entry.minimum_elevation.has_value() ||
        !std::isfinite(*entry.minimum_elevation) ||
        *entry.minimum_elevation > entry.maximum_elevation) {
      stream.close();
      std::filesystem::remove(temporary);
      throw std::invalid_argument(
          "Terrain manifest requires finite, ordered minimum and maximum elevations"
      );
    }
    const TerrainManifestDiskEntry disk = {
        entry.row,
        entry.column,
        entry.maximum_elevation,
        *entry.minimum_elevation,
    };
    stream.write(reinterpret_cast<const char *>(&disk), sizeof(disk));
    if (with_coverage) {
      const uint32_t cells = entry.coverage ? entry.coverage->cell_count : 0U;
      const uint32_t count = entry.coverage && !entry.coverage->full()
                                 ? static_cast<uint32_t>(entry.coverage->rectangles.size())
                                 : 0U;
      stream.write(reinterpret_cast<const char *>(&cells), sizeof(cells));
      stream.write(reinterpret_cast<const char *>(&count), sizeof(count));
      stream.write(reinterpret_cast<const char *>(&coverage_offset), sizeof(coverage_offset));
      coverage_offset += uint64_t(count) * sizeof(TerrainCoverageRect);
    }
  }
  if (with_coverage)
    for (const auto &entry : entries)
      if (entry.coverage && !entry.coverage->full())
        stream.write(
            reinterpret_cast<const char *>(entry.coverage->rectangles.data()),
            static_cast<std::streamsize>(
                entry.coverage->rectangles.size() * sizeof(TerrainCoverageRect)
            )
        );
  if (!stream) {
    stream.close();
    std::filesystem::remove(temporary);
    throw std::runtime_error("Could not write terrain manifest " + path.string());
  }
  stream.close();
  publish_manifest(temporary, path);
}

} // namespace panorama
