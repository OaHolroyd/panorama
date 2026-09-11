#pragma once

#import <Metal/Metal.h>

#include "terrain_cell_coverage.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace panorama {

inline constexpr std::array<char, 8> kMetalTileLodMagic = {'P', 'N', 'T', 'I', 'L', 'E', '0', '5'};
inline constexpr uint32_t kMetalTileLodVersion = 5U;
inline constexpr std::array<char, 8> kLegacyMetalTileLodMagic =
    {'P', 'N', 'T', 'I', 'L', 'E', '0', '4'};
inline constexpr uint32_t kMetalTileLodHeaderSize = 128U;

/// Compression applied to the complete logical Metal tile stream.
enum class MetalTileCompression : uint32_t {
  None = 0U,
  Zlib = 1U,
  Lz4 = 3U,
  Lzma = 4U,
  LzBitmap = 5U,
};

/// Scalar sample representation stored in the terrain payload. Value 1 was
/// Float32 in older files and is no longer supported; keep uint16's wire ID.
enum class MetalTileSampleType : uint32_t {
  Uint16Decimeters = 2U,
};

/// Fixed little-endian header at decompressed stream offset zero.
///
/// All offsets address the logical decompressed stream understood by Metal I/O.
/// Versions 4 and 5 share this header and independently addressable LODs.
/// Version 5 reserves uint16 code zero for no-data; valid elevations decode as
/// (base + code) / 10 metres. Coverage rectangles follow the LOD table.
struct MetalTileHeader {
  std::array<char, 8> magic;
  uint32_t version;
  uint32_t header_size;
  MetalTileCompression compression;
  uint32_t epsg_code;
  uint32_t cell_count;
  uint32_t level_count;
  float maximum_elevation;
  MetalTileSampleType sample_type;
  int32_t elevation_base_decimeters;
  /// Version 5: coverage rectangle count after the LOD table; zero means full.
  uint32_t reserved;
  int64_t row;
  int64_t column;
  double lower_left_x;
  double lower_left_y;
  double cell_size;
  uint64_t vertex_offset;
  uint64_t vertex_byte_count;
  uint32_t lod_count;
  uint32_t lod_entry_size;
  uint64_t lod_table_offset;
  uint64_t lod_table_byte_count;
};

/// One independently addressable vertex payload within a Metal tile.
struct MetalTileLod {
  uint32_t lod;
  uint32_t cell_count;
  uint32_t level_count;
  int32_t elevation_base_decimeters;
  float maximum_elevation;
  uint32_t reserved;
  uint64_t vertex_offset;
  uint64_t vertex_byte_count;
};

/// Compact GPU record containing only the selected vertices, base and validity flag.
struct QuantizedMetalTileRecordLayout {
  uint32_t logical_size;
  uint32_t stride;
  uint32_t vertex_offset;
  uint32_t elevation_base_offset;
  uint32_t no_data_offset;
};

/// One source file, logical byte range, and destination for a Metal I/O load.
struct MetalTileBufferLoad {
  std::filesystem::path path;
  NSUInteger destination_offset;
  id<MTLIOFileHandle> file;
  uint64_t source_offset;
  uint64_t byte_count;
};

/// Return whether a path has one of the custom tile format's suffixes.
[[nodiscard]] bool is_metal_tile_path(const std::filesystem::path &path);

/// Return the compression method encoded by a custom tile's suffix.
[[nodiscard]] MetalTileCompression metal_tile_compression(const std::filesystem::path &path);

/// Return the filename suffix associated with a compression method.
[[nodiscard]] const char *metal_tile_suffix(MetalTileCompression compression);

/// Return the complete number of samples in an N-cell maximum mipmap.
[[nodiscard]] uint64_t metal_tile_mipmap_value_count(uint32_t cell_count);

/// Derive the aligned resident/staging layout of a validated uint16 record.
[[nodiscard]] QuantizedMetalTileRecordLayout
quantized_metal_tile_record_layout(const MetalTileHeader &header);

/// Validate the version, layout, offsets, and compression of one header.
void validate_metal_tile_header(
    const MetalTileHeader &header,
    MetalTileCompression expected_compression
);

/// Read and validate the complete LOD table of a Metal tile.
[[nodiscard]] std::vector<MetalTileLod>
read_metal_tile_lods(const std::filesystem::path &path, const MetalTileHeader &header);

/// Version 4 has no recoverable mask. Version 5 metadata needs no height reads.
[[nodiscard]] std::optional<TerrainCellCoverage>
read_metal_tile_coverage(const std::filesystem::path &path, const MetalTileHeader &header);

/// Write a versioned logical stream containing a table and all LOD payloads.
void write_metal_tile_lods(
    const std::filesystem::path &path,
    const MetalTileHeader &header,
    std::span<const MetalTileLod> lods,
    std::span<const std::byte> payload
);

/// Read and validate the current-format header, without retaining terrain data.
///
/// Compressed files use a small Metal I/O request because their header is
/// part of the compressed logical stream. Raw files use an ordinary read.
[[nodiscard]] MetalTileHeader read_metal_tile_header(const std::filesystem::path &path);

/// Create the Metal I/O queue shared by GPU-buffer tile loads.
[[nodiscard]] id<MTLIOCommandQueue> make_metal_io_queue(id<MTLDevice> device);

/// Open one raw or compressed custom tile for a later direct-I/O command.
[[nodiscard]] id<MTLIOFileHandle>
open_metal_tile_file(id<MTLDevice> device, const std::filesystem::path &path);

/// Return the number of command buffers accepted concurrently by our I/O queue.
[[nodiscard]] NSUInteger metal_tile_io_concurrency();

/// Return the logical alignment of independently readable compressed blocks.
[[nodiscard]] size_t metal_tile_compression_chunk_size();

/// Load independently selected logical byte ranges into a Metal buffer.
///
/// Independent I/O command buffers allow the concurrent queue to service
/// several files at once. Metal I/O decompresses compressed sources; raw
/// sources are copied directly. The caller selects the common source range and
/// each load supplies its source range and destination byte offset.
void load_metal_tiles_into_buffer(
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue,
    std::span<const MetalTileBufferLoad> loads,
    id<MTLBuffer> vertex_buffer,
    NSUInteger vertex_buffer_length
);

} // namespace panorama
