#pragma once

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace panorama {

inline constexpr std::array<char, 8> kMetalTileFloat32Magic =
    {'P', 'N', 'T', 'I', 'L', 'E', '0', '2'};
inline constexpr uint32_t kMetalTileFloat32Version = 2U;
inline constexpr uint32_t kMetalTileFloat32HeaderSize = 96U;
inline constexpr std::array<char, 8> kMetalTileUint16Magic =
    {'P', 'N', 'T', 'I', 'L', 'E', '0', '3'};
inline constexpr uint32_t kMetalTileUint16Version = 3U;
inline constexpr uint32_t kMetalTileUint16HeaderSize = 104U;

/// Compression applied to the complete logical Metal tile stream.
enum class MetalTileCompression : uint32_t {
  None = 0U,
  Zlib = 1U,
  Lz4 = 3U,
  Lzma = 4U,
  LzBitmap = 5U,
};

/// Scalar sample representation stored in the terrain payload.
enum class MetalTileSampleType : uint32_t {
  Float32 = 1U,
  Uint16Decimeters = 2U,
};

/// Fixed little-endian header at decompressed stream offset zero.
///
/// All offsets address the logical decompressed stream understood by Metal
/// I/O. Version 2 stores Float32 vertices directly; version 3 stores Uint16
/// decimetre offsets from `elevation_base_decimeters`. Readers normalize both
/// versions into this in-memory structure.
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
  uint32_t reserved; // Written as zero; ignored when reading early version-3 files.
  int64_t row;
  int64_t column;
  double lower_left_x;
  double lower_left_y;
  double cell_size;
  uint64_t vertex_offset;
  uint64_t vertex_byte_count;
};

/// Header and decoded Float32 CPU payload used only for diagnostics.
struct MetalTileData {
  MetalTileHeader header;
  std::vector<float> vertices;
};

/// Aligned GPU layout for one complete uint16 logical tile record.
struct QuantizedMetalTileRecordLayout {
  uint32_t logical_size;
  uint32_t stride;
  uint32_t vertex_offset;
  uint32_t elevation_base_offset;
};

/// One logical byte range to load into a reserved Metal-buffer range.
struct MetalTileBufferLoad {
  std::filesystem::path path;
  NSUInteger destination_offset;
  id<MTLIOFileHandle> file;
};

/// One committed direct-I/O batch and the event it signals on completion.
///
/// The command and file handles remain retained until the owning installation
/// batch publishes its atlas slots. The mipmap command waits for `event` rather
/// than making the host wait for the I/O command buffer.
struct MetalTileIoSubmission {
  std::vector<std::filesystem::path> paths;
  std::vector<id<MTLIOFileHandle>> files;
  id<MTLIOCommandBuffer> command;
  id<MTLSharedEvent> event;
};

/// Return whether a path has one of the custom tile format's suffixes.
[[nodiscard]] bool is_metal_tile_path(const std::filesystem::path &path);

/// Return the compression method encoded by a custom tile's suffix.
[[nodiscard]] MetalTileCompression metal_tile_compression(const std::filesystem::path &path);

/// Return the filename suffix associated with a compression method.
[[nodiscard]] const char *metal_tile_suffix(MetalTileCompression compression);

/// Return the complete number of Float32 values in an N-cell maximum mipmap.
[[nodiscard]] uint64_t metal_tile_mipmap_value_count(uint32_t cell_count);

/// Derive the aligned resident/staging layout of a validated uint16 record.
[[nodiscard]] QuantizedMetalTileRecordLayout
quantized_metal_tile_record_layout(const MetalTileHeader &header);

/// Validate the version, layout, offsets, and compression of one header.
void validate_metal_tile_header(
    const MetalTileHeader &header,
    MetalTileCompression expected_compression
);

/// Write one raw or Metal-compressed logical tile stream.
void write_metal_tile(
    const std::filesystem::path &path,
    const MetalTileHeader &header,
    std::span<const float> vertices
);

/// Write one fixed-point raw or Metal-compressed logical tile stream.
void write_metal_tile(
    const std::filesystem::path &path,
    const MetalTileHeader &header,
    std::span<const uint16_t> vertices
);

/// Read and decompress a custom tile into host memory for diagnostics.
///
/// Normal ray tracing stages records with `load_metal_tiles_into_buffer` so
/// vertices do not pass through an intermediate host vector.
[[nodiscard]] MetalTileData read_metal_tile(const std::filesystem::path &path);

/// Read and validate only the fixed header, without retaining terrain data.
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

/// Submit Metal-buffer loads without waiting for their completion.
///
/// Every returned command signals its own shared event after its vertex load.
/// A GPU command can wait for all returned events before reading those slots.
[[nodiscard]] std::vector<MetalTileIoSubmission> submit_metal_tiles_into_buffer(
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue,
    std::span<const MetalTileBufferLoad> loads,
    uint64_t vertex_offset,
    uint64_t vertex_byte_count,
    id<MTLBuffer> vertex_buffer,
    NSUInteger vertex_buffer_length
);

/// Decompress equal logical byte ranges into reserved Metal-buffer ranges.
///
/// Independent I/O command buffers allow the concurrent queue to service
/// several files at once. The caller selects the logical source range and each
/// load supplies its destination byte offset.
void load_metal_tiles_into_buffer(
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue,
    std::span<const MetalTileBufferLoad> loads,
    uint64_t vertex_offset,
    uint64_t vertex_byte_count,
    id<MTLBuffer> vertex_buffer,
    NSUInteger vertex_buffer_length
);

} // namespace panorama
