#pragma once

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace panorama {

inline constexpr std::array<char, 8> kMetalTileMagic = {'P', 'N', 'T', 'I', 'L', 'E', '0', '2'};
inline constexpr uint32_t kMetalTileVersion = 2U;

/// Compression applied to the complete logical Metal tile stream.
enum class MetalTileCompression : uint32_t {
  None = 0U,
  Zlib = 1U,
  Lzfse = 2U,
  Lz4 = 3U,
  Lzma = 4U,
  LzBitmap = 5U,
};

/// Scalar sample representation stored in the terrain payload.
enum class MetalTileSampleType : uint32_t {
  Float32 = 1U,
};

/// Fixed little-endian header at decompressed stream offset zero.
///
/// All offsets address the logical decompressed stream understood by Metal
/// I/O. The sole payload is already in the exact flat south-to-north layout
/// consumed by the raytracer's vertex atlas. The mipmap is generated on-GPU.
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
  int64_t row;
  int64_t column;
  double lower_left_x;
  double lower_left_y;
  double cell_size;
  uint64_t vertex_offset;
  uint64_t vertex_byte_count;
};

/// Header and CPU payload used only where host access is explicitly required.
struct MetalTileData {
  MetalTileHeader header;
  std::vector<float> vertices;
};

/// One vertex payload to decompress into a reserved atlas-buffer range.
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

/// Read and decompress a custom tile into host memory for diagnostics.
///
/// Normal ray tracing should use `load_metal_tiles_into_buffer` so vertices do
/// not pass through an intermediate host vector.
[[nodiscard]] MetalTileData read_metal_tile(const std::filesystem::path &path);

/// Read and validate only the fixed header, without retaining terrain data.
///
/// Compressed files use a small Metal I/O request because their header is
/// part of the compressed logical stream. Raw files use an ordinary read.
[[nodiscard]] MetalTileHeader read_metal_tile_header(const std::filesystem::path &path);

/// Create the Metal I/O queue shared by direct atlas installations.
[[nodiscard]] id<MTLIOCommandQueue> make_metal_io_queue(id<MTLDevice> device);

/// Open one raw or compressed custom tile for a later direct-I/O command.
[[nodiscard]] id<MTLIOFileHandle>
open_metal_tile_file(id<MTLDevice> device, const std::filesystem::path &path);

/// Return the number of command buffers accepted concurrently by our I/O queue.
[[nodiscard]] NSUInteger metal_tile_io_concurrency();

/// Submit direct atlas loads without waiting for their completion.
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

/// Decompress several vertex payloads directly into reserved atlas slots.
///
/// The caller has already validated the dataset layout from its origin tile.
/// Each request therefore reads only the fixed-size vertex payload rather than
/// redundantly decompressing the header again. Independent I/O command buffers
/// allow the concurrent queue to service several files at once.
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
