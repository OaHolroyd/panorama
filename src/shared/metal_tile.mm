#include "metal_tile.h"

#import <Foundation/Foundation.h>
#import <Metal/MTLIOCompressor.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

namespace panorama {
namespace {

static_assert(std::endian::native == std::endian::little);
static_assert(std::is_trivially_copyable_v<MetalTileHeader>);
static_assert(sizeof(MetalTileHeader) == kMetalTileLodHeaderSize);
static_assert(std::is_trivially_copyable_v<MetalTileLod>);
static_assert(sizeof(MetalTileLod) == 40U);

/// Exact version-2 disk header retained for Float32 compatibility.
struct MetalTileFloat32Header {
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

static_assert(std::is_trivially_copyable_v<MetalTileFloat32Header>);
static_assert(sizeof(MetalTileFloat32Header) == kMetalTileFloat32HeaderSize);

/// Exact version-3 disk header retained for single-LOD uint16 compatibility.
struct MetalTileUint16Header {
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
  uint32_t reserved;
  int64_t row;
  int64_t column;
  double lower_left_x;
  double lower_left_y;
  double cell_size;
  uint64_t vertex_offset;
  uint64_t vertex_byte_count;
};

static_assert(std::is_trivially_copyable_v<MetalTileUint16Header>);
static_assert(sizeof(MetalTileUint16Header) == kMetalTileUint16HeaderSize);

using MetalTileHeaderBytes = std::array<std::byte, kMetalTileLodHeaderSize>;

// Keep one I/O wave within the queue's explicit command-buffer and command
// concurrency limits. Later waves reuse the same queue after the previous
// commands have completed.
constexpr NSUInteger kMetalIoConcurrency = 8U;

/// Convert the stable on-disk enum to Metal's compression API value.
[[nodiscard]] MTLIOCompressionMethod compression_method(MetalTileCompression compression) {
  switch (compression) {
  case MetalTileCompression::Zlib:
    return MTLIOCompressionMethodZlib;
  case MetalTileCompression::Lz4:
    return MTLIOCompressionMethodLZ4;
  case MetalTileCompression::Lzma:
    return MTLIOCompressionMethodLZMA;
  case MetalTileCompression::LzBitmap:
    return MTLIOCompressionMethodLZBitmap;
  case MetalTileCompression::None:
    break;
  }
  throw std::invalid_argument("Raw Metal tiles do not have a compression codec");
}

/// Open a raw or compressed Metal I/O file handle.
[[nodiscard]] id<MTLIOFileHandle> open_metal_file(
    id<MTLDevice> device,
    const std::filesystem::path &path,
    MetalTileCompression compression
) {
  NSError *error = nil;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
  id<MTLIOFileHandle> handle = compression == MetalTileCompression::None
                                   ? [device newIOFileHandleWithURL:url error:&error]
                                   : [device newIOFileHandleWithURL:url
                                                  compressionMethod:compression_method(compression)
                                                              error:&error];
  if (handle == nil) {
    const char *detail = error == nil ? "unknown error" : error.localizedDescription.UTF8String;
    throw std::runtime_error("Could not open Metal tile " + path.string() + ": " + detail);
  }
  return handle;
}

/// Wait for an I/O command and turn Metal's asynchronous failure into C++.
void complete_io(id<MTLIOCommandBuffer> command, const std::filesystem::path &path) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLIOStatusComplete) {
    const char *detail =
        command.error == nil ? "unknown error" : command.error.localizedDescription.UTF8String;
    throw std::runtime_error("Could not load Metal tile " + path.string() + ": " + detail);
  }
}

/// Decode either on-disk header version into the common in-memory structure.
[[nodiscard]] MetalTileHeader decode_header(const MetalTileHeaderBytes &bytes) {
  std::array<char, 8> magic = {};
  std::memcpy(magic.data(), bytes.data(), magic.size());
  if (magic == kMetalTileLodMagic) {
    MetalTileHeader header = {};
    std::memcpy(&header, bytes.data(), sizeof(header));
    return header;
  }
  if (magic == kMetalTileUint16Magic) {
    MetalTileUint16Header disk = {};
    std::memcpy(&disk, bytes.data(), sizeof(disk));
    return {disk.magic,
            disk.version,
            disk.header_size,
            disk.compression,
            disk.epsg_code,
            disk.cell_count,
            disk.level_count,
            disk.maximum_elevation,
            disk.sample_type,
            disk.elevation_base_decimeters,
            disk.reserved,
            disk.row,
            disk.column,
            disk.lower_left_x,
            disk.lower_left_y,
            disk.cell_size,
            disk.vertex_offset,
            disk.vertex_byte_count,
            0U,
            0U,
            0U,
            0U};
  }
  if (magic == kMetalTileFloat32Magic) {
    MetalTileFloat32Header disk = {};
    std::memcpy(&disk, bytes.data(), sizeof(disk));
    return {
        disk.magic,
        disk.version,
        disk.header_size,
        disk.compression,
        disk.epsg_code,
        disk.cell_count,
        disk.level_count,
        disk.maximum_elevation,
        disk.sample_type,
        0,
        0U,
        disk.row,
        disk.column,
        disk.lower_left_x,
        disk.lower_left_y,
        disk.cell_size,
        disk.vertex_offset,
        disk.vertex_byte_count,
        0U,
        0U,
        0U,
        0U,
    };
  }
  throw std::runtime_error("Metal tile has an unsupported header or version");
}

/// Return the exact version-3 disk representation of a normalized header.
[[nodiscard]] MetalTileUint16Header uint16_disk_header(const MetalTileHeader &header) {
  return {
      header.magic,
      header.version,
      header.header_size,
      header.compression,
      header.epsg_code,
      header.cell_count,
      header.level_count,
      header.maximum_elevation,
      header.sample_type,
      header.elevation_base_decimeters,
      0U,
      header.row,
      header.column,
      header.lower_left_x,
      header.lower_left_y,
      header.cell_size,
      header.vertex_offset,
      header.vertex_byte_count,
  };
}

/// Read and normalize a raw header without requiring a Metal device.
[[nodiscard]] MetalTileHeader read_raw_header(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  MetalTileHeaderBytes bytes = {};
  if (!stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) {
    throw std::runtime_error("Could not read Metal tile header " + path.string());
  }
  return decode_header(bytes);
}

/// Return the exact version-2 disk representation of a normalized header.
[[nodiscard]] MetalTileFloat32Header float32_disk_header(const MetalTileHeader &header) {
  return {
      header.magic,
      header.version,
      header.header_size,
      header.compression,
      header.epsg_code,
      header.cell_count,
      header.level_count,
      header.maximum_elevation,
      header.sample_type,
      header.row,
      header.column,
      header.lower_left_x,
      header.lower_left_y,
      header.cell_size,
      header.vertex_offset,
      header.vertex_byte_count,
  };
}

void publish_temporary_file(
    const std::filesystem::path &temporary,
    const std::filesystem::path &destination
);

/// Write a header and opaque payload through the selected Metal codec.
void write_tile_bytes(
    const std::filesystem::path &path,
    MetalTileCompression compression,
    const void *header,
    size_t header_size,
    const void *payload,
    size_t payload_size
) {
  const std::filesystem::path temporary = path.string() + ".tmp";
  if (compression == MetalTileCompression::None) {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(header), static_cast<std::streamsize>(header_size));
    stream.write(
        reinterpret_cast<const char *>(payload),
        static_cast<std::streamsize>(payload_size)
    );
    if (!stream) {
      stream.close();
      std::filesystem::remove(temporary);
      throw std::runtime_error("Could not write raw Metal tile " + path.string());
    }
    stream.close();
  } else {
    errno = 0;
    MTLIOCompressionContext context = MTLIOCreateCompressionContext(
        temporary.c_str(),
        compression_method(compression),
        MTLIOCompressionContextDefaultChunkSize()
    );
    if (context == nullptr) {
      throw std::runtime_error(
          "Could not create Metal compression context for " + path.string() + ": " +
          std::strerror(errno)
      );
    }
    MTLIOCompressionContextAppendData(context, header, header_size);
    MTLIOCompressionContextAppendData(context, payload, payload_size);
    if (MTLIOFlushAndDestroyCompressionContext(context) != MTLIOCompressionStatusComplete) {
      std::filesystem::remove(temporary);
      throw std::runtime_error("Metal compression failed for " + path.string());
    }
  }
  publish_temporary_file(temporary, path);
}

/// Atomically replace the destination with a completely written temporary file.
void publish_temporary_file(
    const std::filesystem::path &temporary,
    const std::filesystem::path &destination
) {
  std::error_code error;
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error(
        "Could not publish Metal tile " + destination.string() + ": " + error.message()
    );
  }
}

} // namespace

bool is_metal_tile_path(const std::filesystem::path &path) {
  const std::string name = path.filename().string();
  return name.ends_with(".ptile") || name.ends_with(".ptile.zlib") ||
         name.ends_with(".ptile.lz4") || name.ends_with(".ptile.lzma") ||
         name.ends_with(".ptile.lzbitmap");
}

MetalTileCompression metal_tile_compression(const std::filesystem::path &path) {
  const std::string name = path.filename().string();
  if (name.ends_with(".ptile")) {
    return MetalTileCompression::None;
  }
  if (name.ends_with(".ptile.zlib")) {
    return MetalTileCompression::Zlib;
  }
  if (name.ends_with(".ptile.lz4")) {
    return MetalTileCompression::Lz4;
  }
  if (name.ends_with(".ptile.lzma")) {
    return MetalTileCompression::Lzma;
  }
  if (name.ends_with(".ptile.lzbitmap")) {
    return MetalTileCompression::LzBitmap;
  }
  throw std::invalid_argument("Not a recognised Metal tile path: " + path.string());
}

const char *metal_tile_suffix(MetalTileCompression compression) {
  switch (compression) {
  case MetalTileCompression::None:
    return ".ptile";
  case MetalTileCompression::Zlib:
    return ".ptile.zlib";
  case MetalTileCompression::Lz4:
    return ".ptile.lz4";
  case MetalTileCompression::Lzma:
    return ".ptile.lzma";
  case MetalTileCompression::LzBitmap:
    return ".ptile.lzbitmap";
  }
  throw std::invalid_argument("Unknown Metal tile compression method");
}

uint64_t metal_tile_mipmap_value_count(uint32_t cell_count) {
  if (!std::has_single_bit(cell_count)) {
    throw std::invalid_argument("Metal tile cell count must be a power of two");
  }
  uint64_t count = 0U;
  for (uint64_t side = cell_count; side != 0U; side /= 2U) {
    count += side * side;
  }
  return count;
}

QuantizedMetalTileRecordLayout quantized_metal_tile_record_layout(const MetalTileHeader &header) {
  if (header.sample_type != MetalTileSampleType::Uint16Decimeters) {
    throw std::invalid_argument("Quantized record layout requires uint16 terrain");
  }
  const uint64_t maximum_stride = std::numeric_limits<uint32_t>::max();
  if (header.vertex_offset > maximum_stride ||
      header.vertex_byte_count > maximum_stride - header.vertex_offset) {
    throw std::overflow_error("Metal tile logical record exceeds Metal uint indexing");
  }
  const uint64_t logical_size = header.vertex_offset + header.vertex_byte_count;
  const uint64_t stride = (logical_size + 3U) & ~uint64_t{3U};
  if (stride > maximum_stride) {
    throw std::overflow_error("Metal tile logical record exceeds Metal uint indexing");
  }
  return {
      static_cast<uint32_t>(logical_size),
      static_cast<uint32_t>(stride),
      static_cast<uint32_t>(header.vertex_offset),
      static_cast<uint32_t>(offsetof(MetalTileHeader, elevation_base_decimeters)),
  };
}

void validate_metal_tile_header(
    const MetalTileHeader &header,
    MetalTileCompression expected_compression
) {
  const bool float32 = header.magic == kMetalTileFloat32Magic &&
                       header.version == kMetalTileFloat32Version &&
                       header.header_size == kMetalTileFloat32HeaderSize &&
                       header.sample_type == MetalTileSampleType::Float32;
  const bool uint16 = header.magic == kMetalTileUint16Magic &&
                      header.version == kMetalTileUint16Version &&
                      header.header_size == kMetalTileUint16HeaderSize &&
                      header.sample_type == MetalTileSampleType::Uint16Decimeters;
  const bool lod = header.magic == kMetalTileLodMagic && header.version == kMetalTileLodVersion &&
                   header.header_size == kMetalTileLodHeaderSize &&
                   (header.sample_type == MetalTileSampleType::Float32 ||
                    header.sample_type == MetalTileSampleType::Uint16Decimeters);
  if (!float32 && !uint16 && !lod) {
    throw std::runtime_error("Metal tile has an unsupported header or version");
  }
  if (header.compression != expected_compression || header.epsg_code == 0U ||
      !std::has_single_bit(header.cell_count) || header.cell_count == 0U ||
      header.level_count != std::countr_zero(header.cell_count) + 1U ||
      !std::isfinite(header.maximum_elevation) || !std::isfinite(header.lower_left_x) ||
      !std::isfinite(header.lower_left_y) || !std::isfinite(header.cell_size) ||
      header.cell_size <= 0.0) {
    throw std::runtime_error("Metal tile header contains invalid terrain metadata");
  }

  const uint64_t vertex_side = static_cast<uint64_t>(header.cell_count) + 1U;
  const uint64_t sample_bytes =
      header.sample_type == MetalTileSampleType::Float32 ? sizeof(float) : sizeof(uint16_t);
  if (vertex_side > std::numeric_limits<uint64_t>::max() / vertex_side ||
      vertex_side * vertex_side > std::numeric_limits<uint64_t>::max() / sample_bytes) {
    throw std::runtime_error("Metal tile payload dimensions overflow its byte count");
  }
  const uint64_t vertex_bytes = vertex_side * vertex_side * sample_bytes;
  if ((!lod &&
       (header.vertex_offset != header.header_size || header.vertex_byte_count != vertex_bytes)) ||
      (lod && (header.lod_count == 0U || header.lod_count > header.level_count ||
               header.lod_entry_size != sizeof(MetalTileLod) ||
               header.lod_table_offset != header.header_size ||
               header.lod_table_byte_count !=
                   static_cast<uint64_t>(header.lod_count) * sizeof(MetalTileLod) ||
               header.vertex_offset < header.lod_table_offset + header.lod_table_byte_count ||
               header.vertex_byte_count != vertex_bytes))) {
    throw std::runtime_error("Metal tile payload layout does not match its dimensions");
  }

  if (header.sample_type == MetalTileSampleType::Uint16Decimeters) {
    const double minimum_elevation = static_cast<double>(header.elevation_base_decimeters) / 10.0;
    const double maximum_representable =
        static_cast<double>(header.elevation_base_decimeters) / 10.0 +
        static_cast<double>(std::numeric_limits<uint16_t>::max()) / 10.0;
    const double maximum = static_cast<double>(header.maximum_elevation);
    if (maximum < minimum_elevation - 0.1 || maximum > maximum_representable + 0.1) {
      throw std::runtime_error("Metal tile maximum elevation is outside its fixed-point range");
    }
  }
}

void write_metal_tile(
    const std::filesystem::path &path,
    const MetalTileHeader &header,
    std::span<const float> vertices
) {
  validate_metal_tile_header(header, header.compression);
  if (header.version != kMetalTileFloat32Version ||
      header.sample_type != MetalTileSampleType::Float32 ||
      vertices.size_bytes() != header.vertex_byte_count) {
    throw std::invalid_argument("Float32 Metal tile payload does not match its header");
  }
  const MetalTileFloat32Header disk = float32_disk_header(header);
  write_tile_bytes(
      path,
      header.compression,
      &disk,
      sizeof(disk),
      vertices.data(),
      vertices.size_bytes()
  );
}

std::vector<MetalTileLod>
read_metal_tile_lods(const std::filesystem::path &path, const MetalTileHeader &header) {
  validate_metal_tile_header(header, metal_tile_compression(path));
  if (header.version != kMetalTileLodVersion) {
    return {{1U,
             header.cell_count,
             header.level_count,
             header.elevation_base_decimeters,
             header.maximum_elevation,
             0U,
             header.vertex_offset,
             header.vertex_byte_count}};
  }
  std::vector<MetalTileLod> lods(header.lod_count);
  if (header.compression == MetalTileCompression::None) {
    std::ifstream stream(path, std::ios::binary);
    stream.seekg(static_cast<std::streamoff>(header.lod_table_offset));
    if (!stream.read(
            reinterpret_cast<char *>(lods.data()),
            static_cast<std::streamsize>(header.lod_table_byte_count)
        )) {
      throw std::runtime_error("Could not read Metal tile LOD table " + path.string());
    }
  } else {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("No Metal device is available to inspect " + path.string());
    }
    id<MTLIOCommandQueue> queue = make_metal_io_queue(device);
    id<MTLIOFileHandle> file = open_metal_file(device, path, header.compression);
    id<MTLIOCommandBuffer> command = [queue commandBuffer];
    [command loadBytes:lods.data()
                      size:static_cast<NSUInteger>(header.lod_table_byte_count)
              sourceHandle:file
        sourceHandleOffset:header.lod_table_offset];
    complete_io(command, path);
  }
  uint64_t expected_offset = header.lod_table_offset + header.lod_table_byte_count;
  for (uint32_t index = 0U; index < lods.size(); index++) {
    const MetalTileLod &lod = lods[index];
    const uint32_t expected_cell_count = header.cell_count >> index;
    const uint64_t side = static_cast<uint64_t>(expected_cell_count) + 1U;
    const uint64_t bytes =
        side * side *
        (header.sample_type == MetalTileSampleType::Float32 ? sizeof(float) : sizeof(uint16_t));
    if (lod.lod != index + 1U || lod.cell_count != expected_cell_count ||
        lod.level_count != std::countr_zero(expected_cell_count) + 1U ||
        !std::isfinite(lod.maximum_elevation) || lod.vertex_byte_count != bytes ||
        lod.vertex_offset != expected_offset) {
      throw std::runtime_error("Metal tile LOD table contains an invalid representation");
    }
    if (lod.vertex_byte_count > std::numeric_limits<uint64_t>::max() - expected_offset) {
      throw std::runtime_error("Metal tile LOD payload offsets overflow");
    }
    expected_offset += lod.vertex_byte_count;
  }
  const MetalTileLod &base = lods.front();
  if (base.vertex_offset != header.vertex_offset ||
      base.vertex_byte_count != header.vertex_byte_count ||
      base.maximum_elevation != header.maximum_elevation ||
      base.elevation_base_decimeters != header.elevation_base_decimeters) {
    throw std::runtime_error("Metal tile base LOD disagrees with its header");
  }
  return lods;
}

void write_metal_tile_lods(
    const std::filesystem::path &path,
    const MetalTileHeader &header,
    std::span<const MetalTileLod> lods,
    std::span<const std::byte> payload
) {
  validate_metal_tile_header(header, header.compression);
  if (header.version != kMetalTileLodVersion || lods.size() != header.lod_count) {
    throw std::invalid_argument("Metal LOD tile header does not match its LOD table");
  }
  uint64_t expected_payload_size = 0U;
  uint64_t expected_offset = header.lod_table_offset + header.lod_table_byte_count;
  for (const MetalTileLod &lod : lods) {
    if (lod.vertex_offset != expected_offset ||
        lod.vertex_byte_count > std::numeric_limits<uint64_t>::max() - expected_payload_size) {
      throw std::invalid_argument("Metal LOD tile payload layout is invalid");
    }
    expected_payload_size += lod.vertex_byte_count;
    expected_offset += lod.vertex_byte_count;
  }
  if (payload.size_bytes() != expected_payload_size) {
    throw std::invalid_argument("Metal LOD tile payload size does not match its table");
  }
  std::vector<std::byte> stream_payload(header.lod_table_byte_count + payload.size_bytes());
  std::memcpy(stream_payload.data(), lods.data(), static_cast<size_t>(header.lod_table_byte_count));
  std::memcpy(
      stream_payload.data() + header.lod_table_byte_count,
      payload.data(),
      payload.size_bytes()
  );
  write_tile_bytes(
      path,
      header.compression,
      &header,
      sizeof(header),
      stream_payload.data(),
      stream_payload.size()
  );
}

void write_metal_tile(
    const std::filesystem::path &path,
    const MetalTileHeader &header,
    std::span<const uint16_t> vertices
) {
  validate_metal_tile_header(header, header.compression);
  if (header.version != kMetalTileUint16Version ||
      header.sample_type != MetalTileSampleType::Uint16Decimeters ||
      vertices.size_bytes() != header.vertex_byte_count) {
    throw std::invalid_argument("Uint16 Metal tile payload does not match its header");
  }
  const MetalTileUint16Header disk = uint16_disk_header(header);
  write_tile_bytes(
      path,
      header.compression,
      &disk,
      sizeof(disk),
      vertices.data(),
      vertices.size_bytes()
  );
}

id<MTLIOCommandQueue> make_metal_io_queue(id<MTLDevice> device) {
  MTLIOCommandQueueDescriptor *descriptor = [[MTLIOCommandQueueDescriptor alloc] init];
  descriptor.type = MTLIOCommandQueueTypeConcurrent;
  descriptor.priority = MTLIOPriorityNormal;
  descriptor.maxCommandBufferCount = kMetalIoConcurrency;
  descriptor.maxCommandsInFlight = kMetalIoConcurrency;
  NSError *error = nil;
  id<MTLIOCommandQueue> queue = [device newIOCommandQueueWithDescriptor:descriptor error:&error];
  if (queue == nil) {
    const char *detail = error == nil ? "unknown error" : error.localizedDescription.UTF8String;
    throw std::runtime_error(std::string("Could not create Metal I/O queue: ") + detail);
  }
  return queue;
}

id<MTLIOFileHandle> open_metal_tile_file(id<MTLDevice> device, const std::filesystem::path &path) {
  return open_metal_file(device, path, metal_tile_compression(path));
}

NSUInteger metal_tile_io_concurrency() { return kMetalIoConcurrency; }

MetalTileHeader read_metal_tile_header(const std::filesystem::path &path) {
  const MetalTileCompression compression = metal_tile_compression(path);
  if (compression == MetalTileCompression::None) {
    const MetalTileHeader header = read_raw_header(path);
    validate_metal_tile_header(header, compression);
    return header;
  }

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    throw std::runtime_error("No Metal device is available to inspect " + path.string());
  }
  id<MTLIOCommandQueue> queue = make_metal_io_queue(device);
  id<MTLIOFileHandle> handle = open_metal_file(device, path, compression);
  MetalTileHeaderBytes header_bytes = {};
  id<MTLIOCommandBuffer> command = [queue commandBuffer];
  [command loadBytes:header_bytes.data()
                    size:header_bytes.size()
            sourceHandle:handle
      sourceHandleOffset:0U];
  complete_io(command, path);
  const MetalTileHeader header = decode_header(header_bytes);
  validate_metal_tile_header(header, compression);
  return header;
}

void load_metal_tiles_into_buffer(
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue,
    std::span<const MetalTileBufferLoad> loads,
    id<MTLBuffer> vertex_buffer,
    NSUInteger vertex_buffer_length
) {
  for (const MetalTileBufferLoad &load : loads) {
    if (load.byte_count > static_cast<uint64_t>(std::numeric_limits<NSUInteger>::max())) {
      throw std::overflow_error("Metal tile vertex payload exceeds NSUInteger range");
    }
    const NSUInteger load_size = static_cast<NSUInteger>(load.byte_count);
    if (load.destination_offset > vertex_buffer_length ||
        load_size > vertex_buffer_length - load.destination_offset) {
      throw std::out_of_range("Metal tile load exceeds the vertex atlas buffer");
    }
  }

  // Separate command buffers allow a concurrent queue to decompress several
  // independent streams in parallel. Retain their file handles until the
  // whole wave completes, then reuse the queue for the next bounded wave.
  for (size_t wave_start = 0U; wave_start < loads.size(); wave_start += kMetalIoConcurrency) {
    const size_t wave_size =
        std::min(static_cast<size_t>(kMetalIoConcurrency), loads.size() - wave_start);
    std::vector<id<MTLIOFileHandle>> files;
    std::vector<id<MTLIOCommandBuffer>> commands;
    files.reserve(wave_size);
    commands.reserve(wave_size);
    for (size_t index = 0U; index < wave_size; index++) {
      const MetalTileBufferLoad &load = loads[wave_start + index];
      id<MTLIOFileHandle> file =
          load.file == nil ? open_metal_tile_file(device, load.path) : load.file;
      id<MTLIOCommandBuffer> command = [queue commandBuffer];
      if (command == nil) {
        throw std::runtime_error("Could not create a Metal tile I/O command");
      }
      [command loadBuffer:vertex_buffer
                      offset:load.destination_offset
                        size:static_cast<NSUInteger>(load.byte_count)
                sourceHandle:file
          sourceHandleOffset:load.source_offset];
      [command commit];
      files.push_back(file);
      commands.push_back(command);
    }
    for (size_t index = 0U; index < wave_size; index++) {
      id<MTLIOCommandBuffer> command = commands[index];
      [command waitUntilCompleted];
      if (command.status != MTLIOStatusComplete) {
        const char *detail =
            command.error == nil ? "unknown error" : command.error.localizedDescription.UTF8String;
        throw std::runtime_error(
            "Could not load Metal tile " + loads[wave_start + index].path.string() + ": " + detail
        );
      }
    }
  }
}

} // namespace panorama
