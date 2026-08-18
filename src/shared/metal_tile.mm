#include "metal_tile.h"

#import <Foundation/Foundation.h>
#import <Metal/MTLIOCompressor.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
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
static_assert(sizeof(MetalTileHeader) == 96U);

// Keep one I/O wave within the queue's explicit command-buffer and command
// concurrency limits. Later waves reuse the same queue after the previous
// commands have completed.
constexpr NSUInteger kMetalIoConcurrency = 8U;

/// Convert the stable on-disk enum to Metal's compression API value.
[[nodiscard]] MTLIOCompressionMethod compression_method(MetalTileCompression compression) {
  switch (compression) {
  case MetalTileCompression::Zlib:
    return MTLIOCompressionMethodZlib;
  case MetalTileCompression::Lzfse:
    return MTLIOCompressionMethodLZFSE;
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

/// Read a raw header without requiring a Metal device.
[[nodiscard]] MetalTileHeader read_raw_header(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  MetalTileHeader header = {};
  if (!stream.read(reinterpret_cast<char *>(&header), sizeof(header))) {
    throw std::runtime_error("Could not read Metal tile header " + path.string());
  }
  return header;
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
         name.ends_with(".ptile.lzfse") || name.ends_with(".ptile.lz4") ||
         name.ends_with(".ptile.lzma") || name.ends_with(".ptile.lzbitmap");
}

MetalTileCompression metal_tile_compression(const std::filesystem::path &path) {
  const std::string name = path.filename().string();
  if (name.ends_with(".ptile")) {
    return MetalTileCompression::None;
  }
  if (name.ends_with(".ptile.zlib")) {
    return MetalTileCompression::Zlib;
  }
  if (name.ends_with(".ptile.lzfse")) {
    return MetalTileCompression::Lzfse;
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
  case MetalTileCompression::Lzfse:
    return ".ptile.lzfse";
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

void validate_metal_tile_header(
    const MetalTileHeader &header,
    MetalTileCompression expected_compression
) {
  if (header.magic != kMetalTileMagic || header.version != kMetalTileVersion ||
      header.header_size != static_cast<uint32_t>(sizeof(MetalTileHeader))) {
    throw std::runtime_error("Metal tile has an unsupported header or version");
  }
  if (header.compression != expected_compression ||
      header.sample_type != MetalTileSampleType::Float32 || header.epsg_code == 0U ||
      !std::has_single_bit(header.cell_count) || header.cell_count == 0U ||
      header.level_count != std::countr_zero(header.cell_count) + 1U ||
      !std::isfinite(header.maximum_elevation) || !std::isfinite(header.lower_left_x) ||
      !std::isfinite(header.lower_left_y) || !std::isfinite(header.cell_size) ||
      header.cell_size <= 0.0) {
    throw std::runtime_error("Metal tile header contains invalid terrain metadata");
  }

  const uint64_t vertex_side = static_cast<uint64_t>(header.cell_count) + 1U;
  const uint64_t vertex_bytes = vertex_side * vertex_side * sizeof(float);
  if (header.vertex_offset != sizeof(MetalTileHeader) || header.vertex_byte_count != vertex_bytes) {
    throw std::runtime_error("Metal tile payload layout does not match its dimensions");
  }
}

void write_metal_tile(
    const std::filesystem::path &path,
    const MetalTileHeader &header,
    std::span<const float> vertices
) {
  validate_metal_tile_header(header, header.compression);
  if (vertices.size_bytes() != header.vertex_byte_count) {
    throw std::invalid_argument("Metal tile payload does not match its header");
  }

  const std::filesystem::path temporary = path.string() + ".tmp";
  if (header.compression == MetalTileCompression::None) {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(
        reinterpret_cast<const char *>(&header),
        static_cast<std::streamsize>(sizeof(header))
    );
    stream.write(
        reinterpret_cast<const char *>(vertices.data()),
        static_cast<std::streamsize>(vertices.size_bytes())
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
        compression_method(header.compression),
        MTLIOCompressionContextDefaultChunkSize()
    );
    if (context == nullptr) {
      throw std::runtime_error(
          "Could not create Metal compression context for " + path.string() + ": " +
          std::strerror(errno)
      );
    }
    MTLIOCompressionContextAppendData(context, &header, sizeof(header));
    MTLIOCompressionContextAppendData(context, vertices.data(), vertices.size_bytes());
    if (MTLIOFlushAndDestroyCompressionContext(context) != MTLIOCompressionStatusComplete) {
      std::filesystem::remove(temporary);
      throw std::runtime_error("Metal compression failed for " + path.string());
    }
  }
  publish_temporary_file(temporary, path);
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

MetalTileData read_metal_tile(const std::filesystem::path &path) {
  const MetalTileCompression compression = metal_tile_compression(path);
  if (compression == MetalTileCompression::None) {
    const MetalTileHeader header = read_raw_header(path);
    validate_metal_tile_header(header, compression);
    MetalTileData data = {
        header,
        std::vector<float>(header.vertex_byte_count / sizeof(float)),
    };
    std::ifstream stream(path, std::ios::binary);
    stream.seekg(static_cast<std::streamoff>(header.vertex_offset));
    stream.read(reinterpret_cast<char *>(data.vertices.data()), header.vertex_byte_count);
    if (!stream) {
      throw std::runtime_error("Could not read raw Metal tile payload " + path.string());
    }
    return data;
  }

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    throw std::runtime_error("No Metal device is available to decompress " + path.string());
  }
  id<MTLIOCommandQueue> queue = make_metal_io_queue(device);
  id<MTLIOFileHandle> handle = open_metal_file(device, path, compression);
  MetalTileHeader header = {};
  id<MTLIOCommandBuffer> header_command = [queue commandBuffer];
  [header_command loadBytes:&header size:sizeof(header) sourceHandle:handle sourceHandleOffset:0U];
  complete_io(header_command, path);
  validate_metal_tile_header(header, compression);

  MetalTileData data = {
      header,
      std::vector<float>(header.vertex_byte_count / sizeof(float)),
  };
  id<MTLIOCommandBuffer> payload_command = [queue commandBuffer];
  [payload_command loadBytes:data.vertices.data()
                        size:header.vertex_byte_count
                sourceHandle:handle
          sourceHandleOffset:header.vertex_offset];
  complete_io(payload_command, path);
  return data;
}

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
  MetalTileHeader header = {};
  id<MTLIOCommandBuffer> command = [queue commandBuffer];
  [command loadBytes:&header size:sizeof(header) sourceHandle:handle sourceHandleOffset:0U];
  complete_io(command, path);
  validate_metal_tile_header(header, compression);
  return header;
}

std::vector<MetalTileIoSubmission> submit_metal_tiles_into_buffer(
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue,
    std::span<const MetalTileBufferLoad> loads,
    uint64_t vertex_offset,
    uint64_t vertex_byte_count,
    id<MTLBuffer> vertex_buffer,
    NSUInteger vertex_buffer_length
) {
  if (loads.empty()) {
    return {};
  }
  if (loads.size() > kMetalIoConcurrency) {
    throw std::invalid_argument("Metal tile submission exceeds the I/O queue concurrency");
  }
  if (vertex_byte_count > static_cast<uint64_t>(std::numeric_limits<NSUInteger>::max())) {
    throw std::overflow_error("Metal tile vertex payload exceeds NSUInteger range");
  }
  const NSUInteger load_size = static_cast<NSUInteger>(vertex_byte_count);
  for (const MetalTileBufferLoad &load : loads) {
    if (load.destination_offset > vertex_buffer_length ||
        load_size > vertex_buffer_length - load.destination_offset) {
      throw std::out_of_range("Metal tile load exceeds the vertex atlas buffer");
    }
  }

  // Separate command buffers allow a concurrent queue to decompress several
  // independent compressed streams in parallel. Each command gets its own
  // event because a later concurrent command must not satisfy an earlier
  // load's dependency by signaling a larger value on one shared event.
  std::vector<MetalTileIoSubmission> submissions;
  submissions.reserve(loads.size());
  for (const MetalTileBufferLoad &load : loads) {
    id<MTLIOFileHandle> file =
        load.file == nil ? open_metal_tile_file(device, load.path) : load.file;
    id<MTLIOCommandBuffer> command = [queue commandBuffer];
    id<MTLSharedEvent> event = [device newSharedEvent];
    if (command == nil || event == nil) {
      throw std::runtime_error("Could not create an asynchronous Metal tile I/O command");
    }
    [command loadBuffer:vertex_buffer
                    offset:load.destination_offset
                      size:load_size
              sourceHandle:file
        sourceHandleOffset:vertex_offset];
    [command signalEvent:event value:1U];
    [command commit];
    submissions.push_back({{load.path}, {file}, command, event});
  }
  return submissions;
}

void load_metal_tiles_into_buffer(
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue,
    std::span<const MetalTileBufferLoad> loads,
    uint64_t vertex_offset,
    uint64_t vertex_byte_count,
    id<MTLBuffer> vertex_buffer,
    NSUInteger vertex_buffer_length
) {
  // Preserve the synchronous diagnostic/origin API by submitting bounded
  // waves through the asynchronous primitive and waiting only in this wrapper.
  for (size_t wave_start = 0U; wave_start < loads.size(); wave_start += kMetalIoConcurrency) {
    const size_t wave_size =
        std::min(static_cast<size_t>(kMetalIoConcurrency), loads.size() - wave_start);
    const std::vector<MetalTileIoSubmission> submissions = submit_metal_tiles_into_buffer(
        device,
        queue,
        loads.subspan(wave_start, wave_size),
        vertex_offset,
        vertex_byte_count,
        vertex_buffer,
        vertex_buffer_length
    );
    for (const MetalTileIoSubmission &submission : submissions) {
      [submission.command waitUntilCompleted];
      if (submission.command.status != MTLIOStatusComplete) {
        const char *detail = submission.command.error == nil
                                 ? "unknown error"
                                 : submission.command.error.localizedDescription.UTF8String;
        throw std::runtime_error(
            "Could not load Metal tile batch beginning with " + submission.paths.front().string() +
            ": " + detail
        );
      }
    }
  }
}

} // namespace panorama
