#include "gdal_utils.h"
#include "metal_tile_writer.h"
#include "terrain_manifest.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gdal_priv.h>
#include <map>
#include <ogr_spatialref.h>
#include <stdexcept>
#include <vector>
using namespace panorama;
using namespace panorama::terrain;
namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void write_bytes(const std::filesystem::path &path, const std::vector<char> &bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Exercise the actual generator, including upgrading skipped existing tiles.
void check_generator(const std::filesystem::path &executable, const std::filesystem::path &root) {
  GDALAllRegister();
  const auto input = root / "input";
  const auto output = root / "generated";
  std::filesystem::create_directory(input);
  OGRSpatialReference crs;
  require(crs.importFromEPSG(2056) == OGRERR_NONE, "Fixture CRS");
  char *wkt = nullptr;
  require(crs.exportToWkt(&wkt) == OGRERR_NONE, "Fixture WKT");
  SourceGrid source{};
  source.projection_wkt = wkt;
  CPLFree(wkt);
  const TerrainChunk chunk{3,
                           {-42.35F, 5, 100, 40, 200.24F, 0, -3, 25, 80},
                           std::vector<uint8_t>(9, 1),
                           {}};
  // GeoTIFF remains an input format. Create the source fixture directly with
  // GDAL; the application no longer contains a GeoTIFF output writer.
  {
    auto *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "Fixture GeoTIFF driver");
    GdalDatasetPointer dataset(
        driver->Create((input / "source.tif").c_str(), 3, 3, 1, GDT_Float32, nullptr)
    );
    require(dataset != nullptr, "Fixture GeoTIFF creation");
    double transform[] = {2600000, 1, 0, 1200000, 0, -1};
    require(
        dataset->SetGeoTransform(transform) == CE_None &&
            dataset->SetProjection(source.projection_wkt.c_str()) == CE_None,
        "Fixture GeoTIFF georeferencing"
    );
    auto *band = dataset->GetRasterBand(1);
    require(
        band->SetNoDataValue(-9999) == CE_None && band->RasterIO(
                                                      GF_Write,
                                                      0,
                                                      0,
                                                      3,
                                                      3,
                                                      const_cast<float *>(chunk.elevations.data()),
                                                      3,
                                                      3,
                                                      GDT_Float32,
                                                      0,
                                                      0,
                                                      nullptr
                                                  ) == CE_None,
        "Fixture GeoTIFF samples"
    );
  }
  const auto run = [&](NSArray<NSString *> *extra = @[], bool default_output = false) {
    NSTask *task = [[NSTask alloc] init];
    task.executableURL = [NSURL fileURLWithPath:@(executable.c_str())];
    task.currentDirectoryURL = [NSURL fileURLWithPath:@(root.c_str())];
    NSMutableArray<NSString *> *arguments = [@[
      @"--input",
      @(input.c_str()),
      @"--power",
      @"1",
      @"--lod",
      @"point",
      @"--compression",
      @"none"
    ] mutableCopy];
    if (!default_output)
      [arguments addObjectsFromArray:@[ @"--output", @(output.c_str()) ]];
    [arguments addObjectsFromArray:extra];
    task.arguments = arguments;
    NSError *error = nil;
    if (![task launchAndReturnError:&error])
      throw std::runtime_error(
          "Could not launch tile generator: " + std::string(error.localizedDescription.UTF8String)
      );
    [task waitUntilExit];
    return task.terminationStatus;
  };
  require(run() == 0, "Default uint16 tile generation failed");
  const auto path = terrain_manifest_path(output);
  const auto original = read_terrain_manifest(path);
  require(!original.empty(), "Generator omitted manifest entries");
  std::map<std::filesystem::path, std::filesystem::file_time_type> timestamps;
  for (const auto &file : std::filesystem::directory_iterator(output)) {
    if (file.path().extension() == ".ptile") {
      const auto header = read_metal_tile_header(file.path());
      require(
          header.sample_type == MetalTileSampleType::Uint16Decimeters &&
              header.vertex_byte_count == 9U * sizeof(uint16_t),
          "Generator did not default to uint16 payloads"
      );
      timestamps.emplace(file.path(), file.last_write_time());
    }
  }
  require(timestamps.size() == original.size(), "Generator omitted a tile's bounds");
  auto old_entries = original;
  for (auto &entry : old_entries)
    entry.coverage.reset();
  write_terrain_manifest(path, old_entries);
  std::ifstream stream(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
  const uint32_t one = 1, zero = 0;
  std::memcpy(bytes.data() + 8, &one, sizeof(one));
  for (size_t offset = 44; offset < bytes.size(); offset += 24)
    std::memcpy(bytes.data() + offset, &zero, sizeof(zero));
  write_bytes(path, bytes);
  // First run scans payloads to upgrade v1; the next reuses complete v2 entries.
  for (int repetition = 0; repetition < 2; ++repetition) {
    require(run() == 0, "Uint16 tile generation failed");
    const auto upgraded = read_terrain_manifest(path);
    require(upgraded.size() == original.size(), "Upgrade omitted manifest entries");
    for (size_t i = 0; i < original.size(); ++i)
      require(
          upgraded[i].row == original[i].row && upgraded[i].column == original[i].column &&
              upgraded[i].minimum_elevation.has_value() &&
              std::abs(*upgraded[i].minimum_elevation - *original[i].minimum_elevation) < 0.001F &&
              std::abs(upgraded[i].maximum_elevation - original[i].maximum_elevation) < 0.001F &&
              upgraded[i].coverage == original[i].coverage,
          "Generator upgrade changed tile elevation bounds"
      );
    for (const auto &[tile, timestamp] : timestamps)
      require(
          std::filesystem::last_write_time(tile) == timestamp,
          "Manifest upgrade rewrote an existing tile"
      );
  }
  require(run(@[ @"--format", @"metal" ]) != 0, "Generator accepted removed --format option");
  require(run(@[ @"--format", @"geotiff" ]) != 0, "Generator accepted removed GeoTIFF output");
  require(
      run(@[ @"--sample-type", @"uint16" ]) != 0,
      "Generator accepted removed --sample-type option"
  );
  require(run(@[ @"--sample-type", @"float32" ]) != 0, "Generator accepted removed Float32 output");
  require(run(@[], true) == 0, "Default Metal output directory generation failed");
  const auto default_output = root / "data/input-1-level-0-metal-u16-none-lod-point";
  require(
      read_terrain_manifest(terrain_manifest_path(default_output)).size() == original.size(),
      "Default output directory omitted Metal tile metadata"
  );
  const auto default_tile =
      default_output / ("input_level-0_p1_r" + std::to_string(original.front().row) + "_c" +
                        std::to_string(original.front().column) + ".ptile");
  require(
      read_metal_tile_header(default_tile).sample_type == MetalTileSampleType::Uint16Decimeters,
      "Default output is not a uint16 Metal tile"
  );
}

void check_uint16_range(
    const std::filesystem::path &root,
    id<MTLDevice> device,
    id<MTLIOCommandQueue> queue
) {
  // Version 5 reserves code zero while retaining real zero/negative heights.
  const TerrainChunk chunk{3,
                           {-400, 0, 6153.4F, 0, 10, 20, 30, 40, 50},
                           std::vector<uint8_t>(9, 1),
                           {}};
  const DestinationGrid grid{2600000, 1200000, 1, 2, RasterLayout::Level0, 0};
  SourceGrid source{};
  source.epsg_code = 2056U;
  for (const auto compression : {MetalTileCompression::None, MetalTileCompression::Lz4}) {
    const auto path = root / (std::string("uint16-range") + metal_tile_suffix(compression));
    (void)write_metal_tile_chunk(path, chunk, grid, {0, 0}, source, compression);
    const auto header = read_metal_tile_header(path);
    require(
        header.elevation_base_decimeters == -4001 && header.vertex_byte_count == 18,
        "Uint16 file encoding changed"
    );
    id<MTLBuffer> data = [device newBufferWithLength:header.vertex_byte_count
                                             options:MTLResourceStorageModeShared];
    const MetalTileBufferLoad load{path, 0U, nil, header.vertex_offset, header.vertex_byte_count};
    load_metal_tiles_into_buffer(device, queue, std::span(&load, 1), data, data.length);
    const auto *encoded = static_cast<const uint16_t *>(data.contents);
    for (uint32_t y = 0; y < 3; ++y)
      for (uint32_t x = 0; x < 3; ++x) {
        const float height = float(header.elevation_base_decimeters + encoded[y * 3 + x]) / 10.0F;
        require(
            height == chunk.elevations[(2U - y) * 3U + x],
            "Uint16 generation changed elevation values or row order"
        );
      }
    require(
        encoded[6] == 1U && encoded[8] == 65535U,
        "Uint16 range endpoints are no longer valid elevations"
    );
  }

  TerrainChunk partial = chunk;
  partial.covered[0] = 0;
  partial.elevations[0] = -99999; // Must not pollute bounds or the quantization range.
  for (const auto compression : {MetalTileCompression::None, MetalTileCompression::Lz4}) {
    const auto path = root / (std::string("partial") + metal_tile_suffix(compression));
    const auto bounds = write_metal_tile_chunk(path, partial, grid, {0, 0}, source, compression);
    const auto header = read_metal_tile_header(path);
    const auto coverage = read_metal_tile_coverage(path, header);
    require(
        coverage && !coverage->full() && coverage->covered_area(0, 0, 2, 2) == 3,
        "Coverage did not exclude every cell touching the missing corner"
    );
    require(
        !coverage->contains(0.5, 1.5) && coverage->contains(1.5, 1.5),
        "Coverage row order changed"
    );
    require(
        std::abs(bounds.minimum) < 0.01F && std::abs(bounds.maximum - 6153.4F) < 0.01F,
        "Missing height polluted bounds"
    );
    const auto scanned = read_metal_tile_elevation_range(path, device, queue);
    require(
        std::abs(scanned.minimum - bounds.minimum) < 0.001F &&
            std::abs(scanned.maximum - bounds.maximum) < 0.001F && scanned.coverage == coverage,
        "Compressed coverage or bounds did not round-trip"
    );
    const std::array<TerrainManifestEntry, 1> entries = {
        {{0, 0, bounds.maximum, bounds.minimum, coverage}}};
    const auto manifest = root / "partial-manifest.bin";
    write_terrain_manifest(manifest, entries);
    require(read_terrain_manifest(manifest).front().coverage == coverage, "Manifest lost coverage");
    id<MTLBuffer> data = [device newBufferWithLength:header.vertex_byte_count
                                             options:MTLResourceStorageModeShared];
    const MetalTileBufferLoad load{path, 0, nil, header.vertex_offset, header.vertex_byte_count};
    load_metal_tiles_into_buffer(device, queue, std::span(&load, 1), data, data.length);
    const auto *encoded = static_cast<const uint16_t *>(data.contents);
    require(
        encoded[6] == 0 && encoded[7] != 0 && encoded[5] != 0,
        "Missing sample confused with valid zero"
    );
  }

  // Construct a complete legacy Float32 file independently of the uint16-only
  // writer, so the loader must reject its encoding rather than malformed bytes.
  auto header = read_metal_tile_header(root / "uint16-range.ptile");
  auto lod = read_metal_tile_lods(root / "uint16-range.ptile", header).front();
  header.sample_type = static_cast<MetalTileSampleType>(1U);
  header.elevation_base_decimeters = 0;
  header.vertex_byte_count = chunk.elevations.size() * sizeof(float);
  lod.elevation_base_decimeters = 0;
  lod.vertex_byte_count = header.vertex_byte_count;
  const auto legacy = root / "unsupported-float.ptile";
  {
    std::ofstream out(legacy, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    out.write(reinterpret_cast<const char *>(&lod), sizeof(lod));
    out.write(
        reinterpret_cast<const char *>(chunk.elevations.data()),
        static_cast<std::streamsize>(header.vertex_byte_count)
    );
    require(bool(out), "Could not write legacy Float32 fixture");
  }
  bool rejected = false;
  try {
    (void)read_metal_tile_header(legacy);
  } catch (const std::runtime_error &error) {
    rejected = std::string(error.what()).find("uint16") != std::string::npos;
  }
  require(rejected, "Loader did not reject legacy Float32 tiles with an encoding error");
}
} // namespace
int main(int argc, const char *argv[]) {
  try {
    @autoreleasepool {
      char temporary[] = "/tmp/panorama-manifest-test.XXXXXX";
      const char *created = mkdtemp(temporary);
      require(created != nullptr, "Temporary directory");
      const std::filesystem::path root(created);
      std::printf("Manifest fixtures: %s\n", created);
      const auto manifest = terrain_manifest_path(root);
      const std::array entries{TerrainManifestEntry{-3, 5, 123.5F, -44.25F}};
      write_terrain_manifest(manifest, entries);
      const auto decoded = read_terrain_manifest(manifest);
      require(
          decoded.size() == 1 && decoded[0].minimum_elevation == -44.25F &&
              decoded[0].maximum_elevation == 123.5F,
          "Version 2 manifest bounds changed"
      );
      std::ifstream input(manifest, std::ios::binary);
      std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
      require(bytes.size() == 48, "Unexpected manifest layout");
      const uint32_t one = 1, zero = 0;
      std::memcpy(bytes.data() + 8, &one, sizeof(one));
      std::memcpy(bytes.data() + 44, &zero, sizeof(zero));
      write_bytes(root / "v1.bin", bytes);
      const auto legacy = read_terrain_manifest(root / "v1.bin");
      require(
          !legacy[0].minimum_elevation.has_value() && legacy[0].maximum_elevation == 123.5F,
          "Version 1 compatibility"
      );
      const uint32_t two = 2;
      const float invalid_min = 200;
      std::memcpy(bytes.data() + 8, &two, sizeof(two));
      std::memcpy(bytes.data() + 44, &invalid_min, sizeof(invalid_min));
      write_bytes(root / "invalid.bin", bytes);
      bool rejected = false;
      try {
        (void)read_terrain_manifest(root / "invalid.bin");
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "Inverted manifest bounds accepted");
      bytes.pop_back();
      write_bytes(root / "truncated.bin", bytes);
      rejected = false;
      try {
        (void)read_terrain_manifest(root / "truncated.bin");
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "Truncated manifest accepted");

      const TerrainChunk chunk{3,
                               {-42.35F, 5, 100, 40, 200.24F, 0, -3, 25, 80},
                               std::vector<uint8_t>(9, 1),
                               {{2, {-55.17F, 300.26F, 20, 1}, std::vector<uint8_t>(4, 1)}}};
      const DestinationGrid grid{2600000, 1200000, 1, 2, RasterLayout::Level0, 0};
      SourceGrid source{};
      source.epsg_code = 2056U;
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      auto queue = make_metal_io_queue(device);
      check_uint16_range(root, device, queue);
      for (const auto compression : {MetalTileCompression::None, MetalTileCompression::Lz4}) {
        @autoreleasepool {
          const auto path = root / (std::string("uint16-lods") + metal_tile_suffix(compression));
          const auto written =
              write_metal_tile_chunk(path, chunk, grid, {0, 0}, source, compression);
          const auto scanned = read_metal_tile_elevation_range(path, device, queue);
          require(
              written.minimum <= scanned.minimum + 0.001F &&
                  written.maximum >= scanned.maximum - 0.001F,
              "Stored terrain escaped generated manifest range"
          );
          require(
              written.minimum < -55 && written.maximum > 300,
              "Manifest omitted coarser LOD extrema"
          );
          require(
              std::abs(written.minimum - scanned.minimum) < 0.001F &&
                  std::abs(written.maximum - scanned.maximum) < 0.001F,
              "Skipped-file upgrade changed manifest bounds"
          );
        }
      }
      if (argc == 2)
        check_generator(std::filesystem::absolute(argv[1]), root);
    }
    std::puts("Terrain manifest tests passed.");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
