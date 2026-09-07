#include "geotiff_writer.h"
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
  const DestinationGrid grid{2600000, 1200000, 1, 2, RasterLayout::Level0, -9999};
  write_geotiff_chunk(input / "source.tif", chunk, grid, {0, 0}, source);
  const auto run = [&] {
    NSTask *task = [[NSTask alloc] init];
    task.executableURL = [NSURL fileURLWithPath:@(executable.c_str())];
    task.arguments = @[
      @"--input",
      @(input.c_str()),
      @"--output",
      @(output.c_str()),
      @"--format",
      @"metal",
      @"--power",
      @"1",
      @"--lod",
      @"point",
      @"--sample-type",
      @"uint16",
      @"--compression",
      @"none"
    ];
    NSError *error = nil;
    if (![task launchAndReturnError:&error])
      throw std::runtime_error(
          "Could not launch tile generator: " + std::string(error.localizedDescription.UTF8String)
      );
    [task waitUntilExit];
    require(task.terminationStatus == 0, "Tile generator failed");
  };
  run();
  const auto path = terrain_manifest_path(output);
  const auto original = read_terrain_manifest(path);
  require(!original.empty(), "Generator omitted manifest entries");
  std::map<std::filesystem::path, std::filesystem::file_time_type> timestamps;
  for (const auto &file : std::filesystem::directory_iterator(output)) {
    if (file.path().extension() == ".ptile")
      timestamps.emplace(file.path(), file.last_write_time());
  }
  require(timestamps.size() == original.size(), "Generator omitted a tile's bounds");
  std::ifstream stream(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
  const uint32_t one = 1, zero = 0;
  std::memcpy(bytes.data() + 8, &one, sizeof(one));
  for (size_t offset = 44; offset < bytes.size(); offset += 24)
    std::memcpy(bytes.data() + offset, &zero, sizeof(zero));
  write_bytes(path, bytes);
  // First run scans payloads to upgrade v1; the next reuses complete v2 entries.
  for (int repetition = 0; repetition < 2; ++repetition) {
    run();
    const auto upgraded = read_terrain_manifest(path);
    require(upgraded.size() == original.size(), "Upgrade omitted manifest entries");
    for (size_t i = 0; i < original.size(); ++i)
      require(
          upgraded[i].row == original[i].row && upgraded[i].column == original[i].column &&
              upgraded[i].minimum_elevation.has_value() &&
              std::abs(*upgraded[i].minimum_elevation - *original[i].minimum_elevation) < 0.001F &&
              std::abs(upgraded[i].maximum_elevation - original[i].maximum_elevation) < 0.001F,
          "Generator upgrade changed tile elevation bounds"
      );
    for (const auto &[tile, timestamp] : timestamps)
      require(
          std::filesystem::last_write_time(tile) == timestamp,
          "Manifest upgrade rewrote an existing tile"
      );
  }
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
      for (const auto compression : {MetalTileCompression::None, MetalTileCompression::Lz4}) {
        for (const auto sample :
             {MetalTileSampleType::Float32, MetalTileSampleType::Uint16Decimeters}) {
          @autoreleasepool {
            const auto path =
                root / (std::to_string(uint32_t(sample)) + metal_tile_suffix(compression));
            const auto written =
                write_metal_tile_chunk(path, chunk, grid, {0, 0}, source, compression, sample);
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
