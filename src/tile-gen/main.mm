#include "geotiff_writer.h"
#include "metal_tile_writer.h"
#include "rechunker.h"
#include "source_catalogue.h"
#include "terrain_manifest.h"
#include "terrain_types.h"

#include "arguments.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace panorama::terrain {
namespace {

/// File representation selected for prepared terrain chunks.
enum class OutputFormat {
  GeoTiff,
  MetalTile,
};

/// Fully parsed command-line configuration for one terrain-preparation run.
struct Options {
  std::filesystem::path input_directory;
  std::filesystem::path output_directory;
  std::string dataset_name;
  uint32_t tile_cell_count = 1024U;
  uint32_t requested_power = 10U;
  bool explicit_tile_cell_count = false;
  bool explicit_power = false;
  RasterLayout layout = RasterLayout::Level0;
  double origin_x = 0.0;
  double origin_y = 0.0;
  double resolution = 0.0;
  float no_data = 0.0F;
  uint32_t max_tiles = 0U;
  bool overwrite = false;
  bool dry_run = false;
  OutputFormat format = OutputFormat::GeoTiff;
  MetalTileCompression compression = MetalTileCompression::Lz4;
  MetalTileSampleType sample_type = MetalTileSampleType::Float32;
  LodSampling lod_sampling = LodSampling::None;
  bool explicit_compression = false;
  bool explicit_sample_type = false;
};

/// Print the stable command-line contract without constructing any GDAL state.
void print_usage(const char *program) {
  std::printf(
      "usage: %s --input DIR [options]\n"
      "\n"
      "Prepare DTM rasters for GPU terrain tracing. The input directory is scanned\n"
      "recursively for GeoTIFF, raw SRTM HGT, Arc/Info ASC, and ZIP-contained ASC\n"
      "files, then rechunked into aligned GeoTIFF or Metal tiles without reprojection\n"
      "or resampling. ZIP archives are read directly without extraction.\n"
      "\n"
      "All inputs must share one projected or geographic CRS, resolution, pixel\n"
      "registration, and sample-grid alignment. HGT names supply their one-degree\n"
      "WGS 84 bounds and their file size selects 1- or 3-arcsecond resolution.\n"
      "ASC inputs require a companion .prj; a packaged .gml can supply an authoritative\n"
      "EPSG identifier for output Metal tiles.\n"
      "Use --dry-run to validate inputs before writing.\n"
      "\n"
      "Options:\n"
      "  --output DIR        output directory (default: below data/)\n"
      "  --name NAME         output filename prefix (default: input directory name)\n"
      "  --power N           2^N cells per tile (default: 10)\n"
      "  --tile-cells N      arbitrary positive cell count instead of --power\n"
      "  --layout NAME       level-0 or level-1 (default: level-0)\n"
      "  --origin-x X        destination grid western origin (default: 0)\n"
      "  --origin-y Y        destination grid northern origin (default: 0)\n"
      "  --resolution R      output spacing (default: source resolution)\n"
      "  --nodata VALUE      output no-data value, including nan (default: 0)\n"
      "  --format NAME       geotiff or metal (default: geotiff)\n"
      "  --sample-type NAME  float32 or uint16 (Metal tiles; default: float32)\n"
      "  --lod NAME          none, point, mean, or max (Metal tiles; default: none)\n"
      "  --compression NAME  none, zlib, lz4, lzma, or lzbitmap\n"
      "                      (Metal tiles only; default: lz4)\n"
      "  --max-tiles N       stop after N output chunks; zero is unlimited\n"
      "  --overwrite         replace existing output chunks\n"
      "  --dry-run           inspect metadata and report the plan without writing\n"
      "  --help              show this help\n",
      program
  );
}

/// Parse the scalar representation stored in a Metal tile payload.
[[nodiscard]] MetalTileSampleType parse_sample_type(std::string_view text) {
  if (text == "float32") {
    return MetalTileSampleType::Float32;
  }
  if (text == "uint16") {
    return MetalTileSampleType::Uint16Decimeters;
  }
  throw std::invalid_argument("Sample type must be float32 or uint16");
}

/// Parse one codec supported by Metal's compressed-file I/O implementation.
[[nodiscard]] MetalTileCompression parse_compression(std::string_view text) {
  if (text == "none") {
    return MetalTileCompression::None;
  }
  if (text == "zlib") {
    return MetalTileCompression::Zlib;
  }
  if (text == "lz4") {
    return MetalTileCompression::Lz4;
  }
  if (text == "lzma") {
    return MetalTileCompression::Lzma;
  }
  if (text == "lzbitmap") {
    return MetalTileCompression::LzBitmap;
  }
  throw std::invalid_argument("Compression must be none, zlib, lz4, lzma, or lzbitmap");
}

/// Parse the construction method for independently loadable terrain LODs.
[[nodiscard]] LodSampling parse_lod_sampling(std::string_view text) {
  if (text == "none")
    return LodSampling::None;
  if (text == "point")
    return LodSampling::Point;
  if (text == "mean")
    return LodSampling::Mean;
  if (text == "max")
    return LodSampling::Maximum;
  throw std::invalid_argument("LOD method must be none, point, mean, or max");
}

/// Return the human-readable output representation used in progress reports.
[[nodiscard]] const char *format_name(OutputFormat format) {
  return format == OutputFormat::GeoTiff ? "GeoTIFF" : "Metal tile";
}

/// Return the command-line spelling used in default output directory names.
[[nodiscard]] const char *compression_name(MetalTileCompression compression) {
  switch (compression) {
  case MetalTileCompression::None:
    return "none";
  case MetalTileCompression::Zlib:
    return "zlib";
  case MetalTileCompression::Lz4:
    return "lz4";
  case MetalTileCompression::Lzma:
    return "lzma";
  case MetalTileCompression::LzBitmap:
    return "lzbitmap";
  }
  throw std::logic_error("Unknown Metal tile compression method");
}

/// Return the command-line spelling used in output names and progress reports.
[[nodiscard]] const char *lod_sampling_name(LodSampling sampling) {
  switch (sampling) {
  case LodSampling::None:
    return "none";
  case LodSampling::Point:
    return "point";
  case LodSampling::Mean:
    return "mean";
  case LodSampling::Maximum:
    return "max";
  }
  throw std::logic_error("Unknown LOD sampling method");
}

/// Parse a Float32 output sentinel, allowing NaN for collision-free no-data.
[[nodiscard]] float parse_float_or_nan(std::string_view text, const char *name) {
  const std::string owned(text);
  char *end = nullptr;
  errno = 0;
  const float value = std::strtof(owned.c_str(), &end);
  if (errno != 0 || end == owned.c_str() || *end != '\0' || std::isinf(value)) {
    throw std::invalid_argument(std::string("Invalid ") + name + ": " + owned);
  }
  return value;
}

/// Reject filename prefixes which could escape the selected output directory.
void validate_dataset_name(const std::string &name) {
  if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char character) {
        return std::isalnum(character) != 0 || character == '-' || character == '_' ||
               character == '.';
      })) {
    throw std::invalid_argument(
        "Dataset name must contain only letters, numbers, '.', '-', and '_'"
    );
  }
}

/// Convert the command line into one validated options structure.
[[nodiscard]] Options parse_options(int argc, const char *argv[]) {
  Options options;
  for (int index = 1; index < argc; index++) {
    const std::string_view option(argv[index]);
    if (option == "--input") {
      options.input_directory = arguments::option_value(argc, argv, index, option);
    } else if (option == "--output") {
      options.output_directory = arguments::option_value(argc, argv, index, option);
    } else if (option == "--name") {
      options.dataset_name = arguments::option_value(argc, argv, index, option);
    } else if (option == "--power") {
      options.requested_power = arguments::parse_uint32(
          arguments::option_value(argc, argv, index, option),
          "tile power",
          false
      );
      options.explicit_power = true;
    } else if (option == "--tile-cells") {
      options.tile_cell_count = arguments::parse_uint32(
          arguments::option_value(argc, argv, index, option),
          "tile cell count",
          false
      );
      options.explicit_tile_cell_count = true;
    } else if (option == "--layout") {
      const std::string_view layout = arguments::option_value(argc, argv, index, option);
      if (layout == "level-0") {
        options.layout = RasterLayout::Level0;
      } else if (layout == "level-1") {
        options.layout = RasterLayout::Level1;
      } else {
        throw std::invalid_argument("Layout must be level-0 or level-1");
      }
    } else if (option == "--origin-x") {
      options.origin_x = arguments::parse_finite_double(
          arguments::option_value(argc, argv, index, option),
          "X origin"
      );
    } else if (option == "--origin-y") {
      options.origin_y = arguments::parse_finite_double(
          arguments::option_value(argc, argv, index, option),
          "Y origin"
      );
    } else if (option == "--resolution") {
      options.resolution = arguments::parse_finite_double(
          arguments::option_value(argc, argv, index, option),
          "resolution"
      );
    } else if (option == "--nodata") {
      options.no_data =
          parse_float_or_nan(arguments::option_value(argc, argv, index, option), "no-data value");
    } else if (option == "--format") {
      const std::string_view format = arguments::option_value(argc, argv, index, option);
      if (format == "geotiff") {
        options.format = OutputFormat::GeoTiff;
      } else if (format == "metal") {
        options.format = OutputFormat::MetalTile;
      } else {
        throw std::invalid_argument("Format must be geotiff or metal");
      }
    } else if (option == "--compression") {
      options.compression = parse_compression(arguments::option_value(argc, argv, index, option));
      options.explicit_compression = true;
    } else if (option == "--sample-type") {
      options.sample_type = parse_sample_type(arguments::option_value(argc, argv, index, option));
      options.explicit_sample_type = true;
    } else if (option == "--lod") {
      options.lod_sampling = parse_lod_sampling(arguments::option_value(argc, argv, index, option));
    } else if (option == "--max-tiles") {
      options.max_tiles = arguments::parse_uint32(
          arguments::option_value(argc, argv, index, option),
          "maximum tile count",
          true
      );
    } else if (option == "--overwrite") {
      options.overwrite = true;
    } else if (option == "--dry-run") {
      options.dry_run = true;
    } else if (option == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown argument: " + std::string(option));
    }
  }

  if (options.input_directory.empty()) {
    throw std::invalid_argument("--input is required");
  }
  if (options.explicit_power && options.explicit_tile_cell_count) {
    throw std::invalid_argument("--power and --tile-cells are mutually exclusive");
  }
  if (!options.explicit_tile_cell_count) {
    if (options.requested_power >= 31U) {
      throw std::invalid_argument("Tile power must be less than 31");
    }
    options.tile_cell_count = 1U << options.requested_power;
  }
  if (options.resolution < 0.0) {
    throw std::invalid_argument("Resolution must be positive");
  }
  if (options.format == OutputFormat::GeoTiff && options.explicit_compression) {
    throw std::invalid_argument("--compression applies only to --format metal");
  }
  if (options.format == OutputFormat::GeoTiff && options.explicit_sample_type) {
    throw std::invalid_argument("--sample-type applies only to --format metal");
  }
  if (options.format == OutputFormat::GeoTiff && options.lod_sampling != LodSampling::None) {
    throw std::invalid_argument("--lod applies only to --format metal");
  }
  if (options.format == OutputFormat::MetalTile && options.layout != RasterLayout::Level0) {
    throw std::invalid_argument("Metal tiles currently support only --layout level-0");
  }
  if (options.format == OutputFormat::MetalTile && !std::has_single_bit(options.tile_cell_count)) {
    throw std::invalid_argument("Metal tiles require a power-of-two cell count");
  }
  if (options.format == OutputFormat::MetalTile && !std::isfinite(options.no_data)) {
    throw std::invalid_argument("Metal tiles require a finite --nodata value");
  }
  return options;
}

/// Return the conventional output layout name used by the legacy tool.
[[nodiscard]] const char *layout_name(RasterLayout layout) {
  return layout == RasterLayout::Level0 ? "level-0" : "level-1";
}

/// Return a concise directory component for arbitrary or power-of-two tile sizes.
[[nodiscard]] std::string directory_size_name(const Options &options) {
  if (!options.explicit_tile_cell_count) {
    return std::to_string(options.requested_power);
  }
  return "n" + std::to_string(options.tile_cell_count);
}

} // namespace
} // namespace panorama::terrain

/// Discover, rechunk, and write one directory tree of aligned DEM rasters.
int main(int argc, const char *argv[]) {
  using namespace panorama::terrain;

  try {
    Options options = parse_options(argc, argv);
    if (options.dataset_name.empty()) {
      options.dataset_name =
          std::filesystem::absolute(options.input_directory).lexically_normal().filename().string();
    }
    validate_dataset_name(options.dataset_name);

    // Prepared terrain belongs below the project's data directory regardless
    // of where the original source archive was downloaded or mounted. An
    // explicit --output remains available for external datasets and tests.
    if (options.output_directory.empty()) {
      std::string directory_name = options.dataset_name + "-" + directory_size_name(options) + "-" +
                                   layout_name(options.layout);
      if (options.format == OutputFormat::MetalTile) {
        directory_name += options.sample_type == panorama::MetalTileSampleType::Float32
                              ? "-metal-"
                              : "-metal-u16-";
        directory_name += compression_name(options.compression);
        if (options.lod_sampling != LodSampling::None) {
          directory_name += "-lod-";
          directory_name += lod_sampling_name(options.lod_sampling);
        }
      }
      options.output_directory = std::filesystem::path("data") / directory_name;
    }

    // Phase one reads only source metadata. Excluding the selected destination
    // prevents an existing nested output directory from becoming an input on
    // a subsequent overwrite run.
    const SourceCatalogue catalogue =
        SourceCatalogue::discover(options.input_directory, options.output_directory);

    const double resolution =
        options.resolution == 0.0 ? catalogue.grid().x_resolution : options.resolution;
    const DestinationGrid destination = {
        options.origin_x,
        options.origin_y,
        resolution,
        options.tile_cell_count,
        options.layout,
        options.no_data,
    };

    // Phase two derives all source and output placement from affine metadata;
    // filenames play no part in the spatial calculation.
    const RechunkPlan plan = make_rechunk_plan(catalogue, destination);
    std::printf(
        "Discovered %zu DEM rasters below %s.\n"
        "  CRS        : %s\n"
        "  Resolution : %.12g\n"
        "  Output     : %u cells, %u samples (%s, %s%s%s)\n"
        "  Directory  : %s\n"
        "  Candidates : %zu chunks\n",
        catalogue.sources().size(),
        catalogue.input_directory().c_str(),
        catalogue.grid().coordinate_system_name.c_str(),
        destination.resolution,
        destination.tile_cell_count,
        sample_side(destination),
        layout_name(destination.layout),
        format_name(options.format),
        options.lod_sampling == LodSampling::None ? "" : ", LOD ",
        options.lod_sampling == LodSampling::None ? "" : lod_sampling_name(options.lod_sampling),
        options.output_directory.c_str(),
        plan.contributors.size()
    );
    if (options.dry_run) {
      return EXIT_SUCCESS;
    }

    std::filesystem::create_directories(options.output_directory);
    std::map<ChunkKey, float> previous_maximum_by_key;
    const std::filesystem::path manifest =
        panorama::terrain_manifest_path(options.output_directory);
    if (options.format == OutputFormat::MetalTile && std::filesystem::exists(manifest)) {
      for (const panorama::TerrainManifestEntry &entry :
           panorama::read_terrain_manifest(manifest)) {
        if (!previous_maximum_by_key
                 .emplace(ChunkKey{entry.row, entry.column}, entry.maximum_elevation)
                 .second) {
          throw std::runtime_error("Terrain manifest contains duplicate tile keys");
        }
      }
    }

    // The manifest is acceleration metadata: an interrupted update must leave
    // it absent, never silently stale relative to an already replaced tile.
    if (options.format == OutputFormat::MetalTile) {
      std::filesystem::remove(manifest);
    }
    std::vector<panorama::TerrainManifestEntry> manifest_entries;
    uint32_t written = 0U;
    uint32_t skipped = 0U;
    uint32_t partial = 0U;
    uint32_t empty = 0U;
    for (const auto &[key, contributors] : plan.contributors) {
      const std::filesystem::path output =
          options.format == OutputFormat::GeoTiff
              ? geotiff_chunk_path(options.output_directory, options.dataset_name, destination, key)
              : metal_tile_chunk_path(
                    options.output_directory,
                    options.dataset_name,
                    destination,
                    key,
                    options.compression
                );
      if (std::filesystem::exists(output) && !options.overwrite) {
        if (options.max_tiles != 0U && written + skipped >= options.max_tiles) {
          break;
        }
        if (options.format == OutputFormat::MetalTile) {
          const auto previous = previous_maximum_by_key.find(key);
          const float maximum = previous == previous_maximum_by_key.end()
                                    ? panorama::read_metal_tile_header(output).maximum_elevation
                                    : previous->second;
          manifest_entries.push_back({key.row, key.column, maximum});
        }
        skipped++;
        continue;
      }

      const TerrainChunk chunk =
          build_chunk(catalogue, plan, key, contributors, options.lod_sampling);
      const bool has_coverage =
          std::any_of(chunk.covered.begin(), chunk.covered.end(), [](uint8_t value) {
            return value != 0U;
          });
      if (!has_coverage) {
        empty++;
        continue;
      }
      if (options.max_tiles != 0U && written + skipped >= options.max_tiles) {
        break;
      }
      const bool fully_covered =
          std::all_of(chunk.covered.begin(), chunk.covered.end(), [](uint8_t value) {
            return value != 0U;
          });
      if (!fully_covered) {
        partial++;
      }

      // Phase three consumes the same format-neutral chunk. GeoTIFF preserves
      // conventional GIS row order; the Metal writer converts once into the
      // exact atlas representation used by the renderer.
      if (options.format == OutputFormat::GeoTiff) {
        write_geotiff_chunk(output, chunk, destination, key, catalogue.grid());
      } else {
        const float maximum = write_metal_tile_chunk(
            output,
            chunk,
            destination,
            key,
            catalogue.grid(),
            options.compression,
            options.sample_type
        );
        manifest_entries.push_back({key.row, key.column, maximum});
      }
      written++;
    }

    if (options.format == OutputFormat::MetalTile) {
      panorama::write_terrain_manifest(manifest, manifest_entries);
    }

    std::printf(
        "Finished: %u written, %u already present, %u partial chunks, %u empty chunks skipped.\n",
        written,
        skipped,
        partial,
        empty
    );
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
