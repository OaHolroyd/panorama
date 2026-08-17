#include "raytrace_setup.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "loaded_tile.h"
#include "png_writer.h"
#include "timer.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace panorama {
namespace {

#ifndef PANORAMA_METALLIB_PATH
#define PANORAMA_METALLIB_PATH "obj/release/panorama.metallib"
#endif

constexpr const char *kMetallibPath = PANORAMA_METALLIB_PATH;

/// One horizontal direction vector with the same two-float layout as Metal's
/// `float2`; x = sin(azimuth) and y = cos(azimuth).
struct HorizontalDirection {
  float x;
  float y;
};

/// Scalar-only host/device ABI shared by every resident terrain tile.
///
/// This is mirrored exactly by `RaytraceParameters` in panorama.metal.
struct RaytraceParameters {
  float cell_size; // width of a cell (in CRS units, typically metres)
  float observer_elevation;
  uint32_t num_levels; // number of levels in the maximum mipmap
  uint32_t num_azimuth;
  uint32_t num_polar;
  float max_distance; // only used for tile-thresholding not rays
};

static_assert(sizeof(HorizontalDirection) == 2U * sizeof(float));
static_assert(sizeof(RaytraceParameters) == 6U * sizeof(uint32_t));

/// One immutable tile origin in the resident atlas.
///
/// All other tracing parameters are common to the compatible atlas slots and
/// are passed to Metal once per frontier dispatch.
struct ResidentTile {
  float tile_x_min;
  float tile_y_min;
  int64_t row;
  int64_t column;
};

static_assert(sizeof(ResidentTile) == 3U * sizeof(uint64_t));

/// One open-addressed GPU lookup-table entry for a resident tile key.
struct ResidentTileHashEntry {
  int64_t row;
  int64_t column;
  uint32_t slot;     // index into the atleses/metadata buffers
  uint32_t occupied; // whether the hash table contains a valid entry
};

static_assert(sizeof(ResidentTileHashEntry) == 3U * sizeof(uint64_t));

/// One unresolved azimuth-column segment in the GPU work frontier.
///
/// It is mirrored by `TileWorkItem` in Metal. `entry_distance` is exact;
/// only the GPU's cell/tile classification positions are nudged.
struct TileWorkItem {
  uint32_t slot;        // index into the atleses/metadata buffers
  uint32_t azimuth;     // shared azimuth for this column
  uint32_t first_polar; // the first column entry that has not recorded a collision
  uint32_t start_level; // what level to start rays at in this tile
  float entry_distance; // previous distance accrued before hitting this tile
};

static_assert(sizeof(TileWorkItem) == 5U * sizeof(uint32_t));

/// One GPU-emitted continuation whose successor tile is not resident yet.
///
/// This essentially corresponds to a `TileWorkItem` that has not been loaded
/// into the atlas buffers.
struct DeferredTileWork {
  uint32_t azimuth;
  uint32_t first_polar;
  float entry_distance;
};

static_assert(sizeof(DeferredTileWork) == 3U * sizeof(uint32_t));

/// One fully prepared source tile waiting for main-thread atlas installation.
struct PreparedTile {
  uint32_t source_index;
  std::unique_ptr<LoadedTile> tile;
};

/// Host-side lifecycle for one tile in the finite source catalogue.
///
/// Access is protected by the tile-preparation mutex. A tile becomes resident
/// only after the main thread has copied it into an atlas slot.
enum class TileLoadState : uint8_t {
  Unrequested, // not currently resident or queued for preparation
  Queued,      // preparation requested; awaiting a worker
  Loading,     // worker is loading, validating, and generating the mipmap
  Prepared,    // complete CPU-side tile awaits atlas installation
  Resident,    // installed in atlas buffers and available to GPU work
};

/// One request for a source tile, ordered by its earliest ray-entry distance.
struct TileLoadRequest {
  float priority;
  uint32_t source_index;
};

/// Order tile-load requests so the smallest ray-entry distance is served first.
struct TileLoadRequestGreater {
  /// Return whether `left` has lower priority than `right`.
  [[nodiscard]] bool operator()(const TileLoadRequest &left, const TileLoadRequest &right) const {
    return left.priority > right.priority;
  }
};

/// Key identifying one rechunked tile in the global row/column grid.
struct TileKey {
  int64_t row;
  int64_t column;

  /// Order keys by row and then column for associative containers.
  [[nodiscard]] bool operator<(const TileKey &other) const {
    if (row != other.row) {
      return row < other.row;
    }
    return column < other.column;
  }

  /// Return whether two keys refer to the same rechunked tile.
  [[nodiscard]] bool operator==(const TileKey &other) const {
    return row == other.row && column == other.column;
  }
};

/// Mix one unsigned 64-bit value for the resident-tile hash table.
[[nodiscard]] uint64_t mix_tile_hash(uint64_t value) {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return value;
}

/// Return the hash used by both host and Metal resident-tile lookup tables.
[[nodiscard]] uint64_t tile_key_hash(TileKey key) {
  const uint64_t row = mix_tile_hash(static_cast<uint64_t>(key.row));
  const uint64_t column = mix_tile_hash(static_cast<uint64_t>(key.column));
  return mix_tile_hash(row ^ (column + 0x9e3779b97f4a7c15ULL));
}

/// Return a power-of-two hash-table capacity at no more than 50% load.
[[nodiscard]] uint32_t resident_hash_capacity(uint32_t slot_count) {
  const uint64_t required = 2U * static_cast<uint64_t>(slot_count);
  uint64_t capacity = 1U;
  while (capacity < required) {
    capacity <<= 1U;
  }
  if (capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::overflow_error("Resident tile hash table exceeds Metal uint range");
  }
  return static_cast<uint32_t>(capacity);
}

/// Global rechunk-grid origin and fixed physical width of each square tile.
struct TileGrid {
  double origin_x;
  double origin_y;
  double width;
};

/// Print a Foundation error in the command-line form used by the host tools.
void print_error(NSString *context, NSError *error) {
  std::fprintf(stderr, "%s: %s\n", context.UTF8String, error.localizedDescription.UTF8String);
}

/// Return whether Metal's capture layer was enabled before this process began.
[[nodiscard]] bool capture_requested() {
  const char *value = std::getenv("MTL_CAPTURE_ENABLED");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

/// Start a queue-scoped GPU trace when Metal's capture layer is enabled.
[[nodiscard]] bool start_capture_if_requested(id<MTLCommandQueue> queue) {
  if (!capture_requested()) {
    return false;
  }

  NSString *path = [[NSFileManager.defaultManager currentDirectoryPath]
      stringByAppendingPathComponent:@"panorama.gputrace"];
  if ([NSFileManager.defaultManager fileExistsAtPath:path]) {
    throw std::runtime_error(
        "Refusing to overwrite panorama.gputrace; move or remove the existing capture first"
    );
  }

  MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
  descriptor.captureObject = queue;
  descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
  descriptor.outputURL = [NSURL fileURLWithPath:path];
  NSError *error = nil;
  if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor
                                                                      error:&error]) {
    print_error(@"Could not start the Metal GPU capture", error);
    throw std::runtime_error("Could not start the Metal GPU capture");
  }
  std::printf("Capturing GPU work to %s\n", path.UTF8String);
  return true;
}

/// Reject a count that cannot be represented by the Metal `uint` interface.
void validate_configuration(const RaytraceConfig &config) {
  if (config.num_azimuth == 0U || config.num_polar == 0U) {
    throw std::invalid_argument("Ray counts must both be positive");
  }
  if (config.tile_cache_size_bytes == 0U) {
    throw std::invalid_argument("Tile-cache byte budget must be positive");
  }
  if (config.tile_dir.empty()) {
    throw std::invalid_argument("Terrain tile directory must not be empty");
  }
  if (!std::isfinite(config.observer.easting) || !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) || !std::isfinite(config.azimuth_start) ||
      !std::isfinite(config.azimuth_end) || !std::isfinite(config.polar_start) ||
      !std::isfinite(config.polar_end) || !std::isfinite(config.tile_grid_origin_x) ||
      !std::isfinite(config.tile_grid_origin_y) || !std::isfinite(config.tile_width) ||
      !std::isfinite(config.max_distance) || config.tile_width <= 0.0 ||
      config.max_distance <= 0.0F) {
    throw std::invalid_argument("Raytrace configuration must be finite");
  }
}

/// Parse one signed tile-grid coordinate from a file-name component.
[[nodiscard]] int64_t parse_tile_coordinate(std::string_view text, const char *name) {
  int64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::invalid_argument(
        std::string("Invalid ") + name + " tile coordinate: " + std::string(text)
    );
  }
  return value;
}

/// Extract the row/column key encoded in one prepared terrain filename.
[[nodiscard]] TileKey parse_tile_name(const std::filesystem::path &path) {
  const std::string name = path.filename().string();
  const size_t row_marker = name.rfind("_r");
  const size_t column_marker = name.rfind("_c");
  const size_t extension = name.rfind(".tif");
  if (row_marker == std::string::npos || column_marker == std::string::npos ||
      extension == std::string::npos || row_marker >= column_marker || column_marker >= extension) {
    throw std::invalid_argument(
        "Prepared tile name must end in _rROW_cCOLUMN.tif: " + path.string()
    );
  }
  const TileKey key = {
      parse_tile_coordinate(
          std::string_view(name).substr(row_marker + 2U, column_marker - row_marker - 2U),
          "row"
      ),
      parse_tile_coordinate(
          std::string_view(name).substr(column_marker + 2U, extension - column_marker - 2U),
          "column"
      ),
  };
  return key;
}

/// Return the global rechunked tile key containing one projected coordinate.
[[nodiscard]] TileKey tile_key_at(const TileGrid &grid, double easting, double northing) {
  const double column_value = std::floor((easting - grid.origin_x) / grid.width);
  const double row_value = std::floor((grid.origin_y - northing) / grid.width);
  if (column_value < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      column_value > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
      row_value < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      row_value > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::out_of_range("Terrain coordinate is outside the supported tile grid");
  }
  return {static_cast<int64_t>(row_value), static_cast<int64_t>(column_value)};
}

/// Find the sole prepared GeoTIFF in `tile_dir` which has the requested key.
[[nodiscard]] std::filesystem::path
find_tile_path(const std::filesystem::path &tile_dir, TileKey key) {
  std::filesystem::path match;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(tile_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".tif") {
      continue;
    }

    try {
      if (!(parse_tile_name(entry.path()) == key)) {
        continue;
      }
    } catch (const std::invalid_argument &) {
      // Prepared-tile directories may also contain unrelated GeoTIFFs.
      continue;
    }

    if (!match.empty()) {
      throw std::runtime_error("Terrain tile directory contains duplicate row/column tile keys");
    }
    match = entry.path();
  }

  if (match.empty()) {
    throw std::runtime_error(
        "No prepared terrain tile contains the observer: " + tile_dir.string()
    );
  }
  return match;
}

/// Check that a neighbour has the same terrain layout as the origin tile.
void validate_tile_compatibility(const LoadedTile &tile, const LoadedTile &origin) {
  if (!tile.supports_level_0_collisions || tile.vertices == nullptr) {
    throw std::logic_error("Level-0 multi-tile tracing received a tile without vertices");
  }
  if (tile.crs.id() != origin.crs.id() || tile.size != origin.size || tile.delta != origin.delta ||
      tile.num_levels != origin.num_levels) {
    throw std::runtime_error("Terrain tile is incompatible with the origin tile");
  }
}

/// Check that a loaded tile's georeferencing agrees with its filename key.
void validate_tile_position(const LoadedTile &tile, TileKey key, const TileGrid &grid) {
  const double tile_width = static_cast<double>(tile.size) * tile.delta;
  const double expected_x = grid.origin_x + static_cast<double>(key.column) * grid.width;
  const double expected_y = grid.origin_y - static_cast<double>(key.row + 1) * grid.width;
  const double tolerance = 1e-6 * std::max(1.0, grid.width);
  if (!std::isfinite(tile_width) || std::abs(tile_width - grid.width) > tolerance ||
      std::abs(tile.lower_left_x - expected_x) > tolerance ||
      std::abs(tile.lower_left_y - expected_y) > tolerance) {
    throw std::runtime_error("Terrain tile georeferencing disagrees with the configured tile grid");
  }
}

/// Build the scalar Metal parameters shared by compatible terrain tiles.
[[nodiscard]] RaytraceParameters
make_raytrace_parameters(const LoadedTile &tile, const RaytraceConfig &config) {
  if (tile.delta > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::overflow_error("Raytrace geometry does not fit float32");
  }
  if (tile.num_levels == 0U || tile.num_levels >= 32U ||
      tile.size != (1U << (tile.num_levels - 1U))) {
    throw std::logic_error("Terrain tile has an invalid maximum-mipmap hierarchy");
  }

  return {
      static_cast<float>(tile.delta),
      static_cast<float>(config.observer.elevation),
      tile.num_levels,
      config.num_azimuth,
      config.num_polar,
      config.max_distance,
  };
}

/// Build the observer-relative origin and global key stored in one atlas slot.
[[nodiscard]] ResidentTile
make_resident_tile(const LoadedTile &tile, TileKey key, const RaytraceConfig &config) {
  const double local_x = tile.lower_left_x - config.observer.easting;
  const double local_y = tile.lower_left_y - config.observer.northing;
  if (local_x < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_x > static_cast<double>(std::numeric_limits<float>::max()) ||
      local_y < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_y > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::overflow_error("Resident tile origin does not fit float32");
  }
  return {static_cast<float>(local_x), static_cast<float>(local_y), key.row, key.column};
}

/// Construct float32 compass directions from evenly spaced azimuth centres.
[[nodiscard]] std::vector<HorizontalDirection>
make_azimuth_directions(const RaytraceConfig &config) {
  std::vector<HorizontalDirection> directions(config.num_azimuth);
  const double azimuth_step =
      (config.azimuth_end - config.azimuth_start) / static_cast<double>(config.num_azimuth);
  for (uint32_t index = 0; index < config.num_azimuth; index++) {
    const double bearing = config.azimuth_start + (static_cast<double>(index) + 0.5) * azimuth_step;
    directions[index] = {static_cast<float>(std::sin(bearing)),
                         static_cast<float>(std::cos(bearing))};
  }
  return directions;
}

/// Construct float32 vertical slopes from evenly spaced polar-angle centres.
[[nodiscard]] std::vector<float> make_polar_slopes(const RaytraceConfig &config) {
  std::vector<float> slopes(config.num_polar);
  const double polar_step =
      (config.polar_end - config.polar_start) / static_cast<double>(config.num_polar);
  for (uint32_t index = 0; index < config.num_polar; index++) {
    const double angle = config.polar_start + (static_cast<double>(index) + 0.5) * polar_step;
    const double slope = std::tan(angle);
    if (!std::isfinite(slope) ||
        slope < static_cast<double>(std::numeric_limits<float>::lowest()) ||
        slope > static_cast<double>(std::numeric_limits<float>::max())) {
      throw std::invalid_argument("Polar range produces an invalid float32 slope");
    }
    slopes[index] = static_cast<float>(slope);
  }
  return slopes;
}

/// Check that a byte count fits Metal's NSUInteger buffer-length argument.
[[nodiscard]] NSUInteger
checked_buffer_length(size_t element_count, size_t element_size, const char *name) {
  if (element_count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error(std::string(name) + " buffer is too large");
  }
  return static_cast<NSUInteger>(element_count * element_size);
}

/// Allocate a shared Metal buffer or report a precise command-line error.
/// If `data == nullptr`, allocate uninitialised storage. Otherwise Metal
/// copies the supplied bytes into the newly allocated buffer.
[[nodiscard]] id<MTLBuffer>
make_buffer(id<MTLDevice> device, const void *data, NSUInteger length, const char *name) {
  id<MTLBuffer> buffer = data == nullptr ? [device newBufferWithLength:length
                                                               options:MTLResourceStorageModeShared]
                                         : [device newBufferWithBytes:data
                                                               length:length
                                                              options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(std::string("Could not allocate ") + name + " Metal buffer");
  }
  return buffer;
}

/// Fill a shared Metal buffer with zero bytes before its first GPU dispatch.
void clear_buffer(id<MTLBuffer> buffer, const char *name) {
  void *contents = buffer.contents;
  if (contents == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name + " Metal buffer");
  }
  std::memset(contents, 0, buffer.length);
}

/// Return a non-owning float32 view of a completed shared Metal buffer.
[[nodiscard]] std::span<const float>
view_float_buffer(id<MTLBuffer> buffer, size_t num_value, const char *name) {
  const auto *contents = static_cast<const float *>(buffer.contents);
  if (contents == nullptr) {
    throw std::runtime_error(std::string("Could not map ") + name + " Metal buffer");
  }
  return {contents, num_value};
}

/// Return the shortest horizontal distance from the observer to a tile square.
[[nodiscard]] double
tile_minimum_distance(const TileGrid &grid, TileKey key, const ObserverLocation &observer) {
  const double x_min = grid.origin_x + static_cast<double>(key.column) * grid.width;
  const double y_max = grid.origin_y - static_cast<double>(key.row) * grid.width;
  const double x_max = x_min + grid.width;
  const double y_min = y_max - grid.width;
  const double dx = observer.easting < x_min   ? x_min - observer.easting
                    : observer.easting > x_max ? observer.easting - x_max
                                               : 0.0;
  const double dy = observer.northing < y_min   ? y_min - observer.northing
                    : observer.northing > y_max ? observer.northing - y_max
                                                : 0.0;
  return std::hypot(dx, dy);
}

/// Go through all files in the tile directory and record which tiles are
/// available for future loading.
///
/// The resident cache loads this conservative catalogue on demand. A future
/// manifest or spatial index can replace this one-time directory scan without
/// changing the cache or GPU-frontier interfaces.
[[nodiscard]] std::vector<std::pair<TileKey, std::filesystem::path>> find_terrain_sources(
    const std::filesystem::path &tile_dir,
    TileKey origin_key,
    const TileGrid &grid,
    const RaytraceConfig &config
) {
  std::vector<std::pair<TileKey, std::filesystem::path>> tiles;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(tile_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".tif") {
      continue;
    }
    TileKey key;
    try {
      key = parse_tile_name(entry.path());
    } catch (const std::invalid_argument &) {
      // The prepared-tile directory can contain unrelated GeoTIFFs or other files.
      continue;
    }
    if (key == origin_key ||
        tile_minimum_distance(grid, key, config.observer) <= config.max_distance) {
      tiles.emplace_back(key, entry.path());
    }
  }
  std::sort(tiles.begin(), tiles.end(), [origin_key](const auto &left, const auto &right) {
    const uint64_t left_shell =
        static_cast<uint64_t>(std::llabs(left.first.row - origin_key.row)) +
        static_cast<uint64_t>(std::llabs(left.first.column - origin_key.column));
    const uint64_t right_shell =
        static_cast<uint64_t>(std::llabs(right.first.row - origin_key.row)) +
        static_cast<uint64_t>(std::llabs(right.first.column - origin_key.column));
    if (left_shell != right_shell) {
      return left_shell < right_shell;
    }
    return left.first < right.first;
  });
  if (tiles.empty() || !(tiles.front().first == origin_key)) {
    throw std::runtime_error("Could not find the observer tile in its prepared-tile directory");
  }
  return tiles;
}

} // namespace

/// Trace a fixed observer's angular ray field through a set of terrain tiles
/// using a GPU-owned frontier and a CPU-owned resident-tile cache.
///
/// Initially, one work item represents the unresolved polar-ray column for
/// each azimuth entering the observer tile. Each frontier iteration traces
/// every resident work item through its tile, writes terrain hits, and emits
/// at most one unresolved suffix per azimuth. A suffix whose successor tile
/// is resident immediately joins the next frontier; otherwise the host keeps
/// its exact entry distance, requests that tile's preparation, and activates
/// the suffix once an atlas slot becomes available.
///
/// Background workers load, validate, and mipmap requested source tiles. The
/// main thread installs completed tiles into fixed-stride vertex and maximum-
/// mipmap atlases between GPU command buffers, rebuilding the GPU tile-key
/// lookup table after changes. When the atlas is full, it evicts the least-
/// recently-used slot not referenced by the imminent frontier. The loop ends
/// when every azimuth column has intersected terrain, reached the range limit,
/// or left available terrain coverage.
void raytrace_tiled_heightmap(const RaytraceConfig &config) {
  validate_configuration(config);
  // Start a composite timer.
  Timer timer("Total elapsed");
  timer.start_wall("Initial setup");

  // Locate the origin tile (whose dimensions will be used for subsequent tiles)
  // TODO: support starting outside of an origin tile
  const TileGrid grid = {
      config.tile_grid_origin_x,
      config.tile_grid_origin_y,
      config.tile_width,
  };
  const TileKey origin_key = tile_key_at(grid, config.observer.easting, config.observer.northing);
  const std::filesystem::path origin_path = find_tile_path(config.tile_dir, origin_key);

  // The observer tile establishes common data dimensions and the projected
  // coordinate system required by every fixed-stride atlas slot.
  timer.start_work("Tile load");
  LoadedTile origin = LoadedTile::load_tif(origin_path, true);
  timer.stop("Tile load");

  timer.start_work("Mipmap generation");
  origin.compute_mipmap();
  timer.stop("Mipmap generation");

  validate_tile_position(origin, origin_key, grid);

  // Build a deliberately conservative finite source catalogue: every prepared
  // tile which could be reached within the horizontal range. The observer tile
  // is installed immediately; workers prepare other sources only after the
  // GPU frontier or a small neighbour prefetch requests them.
  std::vector<std::pair<TileKey, std::filesystem::path>> paths =
      find_terrain_sources(config.tile_dir, origin_key, grid, config);
  if (config.max_tile_count != 0U && paths.size() > config.max_tile_count) {
    paths.resize(config.max_tile_count);
  }

  // Every rechunked tile must have the origin tile's dimensions, allowing
  // constant per-slot atlas strides. Derive the number of permitted slots
  // from the byte budget rather than hard-coding a tile count.
  const size_t mip_count = origin.mipmap.size();
  const size_t vertex_count = origin.vertices->size();
  const size_t tile_bytes =
      checked_buffer_length(mip_count + vertex_count, sizeof(float), "terrain tile");
  const uint64_t slot_capacity = config.tile_cache_size_bytes / tile_bytes;
  if (slot_capacity == 0U) {
    throw std::runtime_error("Tile-cache byte budget cannot hold one terrain tile");
  }
  const uint32_t tile_count = static_cast<uint32_t>(paths.size());
  const uint64_t bounded_slot_count = std::min(slot_capacity, static_cast<uint64_t>(tile_count));
  if (bounded_slot_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::overflow_error("Tile-cache slot count exceeds Metal uint range");
  }
  const uint32_t atlas_slot_count = static_cast<uint32_t>(bounded_slot_count);
  const uint32_t resident_hash_slot_count = resident_hash_capacity(atlas_slot_count);
  std::map<TileKey, uint32_t> source_index_by_key;
  for (uint32_t source_index = 0U; source_index < tile_count; source_index++) {
    const bool inserted =
        source_index_by_key.emplace(paths[source_index].first, source_index).second;
    if (!inserted) {
      throw std::runtime_error("Prepared-tile directory contains duplicate tile keys");
    }
  }

  // A column has exactly one unresolved suffix globally: it either hits,
  // terminates at the range limit, or exits one tile and becomes one successor
  // segment. Therefore every active, deferred, and waiting frontier is bounded
  // by the azimuth count rather than by the number of terrain sources.
  const size_t frontier_capacity = config.num_azimuth;

  // Angular data and output images are shared by the complete GPU frontier,
  // rather than recreated for every individual terrain-tile dispatch.
  const std::vector<HorizontalDirection> directions = make_azimuth_directions(config);
  const std::vector<float> slopes = make_polar_slopes(config);
  const size_t ray_count = static_cast<size_t>(config.num_azimuth) * config.num_polar;
  const RaytraceParameters shared_parameters = make_raytrace_parameters(origin, config);

  @autoreleasepool {
    // Create the Metal device and the two pipelines which make one frontier
    // iteration: trace active tile segments, then emit their successors.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("No Metal device is available");
    }

    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      throw std::runtime_error("Could not load Metal library");
    }
    id<MTLFunction> trace = [library newFunctionWithName:@"trace_tile_frontier"];
    id<MTLFunction> emit = [library newFunctionWithName:@"emit_tile_frontier"];
    if (trace == nil || emit == nil) {
      throw std::runtime_error("GPU-frontier Metal kernels are missing");
    }

    id<MTLComputePipelineState> trace_pipeline =
        [device newComputePipelineStateWithFunction:trace error:&error];
    if (trace_pipeline == nil) {
      print_error(@"Could not create trace pipeline", error);
      throw std::runtime_error("Could not create trace pipeline");
    }

    id<MTLComputePipelineState> emit_pipeline = [device newComputePipelineStateWithFunction:emit
                                                                                      error:&error];
    if (emit_pipeline == nil) {
      print_error(@"Could not create continuation pipeline", error);
      throw std::runtime_error("Could not create continuation pipeline");
    }

    // Fixed-stride atlas buffers hold the bounded resident cache. A work
    // item's slot selects its terrain by multiplying this common stride in
    // Metal; source tiles not currently resident occupy no slot.
    id<MTLBuffer> mip_atlas = make_buffer(
        device,
        nullptr,
        checked_buffer_length(atlas_slot_count * mip_count, sizeof(float), "mipmap atlas"),
        "mipmap atlas"
    );
    id<MTLBuffer> vertex_atlas = make_buffer(
        device,
        nullptr,
        checked_buffer_length(atlas_slot_count * vertex_count, sizeof(float), "vertex atlas"),
        "vertex atlas"
    );

    // Shared storage lets the main thread install completed background work
    // directly between GPU command buffers. Future cache work may upload
    // through a staging/blit path instead.
    auto *mips = static_cast<float *>(mip_atlas.contents);
    auto *vertices = static_cast<float *>(vertex_atlas.contents);
    if (mips == nullptr || vertices == nullptr) {
      throw std::runtime_error("Could not map terrain atlas");
    }

    // Tile metadata mirrors the atlas slots and contains the observer-relative
    // origin that cannot be inferred from a tile's raw height samples. It
    // initially contains only the observer tile; later slots arrive on demand.
    id<MTLBuffer> tile_buffer = make_buffer(
        device,
        nullptr,
        checked_buffer_length(atlas_slot_count, sizeof(ResidentTile), "tile metadata"),
        "tile metadata"
    );
    auto *resident_tiles = static_cast<ResidentTile *>(tile_buffer.contents);
    if (resident_tiles == nullptr) {
      throw std::runtime_error("Could not map resident tile metadata");
    }

    // Create a mapping from global integer tile keys to resident atlas slots.
    id<MTLBuffer> resident_hash_buffer = make_buffer(
        device,
        nullptr,
        checked_buffer_length(
            resident_hash_slot_count,
            sizeof(ResidentTileHashEntry),
            "resident tile hash"
        ),
        "resident tile hash"
    );
    auto *resident_hash = static_cast<ResidentTileHashEntry *>(resident_hash_buffer.contents);
    if (resident_hash == nullptr) {
      throw std::runtime_error("Could not map resident tile hash");
    }

    // Slot zero reuses the observer tile already loaded above.
    std::memcpy(mips, origin.mipmap.data(), mip_count * sizeof(float));
    std::memcpy(vertices, origin.vertices->data(), vertex_count * sizeof(float));
    resident_tiles[0] = make_resident_tile(origin, origin_key, config);

    // A source-to-slot map records which catalogue tiles are currently in the
    // bounded atlas. `atlas_slot_count` is the out-of-band nonresident value.
    std::vector<uint32_t> resident_slot_by_source(tile_count, atlas_slot_count);
    resident_slot_by_source[0] = 0U;
    std::vector<uint32_t> source_by_resident_slot(atlas_slot_count, tile_count);
    source_by_resident_slot[0] = 0U;
    std::vector<uint64_t> slot_last_used(atlas_slot_count, 0U);
    slot_last_used[0] = 1U;
    uint64_t next_use_stamp = 2U;
    uint32_t resident_tile_count = 1U;

    // Count the observer tile as the first atlas installation: it copied the
    // same vertex and mipmap payload as every later prepared tile.
    uint64_t atlas_installations = 1U;
    uint64_t atlas_bytes_copied = static_cast<uint64_t>(tile_bytes);
    uint64_t cache_evictions = 0U;

    /// Rebuild the GPU lookup table after installing or evicting a tile.
    auto rebuild_resident_hash = [&] {
      // Clearing leaves every bucket unoccupied. The GPU stops its matching
      // linear probe at the first such bucket, so no tombstone state is needed.
      std::fill_n(resident_hash, resident_hash_slot_count, ResidentTileHashEntry{});

      // `resident_hash_slot_count` is a power of two. Masking is therefore
      // equivalent to a modulo operation for both the initial bucket and each
      // later linear-probe step.
      const uint32_t mask = resident_hash_slot_count - 1U;
      for (uint32_t slot = 0U; slot < resident_tile_count; slot++) {
        const uint32_t source_index = source_by_resident_slot[slot];
        if (source_index == tile_count) {
          // An evicted slot remains part of the fixed-size atlas allocation,
          // but has no source tile to expose through the GPU lookup table.
          continue;
        }

        const TileKey key = paths[source_index].first;
        uint32_t hash_index = static_cast<uint32_t>(tile_key_hash(key)) & mask;
        uint32_t probe = 0U;

        // Resolve collisions by linear probing. The table is deliberately at
        // most half full, so successful and unsuccessful GPU lookups remain
        // short in the usual case.
        while (resident_hash[hash_index].occupied != 0U) {
          if (resident_hash[hash_index].row == key.row &&
              resident_hash[hash_index].column == key.column) {
            throw std::logic_error("Resident tile hash contains a duplicate tile key");
          }
          hash_index = (hash_index + 1U) & mask;
          probe++;
          if (probe == resident_hash_slot_count) {
            // This should be impossible with the 50%-load sizing rule; keep
            // the explicit check so a broken invariant fails diagnostically.
            throw std::logic_error("Resident tile hash table is full");
          }
        }

        // Publish the resolved row/column-to-slot mapping for the next GPU
        // frontier dispatch.
        resident_hash[hash_index] = {key.row, key.column, slot, 1U};
      }
    };
    rebuild_resident_hash();

    // Independent GeoTIFF files can be decoded and mipmapped concurrently.
    // Workers return immutable results to the main thread, which alone owns
    // atlas-slot installation between GPU command buffers.
    std::atomic<bool> stop_workers{false};
    std::exception_ptr preparation_error;
    std::mutex preparation_mutex;

    // These counters are protected by `preparation_mutex` when workers update
    // them. The main thread reads them only after every worker has joined.
    // Keep request, actual I/O, and cache-reload counts separate. A request
    // may be deduplicated while queued, whereas an evicted source can require
    // another actual load operation later in the same render.
    uint64_t source_load_requests = 0U;
    uint64_t source_load_operations = 0U;
    uint64_t unique_source_loads = 0U;
    uint64_t source_reloads = 0U;

    // Workers sleep until priority work arrives. The main thread sleeps until
    // a complete tile is ready, and workers sleep when the bounded prepared
    // queue is full and must wait for an atlas installation to consume it.
    std::condition_variable load_request_available;
    std::condition_variable prepared_available;
    std::condition_variable prepared_space_available;

    // Requests are ordered by the exact horizontal entry distance of the ray
    // suffix that needs them. The per-source state and best queued priority
    // make repeated requests cheap and allow stale queue entries to be ignored.
    std::priority_queue<TileLoadRequest, std::vector<TileLoadRequest>, TileLoadRequestGreater>
        load_requests;
    std::vector<TileLoadState> load_states(tile_count, TileLoadState::Unrequested);
    load_states[0] = TileLoadState::Resident;
    std::vector<float> queued_priorities(tile_count, std::numeric_limits<float>::infinity());

    // Remember whether each source has ever reached a worker. This is separate
    // from its current state so the final diagnostics can distinguish unique
    // first loads from reloads after atlas eviction.
    std::vector<uint8_t> source_loaded_before(tile_count, 0U);

    // A prepared tile owns CPU-side vertex and mipmap vectors until the main
    // thread copies them into a resident atlas slot. Its bounded size prevents
    // background preparation from consuming unbounded system memory.
    std::deque<PreparedTile> prepared_tiles;

    // Decide how many concurrent threads to use to load/prep tiles. Default to
    // using all available threads (denoted by the caller specifying zero)
    // unless the caller has specified otherwise.
    const uint32_t hardware_threads = std::thread::hardware_concurrency();
    const uint32_t available_workers =
        config.max_tile_preparation_workers == 0U
            ? std::max(1U, hardware_threads)
            : std::min(config.max_tile_preparation_workers, std::max(1U, hardware_threads));
    const uint32_t worker_count = std::min(tile_count - 1U, available_workers);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    /// Queue one nonresident source tile for priority-ordered preparation.
    auto request_tile_load = [&](uint32_t source_index, float priority) {
      std::lock_guard<std::mutex> lock(preparation_mutex);
      TileLoadState &state = load_states[source_index];
      if (state == TileLoadState::Unrequested) {
        state = TileLoadState::Queued;
        queued_priorities[source_index] = priority;
        load_requests.push({priority, source_index});
        source_load_requests++;
        load_request_available.notify_one();
      } else if (state == TileLoadState::Queued && priority < queued_priorities[source_index]) {
        // Priority queues cannot decrease a key in place. Push a replacement
        // request and let workers discard the now-stale lower-priority entry.
        queued_priorities[source_index] = priority;
        load_requests.push({priority, source_index});
        load_request_available.notify_one();
      }
    };

    /// Start workers only after permanent host/GPU setup has succeeded.
    ///
    /// Each worker repeatedly removes the highest-priority current request,
    /// performs the expensive file loading and mipmap construction without
    /// holding `preparation_mutex`, then publishes the finished CPU tile to
    /// `prepared_tiles`. The main thread alone installs that tile in an atlas
    /// slot, so workers never write GPU-visible terrain buffers directly.
    auto start_workers = [&] {
      for (uint32_t worker = 0U; worker < worker_count; worker++) {
        workers.emplace_back([&] {
          while (true) {
            // Wait without polling until a requested tile is available, or
            // until the main thread asks every worker to stop.
            TileLoadRequest request = {};
            {
              std::unique_lock<std::mutex> lock(preparation_mutex);
              load_request_available.wait(lock, [&] {
                return stop_workers.load(std::memory_order_relaxed) || !load_requests.empty();
              });
              if (stop_workers.load(std::memory_order_relaxed)) {
                break;
              }

              // A newer request may have lowered this source's priority while
              // this entry was waiting in the heap. Ignore those stale heap
              // entries; the replacement remains queued at the new priority.
              request = load_requests.top();
              load_requests.pop();
              if (load_states[request.source_index] != TileLoadState::Queued ||
                  request.priority != queued_priorities[request.source_index]) {
                continue;
              }

              // Claim the request while holding the mutex so no other worker
              // can prepare the same source simultaneously. Release it before
              // the I/O and CPU-heavy work below.
              load_states[request.source_index] = TileLoadState::Loading;
            }

            try {
              const auto &[key, path] = paths[request.source_index];

              // Record total work-time as well as the wall-time
              timer.start_work("Tile load");
              auto tile = std::make_unique<LoadedTile>(LoadedTile::load_tif(path, true));
              timer.stop("Tile load");

              // Count physical load operations separately from unique sources:
              // a source evicted from the bounded atlas can later be loaded
              // again if a deferred frontier item reaches it.
              {
                std::lock_guard<std::mutex> lock(preparation_mutex);
                source_load_operations++;
                if (source_loaded_before[request.source_index] == 0U) {
                  source_loaded_before[request.source_index] = 1U;
                  unique_source_loads++;
                } else {
                  source_reloads++;
                }
              }

              // Ensure the loaded tile is in the correct place on the grid and
              // matches the size of the origin tile.
              validate_tile_compatibility(*tile, origin);
              validate_tile_position(*tile, key, grid);

              // Build the flat maximum-mipmap on the CPU while the source
              // elevations are private to this worker. The main thread will
              // subsequently copy both vectors into matching atlas ranges.
              timer.start_work("Mipmap generation");
              tile->compute_mipmap();
              timer.stop("Mipmap generation");
              if (tile->mipmap.size() != mip_count || tile->vertices->size() != vertex_count) {
                throw std::runtime_error("Resident tile does not match the atlas dimensions");
              }

              // Limit the hand-off queue to one tile per atlas slot. This
              // bounds prepared-but-uninstalled CPU memory and naturally
              // applies back-pressure when GPU frontier processing cannot
              // install tiles as fast as workers prepare them.
              std::unique_lock<std::mutex> lock(preparation_mutex);
              prepared_space_available.wait(lock, [&] {
                return stop_workers.load(std::memory_order_relaxed) ||
                       prepared_tiles.size() < atlas_slot_count;
              });
              if (stop_workers.load(std::memory_order_relaxed)) {
                break;
              }

              // Publish the completed tile atomically with its state change,
              // then wake the main thread if it is waiting for availability.
              load_states[request.source_index] = TileLoadState::Prepared;
              prepared_tiles.push_back({request.source_index, std::move(tile)});
              lock.unlock();
              prepared_available.notify_one();
            } catch (...) {
              // A worker cannot throw across its thread boundary. Preserve the
              // first error for the main thread, stop peer workers, and wake
              // every condition-variable waiter so shutdown can make progress.
              std::lock_guard<std::mutex> lock(preparation_mutex);
              if (preparation_error == nullptr) {
                preparation_error = std::current_exception();
              }
              stop_workers.store(true, std::memory_order_relaxed);
              prepared_available.notify_all();
              prepared_space_available.notify_all();
              break;
            }
          }
        });
      }
    };

    /// Install as many prepared tiles as fit in atlas slots safe to overwrite.
    ///
    /// `pinned_slots` names terrain that the imminent GPU dispatch will read.
    /// Such a slot is never evicted here: shared buffers must keep the same
    /// tile data from command encoding until that command has completed.
    auto install_prepared_tiles = [&](const std::vector<uint8_t> &pinned_slots) {
      timer.start_wall("Atlas installation");

      // Worker failures are stored rather than thrown from their threads.
      // Surface one before changing cache state or encoding another pass.
      std::exception_ptr error;
      {
        std::lock_guard<std::mutex> lock(preparation_mutex);
        error = preparation_error;
      }
      if (error != nullptr) {
        std::rethrow_exception(error);
      }

      bool hash_changed = false;
      while (true) {
        // Fill never-used slots first. Once the atlas is full, select the
        // least-recently-used unpinned slot; the use stamp is updated both
        // when a tile is installed and when the active frontier refers to it.
        uint32_t slot = atlas_slot_count;
        if (resident_tile_count < atlas_slot_count) {
          slot = resident_tile_count;
        } else {
          uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
          for (uint32_t candidate = 0U; candidate < atlas_slot_count; candidate++) {
            if (pinned_slots[candidate] == 0U && slot_last_used[candidate] < oldest_use) {
              slot = candidate;
              oldest_use = slot_last_used[candidate];
            }
          }
        }

        if (slot == atlas_slot_count) {
          // Every slot belongs to the imminent GPU pass. Keep prepared tiles
          // queued until a later pass makes an eviction safe.
          break;
        }

        // Remove one completed CPU tile under the queue mutex, then release
        // it before copying. This lets a worker place its next prepared tile
        // into the bounded queue while the main thread installs this one.
        PreparedTile prepared = {};
        {
          std::lock_guard<std::mutex> lock(preparation_mutex);
          if (prepared_tiles.empty()) {
            break;
          }
          prepared = std::move(prepared_tiles.front());
          prepared_tiles.pop_front();
        }
        prepared_space_available.notify_one();

        // Copy the two immutable terrain payloads into their selected shared
        // atlas slot. Mipmap and vertex arrays use separate fixed-stride
        // buffers, so every kernel work item can find slot `s` by adding
        // `s * mip_count` or `s * vertex_count`. This is the most likely
        // cache-size-sensitive CPU cost.
        timer.start_wall("Atlas copy");
        std::memcpy(
            mips + static_cast<size_t>(slot) * mip_count,
            prepared.tile->mipmap.data(),
            mip_count * sizeof(float)
        );
        std::memcpy(
            vertices + static_cast<size_t>(slot) * vertex_count,
            prepared.tile->vertices->data(),
            vertex_count * sizeof(float)
        );
        timer.stop("Atlas copy");
        resident_tiles[slot] =
            make_resident_tile(*prepared.tile, paths[prepared.source_index].first, config);

        // Remove a displaced source from host residency before assigning the
        // new source. `tile_count` is the sentinel for a never-populated slot.
        // An evicted source can be queued and prepared again if later frontier
        // work needs it.
        const uint32_t evicted_source = source_by_resident_slot[slot];
        if (evicted_source != tile_count) {
          resident_slot_by_source[evicted_source] = atlas_slot_count;
          std::lock_guard<std::mutex> lock(preparation_mutex);
          load_states[evicted_source] = TileLoadState::Unrequested;
          cache_evictions++;
        }

        // Publish the host-side bidirectional source/slot mapping. The GPU
        // lookup table itself is rebuilt after this installation batch, so it
        // never observes an individually half-updated mapping.
        resident_slot_by_source[prepared.source_index] = slot;
        source_by_resident_slot[slot] = prepared.source_index;
        slot_last_used[slot] = next_use_stamp++;
        if (resident_tile_count < atlas_slot_count) {
          resident_tile_count++;
        }
        atlas_installations++;
        atlas_bytes_copied += static_cast<uint64_t>(tile_bytes);
        {
          std::lock_guard<std::mutex> lock(preparation_mutex);
          load_states[prepared.source_index] = TileLoadState::Resident;
        }

        // Rebuilding once after several installations avoids publishing a
        // partially updated hash table and amortises the linear rebuild cost.
        hash_changed = true;
      }

      if (hash_changed) {
        // The next compute command consults this table to turn successor tile
        // grid keys into atlas slots. Rebuild before that command is encoded.
        timer.start_wall("Resident hash rebuild");
        rebuild_resident_hash();
        timer.stop("Resident hash rebuild");
      }
      timer.stop("Atlas installation");
    };

    /// Stop workers, wake bounded-queue waiters, and join every worker thread.
    ///
    /// Each wait predicate observes `stop_workers`, so notifying all three
    /// queues guarantees that workers blocked for input or prepared-space can
    /// exit rather than preventing exception handling or final shutdown.
    auto stop_and_join_workers = [&] {
      stop_workers.store(true, std::memory_order_relaxed);
      load_request_available.notify_all();
      prepared_available.notify_all();
      prepared_space_available.notify_all();
      for (std::thread &worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
    };

    // Allocate the ray/output buffers used by every frontier pass. Distance
    // and elevation zero remain the no-hit sky sentinel.
    id<MTLBuffer> azimuth_buffer = make_buffer(
        device,
        directions.data(),
        checked_buffer_length(directions.size(), sizeof(HorizontalDirection), "azimuth directions"),
        "azimuth directions"
    );
    id<MTLBuffer> slope_buffer = make_buffer(
        device,
        slopes.data(),
        checked_buffer_length(slopes.size(), sizeof(float), "polar slopes"),
        "polar slopes"
    );
    id<MTLBuffer> distances = make_buffer(
        device,
        nullptr,
        checked_buffer_length(ray_count, sizeof(float), "distance output"),
        "distance output"
    );
    id<MTLBuffer> elevations = make_buffer(
        device,
        nullptr,
        checked_buffer_length(ray_count, sizeof(float), "elevation output"),
        "elevation output"
    );
    clear_buffer(distances, "distance output");
    clear_buffer(elevations, "elevation output");

    // The frontier is double buffered. A work item represents one unresolved
    // azimuth-column segment in one resident tile; its polar rays are
    // independently traced by the first kernel. One suffix per azimuth means
    // every frontier buffer needs only `frontier_capacity` entries.
    id<MTLBuffer> work_a = make_buffer(
        device,
        nullptr,
        checked_buffer_length(frontier_capacity, sizeof(TileWorkItem), "active frontier"),
        "active frontier"
    );
    id<MTLBuffer> work_b = make_buffer(
        device,
        nullptr,
        checked_buffer_length(frontier_capacity, sizeof(TileWorkItem), "next frontier"),
        "next frontier"
    );
    id<MTLBuffer> unresolved = make_buffer(
        device,
        nullptr,
        checked_buffer_length(frontier_capacity, sizeof(uint32_t), "unresolved polar indices"),
        "unresolved polar indices"
    );
    id<MTLBuffer> next_count =
        make_buffer(device, nullptr, sizeof(uint32_t), "next frontier count");
    id<MTLBuffer> deferred_items = make_buffer(
        device,
        nullptr,
        checked_buffer_length(frontier_capacity, sizeof(DeferredTileWork), "deferred frontier"),
        "deferred frontier"
    );
    id<MTLBuffer> deferred_count =
        make_buffer(device, nullptr, sizeof(uint32_t), "deferred frontier count");

    // Initially every azimuth column starts at the observer tile with all
    // polar rays unresolved. The observer tile begins at mip level 1; tiles
    // reached through a boundary begin at their maximum level.
    auto *initial = static_cast<TileWorkItem *>(work_a.contents);
    for (uint32_t azimuth = 0U; azimuth < config.num_azimuth; azimuth++) {
      initial[azimuth] = {0U, azimuth, 0U, 1U, 0.0F};
    }

    // The observer tile is resident, so start GPU tracing while background
    // workers continue to prepare later source tiles.
    timer.stop("Initial setup");

    // Command buffers are still synchronised one frontier iteration at a time
    // because the CPU needs the next append count to size the following grid.
    // Removing this wait is a later asynchronous-cache optimisation.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      throw std::runtime_error("Could not create Metal command queue");
    }

    const bool capture = start_capture_if_requested(queue);
    uint32_t active_count = config.num_azimuth;
    id<MTLBuffer> active = work_a, next = work_b;
    std::vector<DeferredTileWork> waiting_work;

    // Reused by the host-side checks below. These make the one-suffix-per-
    // azimuth invariant explicit at the CPU/GPU frontier boundary, where an
    // accidental duplicate would otherwise look like an atlas-cache issue.
    std::vector<uint8_t> claimed_azimuth(config.num_azimuth, 0U);

    // Count kernel lookup outcomes separately from host requests. A deferred
    // successor may later prove to be open sky when absent from the catalogue.
    uint64_t resident_successor_work = 0U;
    uint64_t deferred_successor_work = 0U;

    /// Verify that a GPU work buffer contains one valid entry per azimuth.
    auto validate_frontier = [&](id<MTLBuffer> buffer, uint32_t count, const char *name) {
      if (count > frontier_capacity) {
        throw std::runtime_error(std::string(name) + " exceeds the azimuth frontier capacity");
      }

      const auto *items = static_cast<const TileWorkItem *>(buffer.contents);
      if (items == nullptr) {
        throw std::runtime_error(std::string("Could not map ") + name);
      }

      std::fill(claimed_azimuth.begin(), claimed_azimuth.end(), 0U);
      for (uint32_t index = 0U; index < count; index++) {
        const uint32_t azimuth = items[index].azimuth;
        if (azimuth >= config.num_azimuth || claimed_azimuth[azimuth] != 0U) {
          throw std::runtime_error(
              std::string(name) + " violates the one-suffix-per-azimuth invariant"
          );
        }
        claimed_azimuth[azimuth] = 1U;
      }
    };

    /// Verify that deferred host work also preserves one suffix per azimuth.
    auto validate_waiting_work = [&] {
      if (waiting_work.size() > frontier_capacity) {
        throw std::runtime_error("Deferred frontier exceeds the azimuth frontier capacity");
      }

      std::fill(claimed_azimuth.begin(), claimed_azimuth.end(), 0U);
      for (const DeferredTileWork &deferred : waiting_work) {
        if (deferred.azimuth >= config.num_azimuth || claimed_azimuth[deferred.azimuth] != 0U) {
          throw std::runtime_error(
              "Deferred frontier violates the one-suffix-per-azimuth invariant"
          );
        }
        claimed_azimuth[deferred.azimuth] = 1U;
      }
    };

    /// Append deferred continuations whose source tile now owns an atlas slot.
    auto activate_waiting_work = [&](id<MTLBuffer> buffer, uint32_t count) {
      auto *items = static_cast<TileWorkItem *>(buffer.contents);
      if (items == nullptr) {
        throw std::runtime_error("Could not map active frontier buffer");
      }
      std::vector<DeferredTileWork> still_waiting;
      still_waiting.reserve(waiting_work.size());
      for (const DeferredTileWork &deferred : waiting_work) {
        float x = deferred.entry_distance * directions[deferred.azimuth].x;
        float y = deferred.entry_distance * directions[deferred.azimuth].y;
        const float nudge = std::max(
            1e-3F * shared_parameters.cell_size,
            8.0F * std::numeric_limits<float>::epsilon() *
                std::max(1.0F, std::max(std::abs(x), std::abs(y)))
        );
        if (directions[deferred.azimuth].x != 0.0F) {
          x += std::copysign(nudge, directions[deferred.azimuth].x);
        }
        if (directions[deferred.azimuth].y != 0.0F) {
          y += std::copysign(nudge, directions[deferred.azimuth].y);
        }
        const TileKey key = tile_key_at(
            grid,
            config.observer.easting + static_cast<double>(x),
            config.observer.northing + static_cast<double>(y)
        );
        const auto source_iterator = source_index_by_key.find(key);
        if (source_iterator == source_index_by_key.end()) {
          // No prepared source covers this successor, so it is open sky.
          continue;
        }
        const uint32_t slot = resident_slot_by_source[source_iterator->second];
        if (slot == atlas_slot_count) {
          // Use the exact hand-off distance as the loading priority: nearer
          // requested terrain is more likely to unblock the next frontier.
          request_tile_load(source_iterator->second, deferred.entry_distance);
          still_waiting.push_back(deferred);
          continue;
        }
        if (count >= frontier_capacity) {
          throw std::runtime_error("GPU frontier exceeds the azimuth frontier capacity");
        }
        items[count] = {
            slot,
            deferred.azimuth,
            deferred.first_polar,
            shared_parameters.num_levels,
            deferred.entry_distance,
        };
        count++;
      }
      waiting_work = std::move(still_waiting);
      return count;
    };

    /// Return slots which the supplied imminent frontier must keep resident.
    auto pin_frontier_slots = [&](id<MTLBuffer> buffer, uint32_t count, bool record_use) {
      std::vector<uint8_t> pinned(atlas_slot_count, 0U);
      const auto *items = static_cast<const TileWorkItem *>(buffer.contents);
      if (items == nullptr) {
        throw std::runtime_error("Could not map active frontier buffer");
      }
      for (uint32_t index = 0U; index < count; index++) {
        const uint32_t slot = items[index].slot;
        if (slot >= resident_tile_count) {
          throw std::logic_error("GPU frontier refers to a nonresident tile slot");
        }
        pinned[slot] = 1U;
        if (record_use) {
          slot_last_used[slot] = next_use_stamp++;
        }
      }
      return pinned;
    };

    /// Request all available tiles directly neighbouring the observer tile.
    auto prefetch_observer_neighbours = [&] {
      for (int64_t row_offset = -1; row_offset <= 1; row_offset++) {
        for (int64_t column_offset = -1; column_offset <= 1; column_offset++) {
          if (row_offset == 0 && column_offset == 0) {
            continue;
          }
          const TileKey neighbour_key = {
              origin_key.row + row_offset,
              origin_key.column + column_offset,
          };
          const auto source_iterator = source_index_by_key.find(neighbour_key);
          if (source_iterator != source_index_by_key.end()) {
            request_tile_load(
                source_iterator->second,
                static_cast<float>(tile_minimum_distance(grid, neighbour_key, config.observer))
            );
          }
        }
      }
    };

    // Wall time includes the CPU's per-pass buffer setup, command encoding,
    // submission, and waits. The device timestamps recorded below separately
    // report the GPU's execution-only work within this same region.
    timer.start_wall("GPU raytrace");
    try {
      // Loading starts only after every permanent buffer and the command queue
      // exists. Any later failure is caught below, which stops and joins these
      // threads before unwinding their owning vector.
      start_workers();
      prefetch_observer_neighbours();

      // Keep going until all rays have completed
      while (active_count != 0U) {
        // Account separately for CPU frontier bookkeeping before the Metal
        // command is created. This includes LRU use stamps and counter reset.
        timer.start_wall("Frontier bookkeeping");

        // Each azimuth can have only one unresolved tile segment. Validate
        // that the preceding emission/install pass preserved that contract
        // before this buffer is handed to Metal again.
        validate_frontier(active, active_count, "active frontier");

        // This pass is about to read every referenced slot. Updating the LRU
        // stamp now makes recently traced terrain the last cache victim.
        (void)pin_frontier_slots(active, active_count, true);

        // The trace kernel atomically lowers one entry per active work item to
        // its first unresolved polar index. Reset it to the all-resolved
        // sentinel, then reset the append counter for successor work items.
        auto *first = static_cast<uint32_t *>(unresolved.contents);
        auto *count = static_cast<uint32_t *>(next_count.contents);
        auto *deferred = static_cast<DeferredTileWork *>(deferred_items.contents);
        auto *deferred_total = static_cast<uint32_t *>(deferred_count.contents);
        if (first == nullptr || count == nullptr || deferred == nullptr ||
            deferred_total == nullptr) {
          throw std::runtime_error("Could not map GPU frontier counters");
        }
        std::fill_n(first, active_count, config.num_polar);
        *count = 0U;
        *deferred_total = 0U;
        timer.stop("Frontier bookkeeping");

        // Encode both passes into one ordered command buffer. Metal guarantees
        // that the emission pass sees the trace pass's writes after the first
        // encoder ends; no threadgroup-wide cross-dispatch barrier is needed.
        timer.start_wall("GPU command encoding");
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        command.label = @"GPU tile frontier";
        encoder.label = @"trace_tile_frontier";
        [encoder setComputePipelineState:trace_pipeline];
        [encoder setBuffer:mip_atlas offset:0 atIndex:0];
        [encoder setBuffer:vertex_atlas offset:0 atIndex:1];
        [encoder setBuffer:azimuth_buffer offset:0 atIndex:2];
        [encoder setBuffer:slope_buffer offset:0 atIndex:3];
        [encoder setBuffer:active offset:0 atIndex:4];
        [encoder setBuffer:tile_buffer offset:0 atIndex:5];
        [encoder setBytes:&shared_parameters length:sizeof(shared_parameters) atIndex:6];
        const uint32_t mip_count_u32 = static_cast<uint32_t>(mip_count);
        [encoder setBytes:&mip_count_u32 length:sizeof(mip_count_u32) atIndex:7];
        [encoder setBuffer:distances offset:0 atIndex:8];
        [encoder setBuffer:elevations offset:0 atIndex:9];
        [encoder setBuffer:unresolved offset:0 atIndex:10];
        [encoder dispatchThreads:MTLSizeMake(config.num_polar, active_count, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [encoder endEncoding];

        // One thread per active column converts its first unresolved polar
        // index into an exact successor segment and appends it to `next`.
        encoder = [command computeCommandEncoder];
        encoder.label = @"emit_tile_frontier";
        [encoder setComputePipelineState:emit_pipeline];
        [encoder setBuffer:active offset:0 atIndex:0];
        [encoder setBuffer:tile_buffer offset:0 atIndex:1];
        [encoder setBuffer:azimuth_buffer offset:0 atIndex:2];
        [encoder setBuffer:unresolved offset:0 atIndex:3];
        [encoder setBytes:&shared_parameters length:sizeof(shared_parameters) atIndex:4];
        [encoder setBuffer:resident_hash_buffer offset:0 atIndex:5];
        [encoder setBytes:&resident_hash_slot_count
                   length:sizeof(resident_hash_slot_count)
                  atIndex:6];
        const uint32_t capacity = config.num_azimuth;
        [encoder setBytes:&capacity length:sizeof(capacity) atIndex:7];
        [encoder setBuffer:next offset:0 atIndex:8];
        [encoder setBuffer:next_count offset:0 atIndex:9];
        [encoder setBuffer:deferred_items offset:0 atIndex:10];
        [encoder setBuffer:deferred_count offset:0 atIndex:11];
        [encoder dispatchThreads:MTLSizeMake(active_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        timer.stop("GPU command encoding");

        // This explicit wait lets the CPU read the append counts and choose
        // the next dispatch dimensions. It still overlaps each GPU pass with
        // background loading and mipmap generation.
        timer.start_wall("GPU command wait");
        [command commit];
        [command waitUntilCompleted];
        timer.stop("GPU command wait");
        if (command.status == MTLCommandBufferStatusError) {
          print_error(@"GPU frontier Metal command failed", command.error);
          throw std::runtime_error("GPU frontier Metal command failed");
        }
        timer.add_work("GPU raytrace", 1'000.0 * (command.GPUEndTime - command.GPUStartTime));

        // One active column emits at most one successor. Since the incoming
        // frontier was checked above, both GPU append buffers must fit the
        // same per-azimuth capacity. The kernels guard their writes; these
        // checks turn a counter overflow into a useful host-side failure.
        if (*count > frontier_capacity || *deferred_total > frontier_capacity) {
          throw std::runtime_error("GPU frontier exceeds the azimuth frontier capacity");
        }

        validate_frontier(next, *count, "next frontier");
        resident_successor_work += static_cast<uint64_t>(*count);
        deferred_successor_work += static_cast<uint64_t>(*deferred_total);

        // Preserve every continuation that did not find a resident successor.
        // The host maps it to a source tile and retries it after installation.
        waiting_work.insert(waiting_work.end(), deferred, deferred + *deferred_total);
        validate_waiting_work();

        timer.start_wall("Frontier bookkeeping");
        // The newly appended frontier becomes active on the next iteration;
        // the old buffer is then reused as its empty successor buffer.
        const id<MTLBuffer> temporary = active;
        active = next;
        next = temporary;

        // The next pass may already reference some resident slots. Pin those
        // slots before installing prepared terrain, so LRU eviction cannot
        // overwrite a tile which the imminent GPU dispatch will read.
        const std::vector<uint8_t> pinned_slots = pin_frontier_slots(active, *count, false);
        install_prepared_tiles(pinned_slots);

        // Append deferred continuations whose terrain became resident while
        // the preceding command buffer was executing.
        active_count = activate_waiting_work(active, *count);
        validate_frontier(active, active_count, "activated frontier");
        timer.stop("Frontier bookkeeping");

        // If no resident continuation is ready, wait for a requested tile to
        // complete preparation. There is no GPU work to submit in this case,
        // so an empty pin set makes every currently resident slot evictable.
        while (active_count == 0U && !waiting_work.empty()) {
          std::unique_lock<std::mutex> lock(preparation_mutex);
          timer.start_wall("Tile availability wait");
          prepared_available.wait(lock, [&] {
            return preparation_error != nullptr || !prepared_tiles.empty();
          });
          timer.stop("Tile availability wait");
          lock.unlock();

          const std::vector<uint8_t> no_pinned_slots(atlas_slot_count, 0U);
          install_prepared_tiles(no_pinned_slots);
          active_count = activate_waiting_work(active, active_count);
          validate_frontier(active, active_count, "activated frontier");
        }
      }
    } catch (...) {
      stop_and_join_workers();
      if (capture) {
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
      }
      throw;
    }
    stop_and_join_workers();
    if (capture) {
      [[MTLCaptureManager sharedCaptureManager] stopCapture];
    }
    timer.stop("GPU raytrace");

    // The completed shared output buffers can now be handed directly to the
    // PNG writer. Its elapsed time is a named wall-clock region, separate
    // from terrain preparation and GPU device work.
    timer.start_wall("PNG generation");
    write_colormapped_png(
        "distances.png",
        view_float_buffer(distances, ray_count, "distance output"),
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    write_colormapped_png(
        "elevations.png",
        view_float_buffer(elevations, ray_count, "elevation output"),
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    timer.stop("PNG generation");

    // Report the finite source catalogue separately from the bounded resident
    // cache, so a memory-budget change is visible when rechunk levels vary.
    std::printf(
        "Terrain sources: %u (resident slots %u / cache capacity %llu, preparation workers %u).\n",
        tile_count,
        resident_tile_count,
        static_cast<unsigned long long>(slot_capacity),
        worker_count
    );
    std::printf(
        "  GPU resident successors: %llu, deferred successors: %llu.\n",
        static_cast<unsigned long long>(resident_successor_work),
        static_cast<unsigned long long>(deferred_successor_work)
    );
    std::printf(
        "  Tile requests: %llu, load operations: %llu, unique loads: %llu, reloads: %llu.\n",
        static_cast<unsigned long long>(source_load_requests),
        static_cast<unsigned long long>(source_load_operations),
        static_cast<unsigned long long>(unique_source_loads),
        static_cast<unsigned long long>(source_reloads)
    );
    std::printf(
        "  Atlas installations: %llu, copied: %.3f GiB, evictions: %llu.\n",
        static_cast<unsigned long long>(atlas_installations),
        static_cast<double>(atlas_bytes_copied) / (1024.0 * 1024.0 * 1024.0),
        static_cast<unsigned long long>(cache_evictions)
    );
    timer.print();
  }
}

} // namespace panorama
