#include "raytrace_setup.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "loaded_tile.h"
#include "png_writer.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <numbers>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace panorama {
namespace {

#ifndef PANORAMA_METALLIB_PATH
#define PANORAMA_METALLIB_PATH "obj/release/panorama.metallib"
#endif

constexpr const char *kMetallibPath = PANORAMA_METALLIB_PATH;

/// One horizontal compass direction with the same two-float layout as Metal's
/// `float2`; x = sin(azimuth) and y = cos(azimuth).
struct HorizontalDirection {
  float x;
  float y;
};

/// Scalar-only host/device ABI for one observer-relative terrain tile.
///
/// This is mirrored exactly by `RaytraceParameters` in panorama.metal.
struct RaytraceParameters {
  float tile_x_min;
  float tile_y_min;
  float cell_size;
  float observer_elevation;
  uint32_t num_levels;
  uint32_t num_cell;
  uint32_t num_azimuth;
  uint32_t num_polar;
  float max_distance;
};

static_assert(sizeof(HorizontalDirection) == 2U * sizeof(float));
static_assert(sizeof(RaytraceParameters) == 9U * sizeof(uint32_t));

/// Per-azimuth input state for one CPU-scheduled tile segment.
///
/// It mirrors `TileRayInput` in panorama.metal and is padded to 16 bytes so
/// the host/device layout is unambiguous.
struct TileRayInput {
  uint32_t first_polar;
  uint32_t start_level;
  float entry_distance;
  uint32_t unused;
};

static_assert(sizeof(TileRayInput) == 4U * sizeof(uint32_t));

/// One immutable tile entry in the stage-2 resident atlas.
///
/// The slot's terrain samples live at `slot * per_tile_*_count` in the two
/// atlas buffers.  This structure contains the per-tile geometry needed to
/// interpret those samples and is mirrored by `ResidentTile` in Metal.
struct ResidentTile {
  RaytraceParameters parameters;
  uint32_t generation;
  uint32_t unused_0;
  uint32_t unused_1;
  uint32_t unused_2;
};

static_assert(sizeof(ResidentTile) == 13U * sizeof(uint32_t));

/// One unresolved azimuth-column segment in the stage-2 GPU frontier.
///
/// It is mirrored by `TileWorkItem` in Metal.  `entry_distance` is exact;
/// only the GPU's cell/tile classification positions are nudged.
struct TileWorkItem {
  uint32_t slot;
  uint32_t azimuth;
  uint32_t first_polar;
  uint32_t start_level;
  float entry_distance;
  uint32_t unused_0;
  uint32_t unused_1;
  uint32_t unused_2;
};

static_assert(sizeof(TileWorkItem) == 8U * sizeof(uint32_t));

/// Terminal result written by Metal for one independent polar ray.
///
/// The CPU combines the monotonic hit/non-hit sequence into a continuation.
enum class RayStatus : uint32_t {
  Inactive = 0U,
  Hit = 1U,
  Continue = 2U,
  MaxDistance = 3U,
};

/// Scheduler outcome for the unresolved suffix of one azimuth column.
enum class TileContinuationStatus : uint32_t {
  Resolved,
  Continue,
  MaxDistance,
};

/// Explicit CPU continuation state for one azimuth column after one tile.
struct TileContinuation {
  uint32_t first_unresolved_polar;
  float entry_distance;
  TileContinuationStatus status;
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

/// Unresolved polar suffix entering one tile in an azimuth column.
struct TileRayState {
  uint32_t first_polar;
  float entry_distance;
};

/// Priority-queue entry ordered by Manhattan shell, row, and column.
struct TileQueueEntry {
  uint64_t shell;
  int64_t row;
  int64_t column;
};

/// Reverse comparison that makes std::priority_queue return the nearest tile.
struct TileQueueEntryGreater {
  /// Return whether `left` has lower scheduling priority than `right`.
  [[nodiscard]] bool operator()(const TileQueueEntry &left, const TileQueueEntry &right) const {
    if (left.shell != right.shell) {
      return left.shell > right.shell;
    }
    if (left.row != right.row) {
      return left.row > right.row;
    }
    return left.column > right.column;
  }
};

/// Reusable directory and filename fragments for prepared tile lookup.
struct TileNameTemplate {
  std::filesystem::path directory;
  std::string prefix;
  std::string suffix;
};

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
  if (!std::isfinite(config.observer.easting) || !std::isfinite(config.observer.northing) ||
      !std::isfinite(config.observer.elevation) || !std::isfinite(config.azimuth_start) ||
      !std::isfinite(config.azimuth_end) || !std::isfinite(config.polar_start) ||
      !std::isfinite(config.polar_end) || !std::isfinite(config.max_distance) ||
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

/// Extract the row/column key and reusable name template from a prepared tile path.
[[nodiscard]] std::pair<TileKey, TileNameTemplate>
parse_tile_name(const std::filesystem::path &path) {
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
  return {key, {path.parent_path(), name.substr(0, row_marker), name.substr(extension)}};
}

/// Return the prepared GeoTIFF path corresponding to one global tile key.
[[nodiscard]] std::filesystem::path tile_path(const TileNameTemplate &name_template, TileKey key) {
  return name_template.directory / (name_template.prefix + "_r" + std::to_string(key.row) + "_c" +
                                    std::to_string(key.column) + name_template.suffix);
}

/// Derive the global tile-grid origin from one loaded tile and its parsed key.
[[nodiscard]] TileGrid make_tile_grid(const LoadedTile &tile, TileKey key) {
  const double width = static_cast<double>(tile.size) * tile.delta;
  if (!std::isfinite(width) || width <= 0.0) {
    throw std::invalid_argument("Terrain tile has an invalid physical width");
  }
  // Rechunked rows count southward from their north-edge grid origin, whereas
  // LoadedTile stores a south-west corner. A row therefore begins one width
  // below origin_y - row * width.
  return {
      tile.lower_left_x - static_cast<double>(key.column) * width,
      tile.lower_left_y + static_cast<double>(key.row + 1) * width,
      width,
  };
}

/// Return the global tile key that owns a point after its boundary nudge.
[[nodiscard]] TileKey tile_key_at(const TileGrid &grid, double x, double y) {
  const double column = std::floor((x - grid.origin_x) / grid.width);
  const double row = std::floor((grid.origin_y - y) / grid.width);
  if (column < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      column > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
      row < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      row > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::out_of_range("Ray exit is outside the supported tile grid");
  }
  return {static_cast<int64_t>(row), static_cast<int64_t>(column)};
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
  const double expected_x = grid.origin_x + static_cast<double>(key.column) * grid.width;
  const double expected_y = grid.origin_y - static_cast<double>(key.row + 1) * grid.width;
  const double tolerance = 1e-6 * std::max(1.0, grid.width);
  if (std::abs(tile.lower_left_x - expected_x) > tolerance ||
      std::abs(tile.lower_left_y - expected_y) > tolerance) {
    throw std::runtime_error("Terrain tile georeferencing disagrees with its filename key");
  }
}

/// Build the scalar Metal parameters for one observer-relative terrain tile.
[[nodiscard]] RaytraceParameters
make_raytrace_parameters(const LoadedTile &tile, const RaytraceConfig &config) {
  const double local_x = tile.lower_left_x - config.observer.easting;
  const double local_y = tile.lower_left_y - config.observer.northing;
  if (local_x < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_x > static_cast<double>(std::numeric_limits<float>::max()) ||
      local_y < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      local_y > static_cast<double>(std::numeric_limits<float>::max()) ||
      tile.delta > static_cast<double>(std::numeric_limits<float>::max()) ||
      config.observer.elevation < static_cast<double>(std::numeric_limits<float>::lowest()) ||
      config.observer.elevation > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::overflow_error("Raytrace geometry does not fit float32");
  }
  return {
      static_cast<float>(local_x),
      static_cast<float>(local_y),
      static_cast<float>(tile.delta),
      static_cast<float>(config.observer.elevation),
      tile.num_levels,
      tile.size,
      config.num_azimuth,
      config.num_polar,
      config.max_distance,
  };
}

/// Derive one explicit column continuation from the completed per-ray statuses.
[[nodiscard]] TileContinuation collect_continuation(
    std::span<const uint32_t> statuses,
    uint32_t azimuth,
    uint32_t num_azimuth,
    uint32_t first_polar,
    uint32_t num_polar,
    float tile_exit
) {
  bool seen_non_hit = false;
  for (uint32_t polar = first_polar; polar < num_polar; ++polar) {
    const size_t output_index = static_cast<size_t>(polar) * num_azimuth + azimuth;
    const RayStatus status = static_cast<RayStatus>(statuses[output_index]);
    if (status == RayStatus::Hit) {
      if (seen_non_hit) {
        throw std::runtime_error("Tile rays violate the required hit-prefix ordering");
      }
      continue;
    }
    if (status == RayStatus::Continue) {
      seen_non_hit = true;
      return {polar, tile_exit, TileContinuationStatus::Continue};
    }
    if (status == RayStatus::MaxDistance) {
      seen_non_hit = true;
      return {polar, tile_exit, TileContinuationStatus::MaxDistance};
    }
    throw std::runtime_error("Tile kernel left an active ray without a status");
  }
  return {num_polar, tile_exit, TileContinuationStatus::Resolved};
}

/// Construct float32 compass directions from evenly spaced azimuth centres.
[[nodiscard]] std::vector<HorizontalDirection>
make_azimuth_directions(const RaytraceConfig &config) {
  std::vector<HorizontalDirection> directions(config.num_azimuth);
  const double azimuth_step =
      (config.azimuth_end - config.azimuth_start) / static_cast<double>(config.num_azimuth);
  for (uint32_t index = 0; index < config.num_azimuth; ++index) {
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
  for (uint32_t index = 0; index < config.num_polar; ++index) {
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
/// If `data == nullptr` then allocate a new buffer otherwise link it to the
/// supplied data.
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

/// Return the exact far boundary of one tile segment along a horizontal ray.
[[nodiscard]] float tile_exit_distance(
    const RaytraceParameters &parameters,
    HorizontalDirection direction,
    float entry_distance
) {
  float tile_exit = INFINITY;
  const float x_max = parameters.tile_x_min + float(parameters.num_cell) * parameters.cell_size;
  const float y_max = parameters.tile_y_min + float(parameters.num_cell) * parameters.cell_size;
  if (direction.x > 0.0F) {
    const float candidate = x_max / direction.x;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  } else if (direction.x < 0.0F) {
    const float candidate = parameters.tile_x_min / direction.x;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  }
  if (direction.y > 0.0F) {
    const float candidate = y_max / direction.y;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  } else if (direction.y < 0.0F) {
    const float candidate = parameters.tile_y_min / direction.y;
    if (candidate > entry_distance) {
      tile_exit = std::min(tile_exit, candidate);
    }
  }
  if (!std::isfinite(tile_exit)) {
    throw std::runtime_error("Ray cannot leave the current terrain tile");
  }
  return tile_exit;
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

/// Return every prepared tile which can be reached before the trace range.
///
/// The initial stage-2 implementation deliberately preloads this conservative
/// set.  A later asynchronous cache will replace directory scanning with tile
/// requests and a bounded resident subset.
[[nodiscard]] std::vector<std::pair<TileKey, std::filesystem::path>> find_resident_tiles(
    const TileNameTemplate &name_template,
    TileKey origin_key,
    const TileGrid &grid,
    const RaytraceConfig &config
) {
  std::vector<std::pair<TileKey, std::filesystem::path>> tiles;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(name_template.directory)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".tif") {
      continue;
    }
    TileKey key;
    try {
      key = parse_tile_name(entry.path()).first;
    } catch (const std::invalid_argument &) {
      // The prepared-tile directory can contain unrelated GeoTIFFs.
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

/// Run the retained synchronous scheduler used as the stage-1 reference.
void perform_stage_1_multi_tile_raytrace(const RaytraceConfig &config) {
  validate_configuration(config);
  if (config.tile_path.empty()) {
    throw std::invalid_argument("Raytrace tile path must not be empty");
  }
  // This wall-clock interval covers all stage-1 CPU and GPU tracing work, but
  // deliberately excludes writing diagnostic PNGs below.
  const auto raytrace_start = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> cpu_tile_load_duration =
      std::chrono::duration<double, std::milli>::zero();
  std::chrono::duration<double, std::milli> cpu_mipmap_duration =
      std::chrono::duration<double, std::milli>::zero();

  // The origin tile establishes the shared layout and the global filename
  // grid. Neighbours are loaded only when an unresolved column reaches them.
  const auto origin_load_start = std::chrono::steady_clock::now();
  LoadedTile origin_tile = LoadedTile::load_tif(config.tile_path, true);
  validate_tile_compatibility(origin_tile, origin_tile);
  cpu_tile_load_duration += std::chrono::steady_clock::now() - origin_load_start;
  const auto origin_mipmap_start = std::chrono::steady_clock::now();
  origin_tile.compute_mipmap();
  cpu_mipmap_duration += std::chrono::steady_clock::now() - origin_mipmap_start;
  const auto [origin_key, name_template] = parse_tile_name(config.tile_path);
  const TileGrid grid = make_tile_grid(origin_tile, origin_key);
  validate_tile_position(origin_tile, origin_key, grid);

  const std::vector<HorizontalDirection> directions = make_azimuth_directions(config);
  const std::vector<float> polar_slopes = make_polar_slopes(config);
  const size_t num_ray = static_cast<size_t>(config.num_azimuth) * config.num_polar;
  const size_t input_bytes =
      checked_buffer_length(config.num_azimuth, sizeof(TileRayInput), "tile ray input");

  // Every azimuth starts at the observer tile with its entire polar column
  // unresolved. The priority key is Manhattan distance from that tile.
  std::map<TileKey, std::map<uint32_t, TileRayState>> pending;
  std::map<uint32_t, TileRayState> initial;
  for (uint32_t azimuth = 0; azimuth < config.num_azimuth; ++azimuth) {
    initial.emplace(azimuth, TileRayState{0U, 0.0F});
  }
  pending.emplace(origin_key, std::move(initial));
  std::priority_queue<TileQueueEntry, std::vector<TileQueueEntry>, TileQueueEntryGreater> queue;
  queue.push({0U, origin_key.row, origin_key.column});

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("No Metal device is available");
    }
    NSError *error = nil;
    NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      throw std::runtime_error("Could not load the Metal library");
    }
    id<MTLFunction> kernel = [library newFunctionWithName:@"trace_tile_segment"];
    if (kernel == nil) {
      throw std::runtime_error("Kernel trace_tile_segment is missing");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel
                                                                                 error:&error];
    if (pipeline == nil) {
      print_error(@"Could not create the multi-tile raytrace pipeline", error);
      throw std::runtime_error("Could not create the multi-tile raytrace pipeline");
    }
    id<MTLBuffer> azimuths = make_buffer(
        device,
        directions.data(),
        checked_buffer_length(directions.size(), sizeof(HorizontalDirection), "azimuth directions"),
        "azimuth directions"
    );
    id<MTLBuffer> slopes = make_buffer(
        device,
        polar_slopes.data(),
        checked_buffer_length(polar_slopes.size(), sizeof(float), "polar slopes"),
        "polar slopes"
    );
    id<MTLBuffer> distances = make_buffer(
        device,
        nullptr,
        checked_buffer_length(num_ray, sizeof(float), "distance output"),
        "distance output"
    );
    id<MTLBuffer> elevations = make_buffer(
        device,
        nullptr,
        checked_buffer_length(num_ray, sizeof(float), "elevation output"),
        "elevation output"
    );
    clear_buffer(distances, "distance output");
    clear_buffer(elevations, "elevation output");

    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    if (command_queue == nil) {
      throw std::runtime_error("Could not create a Metal multi-tile command queue");
    }
    const bool capture_active = start_capture_if_requested(command_queue);
    uint32_t processed_tiles = 0U;
    double total_gpu_milliseconds = 0.0;

    try {
      // Keep raytracing tile-by-tile until the queue has no tiles left
      while (!queue.empty()) {
        // Get the next tile from the queue
        const TileQueueEntry queue_entry = queue.top();
        queue.pop();
        const TileKey key = {queue_entry.row, queue_entry.column};

        // find out which unresolved rays pass through this tile and need to be
        // assigned a thread in the next Metal kernel call
        auto pending_iterator = pending.find(key);
        if (pending_iterator == pending.end()) {
          continue;
        }
        std::map<uint32_t, TileRayState> tile_work = std::move(pending_iterator->second);
        pending.erase(pending_iterator);
        // A zero limit means unlimited. Stop before loading or dispatching
        // another tile, so a limit of one traces exactly the observer tile.
        if (config.max_tile_count != 0U && processed_tiles >= config.max_tile_count) {
          break;
        }

        // If the next tile is not the origin, load it from disk
        std::unique_ptr<LoadedTile> neighbour;
        const LoadedTile *tile = &origin_tile;
        if (!(key == origin_key)) {
          const std::filesystem::path path = tile_path(name_template, key);

          if (!std::filesystem::is_regular_file(path)) {
            // Prepared chunks are an availability boundary. Their absence is
            // open coverage, so every incoming suffix terminates as sky.
            // TODO: this is incorrect, sometimes missing tiles are Ocean
            continue;
          }

          // load the tile and prepare the mipmap
          std::printf(
              "Loading tile r=%lld c=%lld: %s\n",
              static_cast<long long>(key.row),
              static_cast<long long>(key.column),
              path.c_str()
          );
          const auto tile_load_start = std::chrono::steady_clock::now();
          neighbour = std::make_unique<LoadedTile>(LoadedTile::load_tif(path, true));
          validate_tile_compatibility(*neighbour, origin_tile);
          validate_tile_position(*neighbour, key, grid);
          cpu_tile_load_duration += std::chrono::steady_clock::now() - tile_load_start;
          const auto tile_mipmap_start = std::chrono::steady_clock::now();
          neighbour->compute_mipmap();
          cpu_mipmap_duration += std::chrono::steady_clock::now() - tile_mipmap_start;
          tile = neighbour.get();
        }
        processed_tiles++;

        // set up the raytracing parameters for this tile
        const RaytraceParameters parameters = make_raytrace_parameters(*tile, config);

        // set up the ray-columns for this tile
        std::vector<TileRayInput> inputs(config.num_azimuth, {config.num_polar, 1U, 0.0F, 0U});
        for (const auto &[azimuth, state] : tile_work) {
          inputs[azimuth] = {
              state.first_polar,
              key == origin_key ? 1U : tile->num_levels,
              state.entry_distance,
              0U,
          };
        }

        // set up tile-specific buffers for the Metal kernel
        id<MTLBuffer> mipmap = make_buffer(
            device,
            tile->mipmap.data(),
            checked_buffer_length(tile->mipmap.size(), sizeof(float), "maximum mipmap"),
            "maximum mipmap"
        );
        id<MTLBuffer> vertices = make_buffer(
            device,
            tile->vertices->data(),
            checked_buffer_length(tile->vertices->size(), sizeof(float), "level-0 vertex heights"),
            "level-0 vertex heights"
        );
        id<MTLBuffer> input_buffer =
            make_buffer(device, inputs.data(), input_bytes, "tile ray input");
        id<MTLBuffer> statuses = make_buffer(
            device,
            nullptr,
            checked_buffer_length(num_ray, sizeof(uint32_t), "ray status output"),
            "ray status output"
        );
        clear_buffer(statuses, "ray status output");

        // Set up the GPU command sequence
        id<MTLCommandBuffer> command = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
          throw std::runtime_error("Could not create a Metal multi-tile command");
        }
        command.label = @"stage-1 multi-tile raytrace";
        encoder.label = @"trace_tile_segment";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:mipmap offset:0 atIndex:0];
        [encoder setBuffer:vertices offset:0 atIndex:1];
        [encoder setBuffer:azimuths offset:0 atIndex:2];
        [encoder setBuffer:slopes offset:0 atIndex:3];
        [encoder setBuffer:input_buffer offset:0 atIndex:4];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
        [encoder setBuffer:distances offset:0 atIndex:6];
        [encoder setBuffer:elevations offset:0 atIndex:7];
        [encoder setBuffer:statuses offset:0 atIndex:8];
        [encoder dispatchThreads:MTLSizeMake(config.num_polar, config.num_azimuth, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [encoder endEncoding];

        // Start raytracing this tile
        [command commit];
        // NOTE: This explicit wait is the defining deliberate limitation of stage 1.
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) {
          print_error(@"The Metal multi-tile raytrace command failed", command.error);
          throw std::runtime_error("The Metal multi-tile raytrace command failed");
        }
        // Each stage-1 tile uses its own command buffer. Sum their device
        // execution intervals so the reported time excludes synchronous CPU
        // tile loading, buffer allocation, and continuation scheduling.
        total_gpu_milliseconds += 1'000.0 * (command.GPUEndTime - command.GPUStartTime);

        // Unpack the statuses of all of the rays
        const std::span<const uint32_t> status_values(
            static_cast<const uint32_t *>(statuses.contents),
            num_ray
        );

        // go through all of the ray-columns that entered the tile and decide what to
        // do with them
        for (const auto &[azimuth, state] : tile_work) {
          const float tile_exit =
              tile_exit_distance(parameters, directions[azimuth], state.entry_distance);

          // Go through the exist statuses of all of the rays in a column and decide what that means
          // about the continuation status of the column as a whole
          const TileContinuation continuation = collect_continuation(
              status_values,
              azimuth,
              config.num_azimuth,
              state.first_polar,
              config.num_polar,
              tile_exit
          );

          // Stop columns which have completed or hit max distance
          if (continuation.status != TileContinuationStatus::Continue ||
              continuation.entry_distance >= config.max_distance) {
            continue;
          }

          // Work out where the ray-column left the tile (and thus where it enters the new one)
          float exit_x = continuation.entry_distance * directions[azimuth].x;
          float exit_y = continuation.entry_distance * directions[azimuth].y;

          // Find the tile that it will progress into
          // Use the exact exit distance for geometry but a small forward nudge
          // for tile ownership. This is the same separation as the Python
          // reference and prevents return-to-the-tile rounding at a seam.
          const float nudge = std::max(
              1e-3F * parameters.cell_size,
              8.0F * std::numeric_limits<float>::epsilon() *
                  std::max(1.0F, std::max(std::abs(exit_x), std::abs(exit_y)))
          );
          if (directions[azimuth].x != 0.0F) {
            exit_x += std::copysign(nudge, directions[azimuth].x);
          }
          if (directions[azimuth].y != 0.0F) {
            exit_y += std::copysign(nudge, directions[azimuth].y);
          }
          const TileKey next_key = tile_key_at(
              grid,
              config.observer.easting + static_cast<double>(exit_x),
              config.observer.northing + static_cast<double>(exit_y)
          );

          if (next_key == key) {
            throw std::runtime_error("Ray did not advance to a neighbouring terrain tile");
          }

          // Add this ray-column to the pending-work map, creating one if this is the first azimuth
          // to reach a given tile
          auto [next_iterator, inserted] = pending.try_emplace(next_key);
          const auto column_inserted =
              next_iterator->second
                  .emplace(
                      azimuth,
                      TileRayState{continuation.first_unresolved_polar, continuation.entry_distance}
                  )
                  .second; // we don't care about the returned iterator

          if (!column_inserted) {
            throw std::runtime_error("An azimuth column entered a tile more than once");
          }

          // Add the next tile to the queue if nothing is wrong and this azimuth was the first to
          // request it (we must avoid repeat requests)
          if (inserted) {
            const uint64_t next_shell =
                static_cast<uint64_t>(std::llabs(next_key.row - origin_key.row)) +
                static_cast<uint64_t>(std::llabs(next_key.column - origin_key.column));
            queue.push({next_shell, next_key.row, next_key.column});
          }
        }
      }
    } catch (...) {
      if (capture_active) {
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
      }
      throw;
    }
    if (capture_active) {
      [[MTLCaptureManager sharedCaptureManager] stopCapture];
    }
    std::printf("Processed %u terrain tiles.\n", processed_tiles);
    std::printf("  GPU raytrace   : %8.3f ms\n", total_gpu_milliseconds);
    std::printf("  CPU tile load  : %8.3f ms\n", cpu_tile_load_duration.count());
    std::printf("  CPU mipmaps    : %8.3f ms\n", cpu_mipmap_duration.count());
    const std::chrono::duration<double, std::milli> raytrace_duration =
        std::chrono::steady_clock::now() - raytrace_start;
    std::printf("  CPU + GPU total: %8.3f ms\n", raytrace_duration.count());

    // Write the data to the output files
    const auto image_start = std::chrono::steady_clock::now();
    write_colormapped_png(
        "distances.png",
        view_float_buffer(distances, num_ray, "distance output"),
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    write_colormapped_png(
        "elevations.png",
        view_float_buffer(elevations, num_ray, "elevation output"),
        config.num_azimuth,
        config.num_polar,
        colormaps::viridis
    );
    const std::chrono::duration<double, std::milli> image_duration =
        std::chrono::steady_clock::now() - image_start;
    std::printf("  PNG generation : %8.3f ms\n", image_duration.count());
  }
}

/// Trace an all-resident, GPU-scheduled tile frontier.
///
/// This is stage 2's correctness implementation: it preloads the finite
/// scene, then only the GPU creates continuation work items.  Cache misses,
/// eviction, and asynchronous loading deliberately remain a later extension.
void perform_multi_tile_raytrace(const RaytraceConfig &config) {
  validate_configuration(config);
  const auto started = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> tile_load =
      std::chrono::duration<double, std::milli>::zero();
  std::chrono::duration<double, std::milli> mipmap_generation =
      std::chrono::duration<double, std::milli>::zero();

  const auto origin_load_started = std::chrono::steady_clock::now();
  LoadedTile origin = LoadedTile::load_tif(config.tile_path, true);
  tile_load += std::chrono::steady_clock::now() - origin_load_started;
  const auto origin_mipmap_started = std::chrono::steady_clock::now();
  origin.compute_mipmap();
  mipmap_generation += std::chrono::steady_clock::now() - origin_mipmap_started;
  const auto [origin_key, names] = parse_tile_name(config.tile_path);
  const TileGrid grid = make_tile_grid(origin, origin_key);
  validate_tile_position(origin, origin_key, grid);
  std::vector<std::pair<TileKey, std::filesystem::path>> paths =
      find_resident_tiles(names, origin_key, grid, config);
  if (config.max_tile_count != 0U && paths.size() > config.max_tile_count) {
    paths.resize(config.max_tile_count);
  }

  const size_t mip_count = origin.mipmap.size();
  const size_t vertex_count = origin.vertices->size();
  const size_t tile_bytes =
      checked_buffer_length(mip_count + vertex_count, sizeof(float), "terrain tile");
  const uint64_t slot_capacity = config.tile_cache_size_bytes / tile_bytes;
  if (slot_capacity == 0U || paths.size() > slot_capacity) {
    throw std::runtime_error("Tile-cache byte budget cannot hold the all-resident stage-2 scene");
  }
  const uint32_t tile_count = static_cast<uint32_t>(paths.size());
  const size_t work_capacity = static_cast<size_t>(tile_count) * config.num_azimuth;
  if (work_capacity > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("GPU frontier has too many work items");
  }

  const std::vector<HorizontalDirection> directions = make_azimuth_directions(config);
  const std::vector<float> slopes = make_polar_slopes(config);
  const size_t ray_count = static_cast<size_t>(config.num_azimuth) * config.num_polar;

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil)
      throw std::runtime_error("No Metal device is available");
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:kMetallibPath]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil) {
      print_error(@"Could not load the Metal library", error);
      throw std::runtime_error("Could not load Metal library");
    }
    id<MTLFunction> trace = [library newFunctionWithName:@"trace_tile_frontier"];
    id<MTLFunction> emit = [library newFunctionWithName:@"emit_tile_frontier"];
    if (trace == nil || emit == nil)
      throw std::runtime_error("Stage-2 Metal kernels are missing");
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

    id<MTLBuffer> mip_atlas = make_buffer(
        device,
        nullptr,
        checked_buffer_length(tile_count * mip_count, sizeof(float), "mipmap atlas"),
        "mipmap atlas"
    );
    id<MTLBuffer> vertex_atlas = make_buffer(
        device,
        nullptr,
        checked_buffer_length(tile_count * vertex_count, sizeof(float), "vertex atlas"),
        "vertex atlas"
    );
    auto *mips = static_cast<float *>(mip_atlas.contents);
    auto *vertices = static_cast<float *>(vertex_atlas.contents);
    if (mips == nullptr || vertices == nullptr)
      throw std::runtime_error("Could not map terrain atlas");
    std::vector<ResidentTile> metadata(tile_count);
    for (uint32_t slot = 0U; slot < tile_count; ++slot) {
      const auto &[key, path] = paths[slot];
      std::unique_ptr<LoadedTile> loaded_neighbour;
      const LoadedTile *neighbour = &origin;
      if (slot != 0U) {
        const auto neighbour_load_started = std::chrono::steady_clock::now();
        loaded_neighbour = std::make_unique<LoadedTile>(LoadedTile::load_tif(path, true));
        tile_load += std::chrono::steady_clock::now() - neighbour_load_started;
        const auto neighbour_mipmap_started = std::chrono::steady_clock::now();
        loaded_neighbour->compute_mipmap();
        mipmap_generation += std::chrono::steady_clock::now() - neighbour_mipmap_started;
        neighbour = loaded_neighbour.get();
      }
      validate_tile_compatibility(*neighbour, origin);
      validate_tile_position(*neighbour, key, grid);
      if (neighbour->mipmap.size() != mip_count || neighbour->vertices->size() != vertex_count) {
        throw std::runtime_error("Resident tile does not match the atlas dimensions");
      }
      std::memcpy(
          mips + static_cast<size_t>(slot) * mip_count,
          neighbour->mipmap.data(),
          mip_count * sizeof(float)
      );
      std::memcpy(
          vertices + static_cast<size_t>(slot) * vertex_count,
          neighbour->vertices->data(),
          vertex_count * sizeof(float)
      );
      metadata[slot] = {make_raytrace_parameters(*neighbour, config), 1U, 0U, 0U, 0U};
    }
    id<MTLBuffer> tile_buffer = make_buffer(
        device,
        metadata.data(),
        checked_buffer_length(tile_count, sizeof(ResidentTile), "tile metadata"),
        "tile metadata"
    );
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
    id<MTLBuffer> work_a = make_buffer(
        device,
        nullptr,
        checked_buffer_length(work_capacity, sizeof(TileWorkItem), "active frontier"),
        "active frontier"
    );
    id<MTLBuffer> work_b = make_buffer(
        device,
        nullptr,
        checked_buffer_length(work_capacity, sizeof(TileWorkItem), "next frontier"),
        "next frontier"
    );
    id<MTLBuffer> unresolved = make_buffer(
        device,
        nullptr,
        checked_buffer_length(work_capacity, sizeof(uint32_t), "unresolved polar indices"),
        "unresolved polar indices"
    );
    id<MTLBuffer> next_count =
        make_buffer(device, nullptr, sizeof(uint32_t), "next frontier count");
    auto *initial = static_cast<TileWorkItem *>(work_a.contents);
    for (uint32_t azimuth = 0U; azimuth < config.num_azimuth; ++azimuth)
      initial[azimuth] = {0U, azimuth, 0U, 1U, 0.0F, 0U, 0U, 0U};

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil)
      throw std::runtime_error("Could not create Metal command queue");
    const bool capture = start_capture_if_requested(queue);
    double gpu_ms = 0.0;
    uint32_t active_count = config.num_azimuth;
    id<MTLBuffer> active = work_a, next = work_b;
    try {
      while (active_count != 0U) {
        auto *first = static_cast<uint32_t *>(unresolved.contents);
        auto *count = static_cast<uint32_t *>(next_count.contents);
        if (first == nullptr || count == nullptr)
          throw std::runtime_error("Could not map GPU frontier counters");
        std::fill_n(first, active_count, config.num_polar);
        *count = 0U;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        command.label = @"stage-2 GPU tile frontier";
        encoder.label = @"trace_tile_frontier";
        [encoder setComputePipelineState:trace_pipeline];
        [encoder setBuffer:mip_atlas offset:0 atIndex:0];
        [encoder setBuffer:vertex_atlas offset:0 atIndex:1];
        [encoder setBuffer:azimuth_buffer offset:0 atIndex:2];
        [encoder setBuffer:slope_buffer offset:0 atIndex:3];
        [encoder setBuffer:active offset:0 atIndex:4];
        [encoder setBuffer:tile_buffer offset:0 atIndex:5];
        const uint32_t mip_count_u32 = static_cast<uint32_t>(mip_count);
        [encoder setBytes:&mip_count_u32 length:sizeof(mip_count_u32) atIndex:6];
        [encoder setBuffer:distances offset:0 atIndex:7];
        [encoder setBuffer:elevations offset:0 atIndex:8];
        [encoder setBuffer:unresolved offset:0 atIndex:9];
        [encoder dispatchThreads:MTLSizeMake(config.num_polar, active_count, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        encoder.label = @"emit_tile_frontier";
        [encoder setComputePipelineState:emit_pipeline];
        [encoder setBuffer:active offset:0 atIndex:0];
        [encoder setBuffer:tile_buffer offset:0 atIndex:1];
        [encoder setBuffer:azimuth_buffer offset:0 atIndex:2];
        [encoder setBuffer:unresolved offset:0 atIndex:3];
        [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:4];
        const uint32_t capacity = static_cast<uint32_t>(work_capacity);
        [encoder setBytes:&capacity length:sizeof(capacity) atIndex:5];
        [encoder setBuffer:next offset:0 atIndex:6];
        [encoder setBuffer:next_count offset:0 atIndex:7];
        [encoder dispatchThreads:MTLSizeMake(active_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) {
          print_error(@"Stage-2 Metal command failed", command.error);
          throw std::runtime_error("Stage-2 Metal command failed");
        }
        gpu_ms += 1'000.0 * (command.GPUEndTime - command.GPUStartTime);
        if (*count > work_capacity)
          throw std::runtime_error("GPU frontier exceeded its work-item capacity");
        active_count = *count;
        const id<MTLBuffer> temporary = active;
        active = next;
        next = temporary;
      }
    } catch (...) {
      if (capture)
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
      throw;
    }
    if (capture)
      [[MTLCaptureManager sharedCaptureManager] stopCapture];
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - started;
    std::printf(
        "Resident terrain tiles: %u (cache capacity %llu).\n",
        tile_count,
        static_cast<unsigned long long>(slot_capacity)
    );
    std::printf("  GPU raytrace   : %8.3f ms\n", gpu_ms);
    std::printf("  CPU tile load  : %8.3f ms\n", tile_load.count());
    std::printf("  CPU mipmaps    : %8.3f ms\n", mipmap_generation.count());
    std::printf("  CPU + GPU total: %8.3f ms\n", elapsed.count());
    const auto image_started = std::chrono::steady_clock::now();
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
    const std::chrono::duration<double, std::milli> image_elapsed =
        std::chrono::steady_clock::now() - image_started;
    std::printf("  PNG generation : %8.3f ms\n", image_elapsed.count());
  }
}

} // namespace panorama
