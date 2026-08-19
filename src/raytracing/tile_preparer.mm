#include "tile_preparer.h"

#include "metal_tile.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

namespace panorama {
namespace {

/// Host-side lifecycle for one source in the finite terrain catalogue.
enum class TileLoadState : uint8_t {
  Unrequested,
  Queued,
  Loading,
  Prepared,
  Resident,
};

/// One priority-ordered request to prepare a source tile.
struct TileLoadRequest {
  float priority;
  uint32_t source_index;
};

/// Put smaller ray-entry distances at the front of the request queue.
struct TileLoadRequestGreater {
  /// Return whether `left` should be served after `right`.
  [[nodiscard]] bool operator()(const TileLoadRequest &left, const TileLoadRequest &right) const {
    return left.priority > right.priority;
  }
};

/// Check that a prepared source has the origin tile's atlas layout.
void validate_tile_compatibility(const LoadedTile &tile, const LoadedTile &origin) {
  if (!tile.supports_level_0_collisions || tile.vertices == nullptr) {
    throw std::logic_error("Level-0 multi-tile tracing received a tile without vertices");
  }
  if (tile.crs.id() != origin.crs.id() || tile.size != origin.size || tile.delta != origin.delta ||
      tile.num_levels != origin.num_levels) {
    throw std::runtime_error("Terrain tile is incompatible with the origin tile");
  }
}

} // namespace

/// Check that a loaded tile's georeferencing agrees with its catalogue key.
void validate_terrain_tile_position(const LoadedTile &tile, TileKey key, const TileGrid &grid) {
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

/// Mutable state kept behind the preparer's small public interface.
struct AsyncTilePreparer::State {
  id<MTLDevice> device;
  std::span<const TerrainSource> sources;
  const LoadedTile &origin;
  TileGrid grid;
  size_t expected_mipmap_values;
  size_t expected_vertex_values;
  uint32_t prepared_capacity;
  Timer &timer;

  std::atomic<bool> stop{false};
  mutable std::mutex mutex;
  std::condition_variable request_available;
  std::condition_variable prepared_available;
  std::condition_variable prepared_space_available;
  std::priority_queue<TileLoadRequest, std::vector<TileLoadRequest>, TileLoadRequestGreater>
      requests;
  std::vector<TileLoadState> states;
  std::vector<float> queued_priorities;
  std::vector<uint8_t> requested_before;
  std::deque<PreparedTile> prepared;
  std::vector<std::thread> workers;
  std::exception_ptr error;
  uint64_t request_count = 0U;
  uint64_t unique_request_count = 0U;
  uint64_t duplicate_request_count = 0U;
  uint32_t worker_count = 0U;
  bool started = false;

  /// Initialise immutable preparation inputs before worker threads exist.
  State(
      id<MTLDevice> device_value,
      std::span<const TerrainSource> source_values,
      const LoadedTile &origin_value,
      TileGrid grid_value,
      uint32_t queue_capacity,
      Timer &timer_value
  )
      : device(device_value), sources(source_values), origin(origin_value), grid(grid_value),
        expected_mipmap_values(
            static_cast<size_t>(metal_tile_mipmap_value_count(origin_value.size))
        ),
        expected_vertex_values(
            (static_cast<size_t>(origin_value.size) + 1U) *
            (static_cast<size_t>(origin_value.size) + 1U)
        ),
        prepared_capacity(queue_capacity), timer(timer_value) {}
};

AsyncTilePreparer::AsyncTilePreparer(
    id<MTLDevice> device,
    std::span<const TerrainSource> sources,
    const LoadedTile &origin,
    TileGrid grid,
    uint32_t prepared_capacity,
    uint32_t configured_workers,
    Timer &timer
)
    : state_(
          std::make_unique<State>(
              device,
              sources,
              origin,
              grid,
              prepared_capacity,
              timer
          )
      ) {
  if (sources.empty() || prepared_capacity == 0U) {
    throw std::invalid_argument("Tile preparer requires sources and prepared-tile capacity");
  }

  state_->states.assign(sources.size(), TileLoadState::Unrequested);
  state_->states[0] = TileLoadState::Resident;
  state_->queued_priorities.assign(sources.size(), std::numeric_limits<float>::infinity());
  state_->requested_before.assign(sources.size(), 0U);

  const uint32_t hardware_threads = std::thread::hardware_concurrency();
  const uint32_t available_workers =
      configured_workers == 0U ? std::max(1U, hardware_threads)
                               : std::min(configured_workers, std::max(1U, hardware_threads));
  const uint32_t worker_count =
      std::min(static_cast<uint32_t>(sources.size() - 1U), available_workers);
  state_->worker_count = worker_count;
  state_->workers.reserve(worker_count);
}

AsyncTilePreparer::~AsyncTilePreparer() { stop_and_join(); }

void AsyncTilePreparer::start() {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.started) {
    throw std::logic_error("Tile preparer workers have already started");
  }
  state.started = true;

  for (uint32_t worker = 0U; worker < state.worker_count; worker++) {
    state.workers.emplace_back([this] {
      State &worker_state = *state_;
      while (true) {
        // Sleep until the main scheduler requests terrain. Keeping the mutex
        // only around queue state lets independent GeoTIFF decoding overlap.
        TileLoadRequest request = {};
        {
          std::unique_lock<std::mutex> worker_lock(worker_state.mutex);
          worker_state.request_available.wait(worker_lock, [&] {
            return worker_state.stop.load(std::memory_order_relaxed) ||
                   !worker_state.requests.empty();
          });
          if (worker_state.stop.load(std::memory_order_relaxed)) {
            break;
          }

          request = worker_state.requests.top();
          worker_state.requests.pop();

          // A later request may have lowered this source's priority while its
          // older heap entry waited. Ignore the stale entry; the replacement
          // remains in the queue at the current priority.
          if (worker_state.states[request.source_index] != TileLoadState::Queued ||
              request.priority != worker_state.queued_priorities[request.source_index]) {
            continue;
          }
          worker_state.states[request.source_index] = TileLoadState::Loading;
        }

        try {
          const TerrainSource &source = worker_state.sources[request.source_index];

          // A custom tile's terrain payload already stores vertices in atlas
          // order and contains no mipmap. Keep the payload out of host memory;
          // opening its Metal handle here also keeps work off the scheduler.
          std::unique_ptr<LoadedTile> tile;
          id<MTLIOFileHandle> metal_file;
          if (is_metal_tile_path(source.path)) {
            worker_state.timer.start_work("Metal tile open");
            metal_file = open_metal_tile_file(worker_state.device, source.path);
            worker_state.timer.stop("Metal tile open");
          } else {
            // Load and prepare GeoTIFFs outside the mutex. Timer work time
            // intentionally sums concurrent worker effort, unlike wall time.
            worker_state.timer.start_work("Tile load");
            tile = std::make_unique<LoadedTile>(LoadedTile::load_tif(source.path, true));
            worker_state.timer.stop("Tile load");

            validate_tile_compatibility(*tile, worker_state.origin);
            validate_terrain_tile_position(*tile, source.key, worker_state.grid);

            // The resulting maximum hierarchy has exactly the fixed stride
            // the resident atlas uses for every compatible source.
            worker_state.timer.start_work("Mipmap generation");
            tile->compute_mipmap();
            worker_state.timer.stop("Mipmap generation");
            if (tile->mipmap.size() != worker_state.expected_mipmap_values ||
                tile->vertices->size() != worker_state.expected_vertex_values) {
              throw std::runtime_error("Resident tile does not match the atlas dimensions");
            }
          }

          // Bound the prepared hand-off queue to the atlas capacity. This
          // applies back-pressure instead of preparing sources the cache
          // cannot install while the current frontier is in flight.
          std::unique_lock<std::mutex> worker_lock(worker_state.mutex);
          worker_state.prepared_space_available.wait(worker_lock, [&] {
            return worker_state.stop.load(std::memory_order_relaxed) ||
                   worker_state.prepared.size() < worker_state.prepared_capacity;
          });
          if (worker_state.stop.load(std::memory_order_relaxed)) {
            break;
          }
          worker_state.states[request.source_index] = TileLoadState::Prepared;
          worker_state.prepared.push_back({request.source_index, std::move(tile), metal_file});
          worker_lock.unlock();
          worker_state.prepared_available.notify_one();
        } catch (...) {
          // Exceptions cannot cross a std::thread boundary. Store only the
          // first one, stop peer workers, and wake every possible waiter.
          std::lock_guard<std::mutex> worker_lock(worker_state.mutex);
          if (worker_state.error == nullptr) {
            worker_state.error = std::current_exception();
          }
          worker_state.stop.store(true, std::memory_order_relaxed);
          worker_state.request_available.notify_all();
          worker_state.prepared_available.notify_all();
          worker_state.prepared_space_available.notify_all();
          break;
        }
      }
    });
  }
}

void AsyncTilePreparer::request(uint32_t source_index, float priority) {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (source_index >= state.sources.size()) {
    throw std::out_of_range("Tile preparation request refers to an unknown source");
  }
  state.request_count++;
  if (state.requested_before[source_index] == 0U) {
    state.requested_before[source_index] = 1U;
    state.unique_request_count++;
  } else {
    state.duplicate_request_count++;
  }
  TileLoadState &source_state = state.states[source_index];
  if (source_state == TileLoadState::Unrequested) {
    source_state = TileLoadState::Queued;
    state.queued_priorities[source_index] = priority;
    state.requests.push({priority, source_index});
    state.request_available.notify_one();
  } else if (source_state == TileLoadState::Queued &&
             priority < state.queued_priorities[source_index]) {
    state.queued_priorities[source_index] = priority;
    state.requests.push({priority, source_index});
    state.request_available.notify_one();
  }
}

std::optional<PreparedTile> AsyncTilePreparer::try_take_prepared() {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.prepared.empty()) {
    return std::nullopt;
  }
  PreparedTile tile = std::move(state.prepared.front());
  state.prepared.pop_front();
  state.prepared_space_available.notify_one();
  return tile;
}

void AsyncTilePreparer::wait_for_prepared() {
  State &state = *state_;
  std::unique_lock<std::mutex> lock(state.mutex);
  state.prepared_available.wait(lock, [&] {
    return state.error != nullptr || !state.prepared.empty() ||
           state.stop.load(std::memory_order_relaxed);
  });
  if (state.error != nullptr) {
    std::rethrow_exception(state.error);
  }
}

void AsyncTilePreparer::rethrow_if_failed() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (state_->error != nullptr) {
    std::rethrow_exception(state_->error);
  }
}

void AsyncTilePreparer::mark_resident(uint32_t source_index) {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.states.at(source_index) = TileLoadState::Resident;
}

void AsyncTilePreparer::mark_evicted(uint32_t source_index) {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.states.at(source_index) = TileLoadState::Unrequested;
}

void AsyncTilePreparer::stop_and_join() {
  if (state_ == nullptr) {
    return;
  }
  State &state = *state_;
  state.stop.store(true, std::memory_order_relaxed);
  state.request_available.notify_all();
  state.prepared_available.notify_all();
  state.prepared_space_available.notify_all();
  for (std::thread &worker : state.workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

TilePreparationStatistics AsyncTilePreparer::statistics() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return {
      state_->request_count,
      state_->unique_request_count,
      state_->duplicate_request_count,
      state_->worker_count,
  };
}

} // namespace panorama
