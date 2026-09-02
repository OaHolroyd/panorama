#include "tile_preparer.h"

#include "metal_tile.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

namespace panorama {
namespace {

/// Host-side lifecycle for one independently loadable terrain variant.
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
  TileVariant variant;
};

/// Put smaller ray-entry distances at the front of the request queue.
struct TileLoadRequestGreater {
  /// Return whether `left` should be served after `right`.
  [[nodiscard]] bool operator()(const TileLoadRequest &left, const TileLoadRequest &right) const {
    return left.priority > right.priority;
  }
};

} // namespace

/// Mutable state kept behind the preparer's small public interface.
struct AsyncTilePreparer::State {
  id<MTLDevice> device;
  std::span<const TerrainSource> sources;
  uint32_t prepared_capacity;
  Timer &timer;

  std::atomic<bool> stop{false};
  mutable std::mutex mutex;
  std::condition_variable request_available;
  std::condition_variable prepared_available;
  std::condition_variable prepared_space_available;
  std::priority_queue<TileLoadRequest, std::vector<TileLoadRequest>, TileLoadRequestGreater>
      requests;
  std::map<TileVariant, TileLoadState> states;
  std::map<TileVariant, float> queued_priorities;
  std::map<TileVariant, uint8_t> requested_before;
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
      uint32_t queue_capacity,
      Timer &timer_value
  )
      : device(device_value), sources(source_values), prepared_capacity(queue_capacity),
        timer(timer_value) {}
};

AsyncTilePreparer::AsyncTilePreparer(
    id<MTLDevice> device,
    std::span<const TerrainSource> sources,
    uint32_t prepared_capacity,
    uint32_t configured_workers,
    Timer &timer
)
    : state_(std::make_unique<State>(device, sources, prepared_capacity, timer)) {
  if (sources.empty() || prepared_capacity == 0U) {
    throw std::invalid_argument("Tile preparer requires sources and prepared-tile capacity");
  }

  state_->states[{0U, 1U}] = TileLoadState::Resident;

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
        // only around queue state lets independent metadata reads overlap.
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
          if (worker_state.states.at(request.variant) != TileLoadState::Queued ||
              request.priority != worker_state.queued_priorities.at(request.variant)) {
            continue;
          }
          worker_state.states[request.variant] = TileLoadState::Loading;
        }

        try {
          const TerrainSource &source = worker_state.sources[request.variant.source_index];

          // Payloads remain on disk until the cache assigns a safe atlas slot.
          // Opening the handle here keeps file-system work off the scheduler.
          const MetalTileHeader header = read_metal_tile_header(source.path);
          const std::vector<MetalTileLod> lods = read_metal_tile_lods(source.path, header);
          const auto selected =
              std::find_if(lods.begin(), lods.end(), [&](const MetalTileLod &lod) {
                return lod.lod == request.variant.lod;
              });
          if (selected == lods.end()) {
            throw std::out_of_range("Requested terrain LOD is unavailable in the Metal tile");
          }
          worker_state.timer.start_work("Metal tile open");
          id<MTLIOFileHandle> metal_file = open_metal_tile_file(worker_state.device, source.path);
          worker_state.timer.stop("Metal tile open");

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
          worker_state.states[request.variant] = TileLoadState::Prepared;
          worker_state.prepared.push_back({request.variant, metal_file, *selected});
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

void AsyncTilePreparer::request(uint32_t source_index, uint32_t lod, float priority) {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (source_index >= state.sources.size()) {
    throw std::out_of_range("Tile preparation request refers to an unknown source");
  }
  if (lod == 0U || !std::isfinite(priority)) {
    throw std::invalid_argument("Tile preparation request requires a valid LOD and priority");
  }
  const TileVariant variant = {source_index, lod};
  state.request_count++;
  if (state.requested_before[variant] == 0U) {
    state.requested_before[variant] = 1U;
    state.unique_request_count++;
  } else {
    state.duplicate_request_count++;
  }
  TileLoadState &variant_state = state.states[variant];
  if (variant_state == TileLoadState::Unrequested) {
    variant_state = TileLoadState::Queued;
    state.queued_priorities[variant] = priority;
    state.requests.push({priority, variant});
    state.request_available.notify_one();
  } else if (variant_state == TileLoadState::Queued &&
             priority < state.queued_priorities[variant]) {
    state.queued_priorities[variant] = priority;
    state.requests.push({priority, variant});
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

void AsyncTilePreparer::mark_resident(TileVariant variant) {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.states.at(variant) = TileLoadState::Resident;
}

void AsyncTilePreparer::mark_evicted(TileVariant variant) {
  State &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.states.at(variant) = TileLoadState::Unrequested;
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
