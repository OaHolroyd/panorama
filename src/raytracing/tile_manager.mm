#include "tile_manager.h"

#include "metal_tile.h"
#include "tile_manager_state.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace panorama {
namespace {

/// Reduce the prepared-file header to geometry needed throughout a session.
[[nodiscard]] TileGeometry read_tile_geometry(const std::filesystem::path &path) {
  const MetalTileHeader header = read_metal_tile_header(path);
  return {Crs::from_epsg(header.epsg_code),
          header.maximum_elevation,
          header.cell_count,
          header.lower_left_x,
          header.lower_left_y,
          header.cell_size,
          header.level_count};
}

/// Confirm the origin file and catalogue use the same global square grid.
/// Successor lookup derives positions from that grid rather than reopening
/// every file, so accepting a disagreement here would shift later tiles.
void validate_tile_position(const TileGeometry &tile, TileKey key, const TileGrid &grid) {
  const double width = static_cast<double>(tile.cell_count) * tile.cell_size;
  const double expected_x = grid.origin_x + static_cast<double>(key.column) * grid.width;
  const double expected_y = grid.origin_y - static_cast<double>(key.row + 1) * grid.width;
  const double tolerance = 1e-6 * std::max(1.0, grid.width);
  if (!std::isfinite(width) || std::abs(width - grid.width) > tolerance ||
      std::abs(tile.lower_left_x - expected_x) > tolerance ||
      std::abs(tile.lower_left_y - expected_y) > tolerance) {
    throw std::runtime_error("Terrain tile georeferencing disagrees with the configured tile grid");
  }
}

} // namespace

void TileManager::State::rebuild_lod_plan(float angle) {
  if (!std::isfinite(angle) || angle <= 0.0F) {
    throw std::invalid_argument("Terrain LOD planning requires a positive pixel angle");
  }
  const std::vector<TerrainSource> &source_values = catalogue->sources();
  lod_by_source.resize(source_values.size());
  // Select each source once for this observer/view combination. HostFrontier,
  // workers, residency lookup, and Metal metadata then carry the chosen LOD
  // as part of TileVariant instead of independently recomputing policy.
  for (uint32_t source_index = 0U; source_index < static_cast<uint32_t>(source_values.size());
       source_index++) {
    lod_by_source[source_index] = tile_lod(
        catalogue->grid(),
        source_values[source_index].key,
        config.observer,
        static_cast<float>(origin->cell_size),
        angle,
        config.lod_scale,
        source_values[source_index].lod_count
    );
  }
  pixel_angle = angle;
}

void TileManager::State::start_workers(uint32_t configured_workers) {
  if (catalogue->sources().empty() || prepared_capacity == 0U || device == nil ||
      loader_timer == nullptr) {
    throw std::logic_error("TileManager loading resources are unavailable");
  }
  std::lock_guard<std::mutex> lock(loader_mutex);
  if (workers_started) {
    throw std::logic_error("TileManager workers have already started");
  }
  workers_started = true;
  load_states[{0U, 1U}] = TileLoadState::Resident;

  const uint32_t hardware_threads = std::thread::hardware_concurrency();
  const uint32_t available_workers =
      configured_workers == 0U ? std::max(1U, hardware_threads)
                               : std::min(configured_workers, std::max(1U, hardware_threads));
  worker_count =
      std::min(static_cast<uint32_t>(catalogue->sources().size() - 1U), available_workers);
  workers.reserve(worker_count);

  // Workers prepare only control-plane state. Terrain payloads stay on disk
  // until the render thread has selected an evictable destination slot.
  for (uint32_t worker = 0U; worker < worker_count; worker++) {
    workers.emplace_back([this] {
      while (true) {
        TileLoadRequest request = {};
        {
          std::unique_lock<std::mutex> lock(loader_mutex);
          request_available.wait(lock, [&] {
            return stop_requested.load(std::memory_order_relaxed) || !requests.empty();
          });
          if (stop_requested.load(std::memory_order_relaxed)) {
            return;
          }
          request = requests.top();
          requests.pop();
          // Reprioritization uses lazy queue deletion: the lifecycle and best
          // priority maps identify entries superseded by a later request.
          if (load_states.at(request.variant) != TileLoadState::Queued ||
              request.priority != queued_priorities.at(request.variant)) {
            continue;
          }
          load_states[request.variant] = TileLoadState::Loading;
        }

        try {
          // Validate the requested LOD before retaining its file handle. This
          // keeps malformed-file failures on a worker while the coordinating
          // thread receives them through `loader_error`.
          const TerrainSource &source = catalogue->sources()[request.variant.source_index];
          const MetalTileHeader header = read_metal_tile_header(source.path);
          const std::vector<MetalTileLod> lods = read_metal_tile_lods(source.path, header);
          const auto selected =
              std::find_if(lods.begin(), lods.end(), [&](const MetalTileLod &lod) {
                return lod.lod == request.variant.lod;
              });
          if (selected == lods.end()) {
            throw std::out_of_range("Requested terrain LOD is unavailable in the Metal tile");
          }
          loader_timer->start_work("Metal tile open");
          id<MTLIOFileHandle> file = open_metal_tile_file(device, source.path);
          loader_timer->stop("Metal tile open");

          std::unique_lock<std::mutex> lock(loader_mutex);
          // Bound retained handles and metadata by atlas capacity. A worker
          // resumes as soon as installation removes one prepared item.
          prepared_space_available.wait(lock, [&] {
            return stop_requested.load(std::memory_order_relaxed) ||
                   prepared_tiles.size() < prepared_capacity;
          });
          if (stop_requested.load(std::memory_order_relaxed)) {
            return;
          }
          load_states[request.variant] = TileLoadState::Prepared;
          prepared_tiles.push_back({request.variant, file, *selected});
          lock.unlock();
          prepared_available.notify_one();
        } catch (...) {
          // The first worker failure terminates the pool and wakes every host
          // wait so the exception can be rethrown on the render thread.
          std::lock_guard<std::mutex> lock(loader_mutex);
          if (loader_error == nullptr) {
            loader_error = std::current_exception();
          }
          stop_requested.store(true, std::memory_order_relaxed);
          request_available.notify_all();
          prepared_available.notify_all();
          prepared_space_available.notify_all();
          return;
        }
      }
    });
  }
}

void TileManager::State::request_tile(uint32_t source_index, uint32_t lod, float priority) {
  std::lock_guard<std::mutex> lock(loader_mutex);
  if (source_index >= catalogue->sources().size()) {
    throw std::out_of_range("Tile request refers to an unknown source");
  }
  if (lod == 0U || !std::isfinite(priority)) {
    throw std::invalid_argument("Tile request requires a valid LOD and priority");
  }
  const TileVariant variant = {source_index, lod};
  request_count++;
  if (requested_before[variant] == 0U) {
    requested_before[variant] = 1U;
    unique_request_count++;
  } else {
    duplicate_request_count++;
  }
  TileLoadState &state = load_states[variant];
  if (state == TileLoadState::Unrequested) {
    state = TileLoadState::Queued;
    queued_priorities[variant] = priority;
    requests.push({priority, variant});
    request_available.notify_one();
  } else if (state == TileLoadState::Queued && priority < queued_priorities[variant]) {
    // std::priority_queue cannot update an entry in place. Push the improved
    // request and let the worker discard the older entry when it reaches it.
    queued_priorities[variant] = priority;
    requests.push({priority, variant});
    request_available.notify_one();
  }
}

std::optional<PreparedTile> TileManager::State::try_take_prepared() {
  std::lock_guard<std::mutex> lock(loader_mutex);
  if (prepared_tiles.empty()) {
    return std::nullopt;
  }
  PreparedTile tile = std::move(prepared_tiles.front());
  prepared_tiles.pop_front();
  // File handles leave the bounded queue only after the render thread has
  // reserved an atlas destination, making room for one more worker result.
  prepared_space_available.notify_one();
  return tile;
}

void TileManager::State::wait_for_prepared() {
  std::unique_lock<std::mutex> lock(loader_mutex);
  prepared_available.wait(lock, [&] {
    return loader_error != nullptr || !prepared_tiles.empty() ||
           stop_requested.load(std::memory_order_relaxed);
  });
  if (loader_error != nullptr) {
    std::rethrow_exception(loader_error);
  }
}

void TileManager::State::rethrow_if_failed() const {
  std::lock_guard<std::mutex> lock(loader_mutex);
  if (loader_error != nullptr) {
    std::rethrow_exception(loader_error);
  }
}

void TileManager::State::stop_workers() {
  stop_requested.store(true, std::memory_order_relaxed);
  request_available.notify_all();
  prepared_available.notify_all();
  prepared_space_available.notify_all();
  for (std::thread &worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

TileManager::TileManager(const RaytraceConfig &config, float initial_pixel_angle)
    : state_(std::make_unique<State>()) {
  State &state = *state_;
  state.config = config;
  // Discovery places the observer source at index zero and fixes source
  // indices for the lifetime of GPU catalogue hashes and deferred work.
  state.catalogue = std::make_unique<TerrainCatalogue>(TerrainCatalogue::discover(
      config.tile_dir,
      config.observer,
      config.max_distance,
      config.max_tile_count,
      config.allow_observer_fallback
  ));
  state.config.observer = state.catalogue->observer();
  state.origin = std::make_unique<TileGeometry>(read_tile_geometry(state.catalogue->origin().path));
  const MetalTileHeader header = read_metal_tile_header(state.catalogue->origin().path);
  state.trace_quantized =
      state.config.retain_quantized && header.sample_type == MetalTileSampleType::Uint16Decimeters;
  validate_tile_position(*state.origin, state.catalogue->origin().key, state.catalogue->grid());
  const size_t mip_count =
      static_cast<size_t>(metal_tile_mipmap_value_count(state.origin->cell_count));
  if (mip_count > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("Terrain tile mipmap exceeds Metal uint indexing");
  }
  state.mipmap_values = static_cast<uint32_t>(mip_count);
  state.rebuild_lod_plan(initial_pixel_angle);
}

TileManager::~TileManager() { stop(); }

void TileManager::attach_gpu(id<MTLDevice> device, Timer &timer) {
  State &state = *state_;
  if (state.atlas_attached || device == nil) {
    throw std::logic_error("Tile manager GPU residency is already attached or invalid");
  }
  const MetalTileHeader header = read_metal_tile_header(state.catalogue->origin().path);
  QuantizedMetalTileRecordLayout layout = {};
  if (state.trace_quantized) {
    layout = quantized_metal_tile_record_layout(header);
  }
  const size_t vertex_side = static_cast<size_t>(state.origin->cell_count) + 1U;
  const size_t vertex_count = vertex_side * vertex_side;
  const size_t tile_bytes =
      state.trace_quantized
          ? static_cast<size_t>(state.mipmap_values) * sizeof(uint16_t) + layout.stride
          : (static_cast<size_t>(state.mipmap_values) + vertex_count) * sizeof(float);
  // Every slot retains the LOD-1 stride even when it currently contains a
  // coarser variant. That fixed addressing keeps Metal bindings simple and
  // makes the byte budget a direct division.
  const uint64_t capacity = state.config.tile_cache_size_bytes / tile_bytes;
  if (capacity == 0U || capacity > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Tile-cache byte budget cannot hold a valid terrain atlas");
  }
  const uint32_t slots = static_cast<uint32_t>(
      std::min<uint64_t>(capacity, static_cast<uint64_t>(state.catalogue->sources().size()))
  );
  state.attach_atlas(device, slots, timer);
  state.loader_timer = &timer;
  state.prepared_capacity = slots;
  state.start_workers(state.config.max_tile_preparation_workers);
}

void TileManager::set_pixel_angle(float pixel_angle) { state_->rebuild_lod_plan(pixel_angle); }

void TileManager::set_lod_scale(float lod_scale) {
  if (!std::isfinite(lod_scale) || lod_scale < 0.0F) {
    throw std::invalid_argument("Terrain LOD scale must be finite and nonnegative");
  }
  state_->config.lod_scale = lod_scale;
  state_->rebuild_lod_plan(state_->pixel_angle);
}

bool TileManager::relocate_observer(ObserverLocation observer) {
  State &state = *state_;
  const TileKey key = tile_key_at(state.catalogue->grid(), observer.easting, observer.northing);
  const std::optional<uint32_t> source = state.catalogue->find_source(key);
  if (!source.has_value()) {
    return false;
  }
  state.config.observer = observer;
  state.observer_source_index = *source;
  // LOD distance and observer-relative Float32 metadata both change, but the
  // catalogue, resident payload bytes, pipelines, and worker pool remain valid.
  state.rebuild_lod_plan(state.pixel_angle);
  if (state.atlas_attached) {
    state.rebase_observer(observer);
  }
  return true;
}

const TerrainCatalogue &TileManager::catalogue() const { return *state_->catalogue; }
const std::vector<TerrainSource> &TileManager::sources() const {
  return state_->catalogue->sources();
}
const TileGeometry &TileManager::origin_geometry() const { return *state_->origin; }
uint32_t TileManager::lod_for_source(uint32_t source_index) const {
  return state_->lod_by_source.at(source_index);
}
float TileManager::tile_width() const {
  return static_cast<float>(state_->catalogue->grid().width);
}
uint32_t TileManager::observer_source_index() const { return state_->observer_source_index; }
float TileManager::pixel_angle() const { return state_->pixel_angle; }
uint32_t TileManager::mipmap_value_count() const { return state_->mipmap_values; }
bool TileManager::traces_quantized() const { return state_->trace_quantized; }
uint32_t TileManager::slot_capacity() const { return state_->slot_capacity; }
TileManagerBindings TileManager::bindings() const { return state_->bindings(); }

uint32_t TileManager::slot_for_source(uint32_t source_index) const {
  return state_->slot_for_variant({source_index, lod_for_source(source_index)});
}
void TileManager::request(uint32_t source_index, float priority) {
  state_->request_tile(source_index, lod_for_source(source_index), priority);
}
std::vector<TileVariant>
TileManager::install_available(std::span<const uint8_t> pinned_slots, Timer &timer) {
  return state_->install_prepared(pinned_slots, timer);
}
void TileManager::wait_for_available() { state_->wait_for_prepared(); }
void TileManager::record_slot_use(std::span<const uint32_t> slots) {
  state_->record_slot_use(slots);
}
uint32_t TileManager::ensure_observer_resident(Timer &timer) {
  const uint32_t source = state_->observer_source_index;
  const std::vector<uint8_t> unpinned(slot_capacity(), 0U);
  uint32_t slot = slot_for_source(source);
  // The initial GPU frontier cannot be formed without this particular tile.
  // Drive the ordinary async lifecycle synchronously until it is published.
  while (slot == slot_capacity()) {
    request(source, 0.0F);
    (void)install_available(unpinned, timer);
    slot = slot_for_source(source);
    if (slot == slot_capacity()) {
      timer.start_wall("Tile availability wait");
      wait_for_available();
      timer.stop("Tile availability wait");
    }
  }
  return slot;
}
std::optional<float> TileManager::sample_terrain(double easting, double northing) {
  return state_->sample_terrain(easting, northing);
}
void TileManager::stop() {
  if (state_ != nullptr) {
    state_->stop_workers();
  }
}
TileManagerStatistics TileManager::statistics() const { return state_->statistics(); }

} // namespace panorama
