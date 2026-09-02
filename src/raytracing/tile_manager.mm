#include "tile_manager.h"

#include "metal_tile.h"
#include "tile_preparer.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace panorama {
namespace {

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

struct TileManager::State {
  RaytraceConfig config;
  std::unique_ptr<TerrainCatalogue> catalogue;
  std::unique_ptr<TileGeometry> origin;
  std::vector<uint32_t> lod_by_source;
  float pixel_angle = 0.0F;
  uint32_t observer_source_index = 0U;
  uint32_t mipmap_values = 0U;
  bool trace_quantized = false;
  std::unique_ptr<ResidentTileCache> cache;
  std::unique_ptr<AsyncTilePreparer> preparer;

  void rebuild_lod_plan(float angle) {
    if (!std::isfinite(angle) || angle <= 0.0F) {
      throw std::invalid_argument("Terrain LOD planning requires a positive pixel angle");
    }
    const std::vector<TerrainSource> &source_values = catalogue->sources();
    lod_by_source.resize(source_values.size());
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
};

TileManager::TileManager(const RaytraceConfig &config, float initial_pixel_angle)
    : state_(std::make_unique<State>()) {
  State &state = *state_;
  state.config = config;
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
  if (state.cache != nullptr || device == nil) {
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
  const uint64_t capacity = state.config.tile_cache_size_bytes / tile_bytes;
  if (capacity == 0U || capacity > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Tile-cache byte budget cannot hold a valid terrain atlas");
  }
  const uint32_t slots = static_cast<uint32_t>(
      std::min<uint64_t>(capacity, static_cast<uint64_t>(state.catalogue->sources().size()))
  );
  state.cache = std::make_unique<ResidentTileCache>(
      device,
      state.catalogue->sources(),
      *state.origin,
      state.catalogue->origin().key,
      state.config,
      state.trace_quantized,
      slots,
      timer
  );
  state.preparer = std::make_unique<AsyncTilePreparer>(
      device,
      state.catalogue->sources(),
      slots,
      state.config.max_tile_preparation_workers,
      timer
  );
  state.preparer->start();
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
  state.rebuild_lod_plan(state.pixel_angle);
  if (state.cache != nullptr) {
    state.cache->rebase_observer(observer);
  }
  return true;
}

const TerrainCatalogue &TileManager::catalogue() const { return *state_->catalogue; }
const std::vector<TerrainSource> &TileManager::sources() const {
  return state_->catalogue->sources();
}
const TileGeometry &TileManager::origin_geometry() const { return *state_->origin; }
const std::vector<uint32_t> &TileManager::lod_by_source() const { return state_->lod_by_source; }
uint32_t TileManager::observer_source_index() const { return state_->observer_source_index; }
float TileManager::pixel_angle() const { return state_->pixel_angle; }
uint32_t TileManager::mipmap_value_count() const { return state_->mipmap_values; }
bool TileManager::traces_quantized() const { return state_->trace_quantized; }
uint32_t TileManager::slot_capacity() const { return state_->cache->slot_capacity(); }
ResidentTileCacheBindings TileManager::bindings() const { return state_->cache->bindings(); }

uint32_t TileManager::slot_for_source(uint32_t source_index) const {
  return state_->cache->slot_for_variant({source_index, state_->lod_by_source.at(source_index)});
}
void TileManager::request(uint32_t source_index, float priority) {
  state_->preparer->request(source_index, state_->lod_by_source.at(source_index), priority);
}
std::vector<TerrainTileVariant>
TileManager::install_prepared(std::span<const uint8_t> pinned_slots, Timer &timer) {
  return state_->cache->install_prepared(*state_->preparer, pinned_slots, timer);
}
void TileManager::wait_for_prepared() { state_->preparer->wait_for_prepared(); }
void TileManager::record_slot_use(std::span<const uint32_t> slots) {
  state_->cache->record_slot_use(slots);
}
uint32_t TileManager::ensure_observer_resident(Timer &timer) {
  const uint32_t source = state_->observer_source_index;
  const std::vector<uint8_t> unpinned(slot_capacity(), 0U);
  uint32_t slot = slot_for_source(source);
  while (slot == slot_capacity()) {
    request(source, 0.0F);
    (void)install_prepared(unpinned, timer);
    slot = slot_for_source(source);
    if (slot == slot_capacity()) {
      timer.start_wall("Tile availability wait");
      wait_for_prepared();
      timer.stop("Tile availability wait");
    }
  }
  return slot;
}
void TileManager::stop() {
  if (state_ != nullptr && state_->preparer != nullptr) {
    state_->preparer->stop_and_join();
  }
}
TilePreparationStatistics TileManager::preparation_statistics() const {
  return state_->preparer->statistics();
}
ResidentTileCacheStatistics TileManager::residency_statistics() const {
  return state_->cache->statistics();
}

} // namespace panorama
