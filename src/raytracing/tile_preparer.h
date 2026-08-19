#pragma once

#include "loaded_tile.h"
#include "terrain_catalogue.h"
#include "timer.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace panorama {

/// One source awaiting installation in the resident atlas.
///
/// GeoTIFF sources carry a fully prepared CPU tile. A custom source instead
/// carries a pre-opened Metal file handle but no CPU payload. Opening handles
/// on workers keeps that cost outside synchronous atlas installation.
struct PreparedTile {
  uint32_t source_index;
  std::unique_ptr<LoadedTile> tile;
  id<MTLIOFileHandle> metal_file;
};

/// Final counters describing background tile preparation work.
struct TilePreparationStatistics {
  uint64_t requests;
  uint64_t load_operations;
  uint64_t unique_loads;
  uint64_t reloads;
  uint32_t worker_count;
};

/// Check that a loaded tile's georeferencing agrees with its catalogue key.
void validate_terrain_tile_position(const LoadedTile &tile, TileKey key, const TileGrid &grid);

/// Prepare compatible terrain sources without owning atlas or command resources.
///
/// The main thread requests catalogue indices by ray-entry priority. Workers
/// load, validate, and mipmap GeoTIFFs, then place them into a bounded queue.
/// Custom Metal sources need no CPU decoding, so workers open their Metal I/O
/// handles in parallel. The cache remains solely responsible for GPU slots,
/// Metal I/O, fixed-point conversion, mipmap generation, and residency state.
class AsyncTilePreparer {
public:
  /// Construct a stopped preparer with a bounded prepared-tile hand-off queue.
  AsyncTilePreparer(
      id<MTLDevice> device,
      std::span<const TerrainSource> sources,
      const LoadedTile &origin,
      TileGrid grid,
      uint32_t prepared_capacity,
      uint32_t configured_workers,
      Timer &timer
  );

  AsyncTilePreparer(const AsyncTilePreparer &) = delete;
  AsyncTilePreparer &operator=(const AsyncTilePreparer &) = delete;

  /// Stop and join workers if the owner exits through an exception path.
  ~AsyncTilePreparer();

  /// Start the configured background workers exactly once.
  void start();

  /// Queue a nonresident source, keeping its smallest requested priority.
  void request(uint32_t source_index, float priority);

  /// Return and remove one completed CPU tile, or no value when none is ready.
  [[nodiscard]] std::optional<PreparedTile> try_take_prepared();

  /// Block until one prepared tile is available or a worker reports an error.
  void wait_for_prepared();

  /// Rethrow the first worker failure, if a worker has encountered one.
  void rethrow_if_failed() const;

  /// Record that the atlas has installed one prepared source.
  void mark_resident(uint32_t source_index);

  /// Record that an atlas eviction makes a source eligible for reloading.
  void mark_evicted(uint32_t source_index);

  /// Stop workers, wake all waiters, and join every worker thread.
  void stop_and_join();

  /// Return stable preparation counters after workers have stopped.
  [[nodiscard]] TilePreparationStatistics statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
