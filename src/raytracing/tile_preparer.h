#pragma once

#include "metal_tile.h"
#include "tile_manager.h"
#include "timer.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <exception>
#include <span>
#include <vector>

namespace panorama {

/// Final counters describing background tile preparation work.
struct TilePreparationStatistics {
  /// All scheduler request calls; the two following counters partition this total.
  uint64_t requests;
  /// First request for a source/LOD variant during this preparer's lifetime.
  uint64_t unique_requests;
  /// Repeat request, whether deduplicated or requeued after an eviction.
  uint64_t duplicate_requests;
  uint32_t worker_count;
};

/// Private TileManager implementation which prepares `.ptile` sources without
/// owning atlas or command resources.
///
/// The main thread requests catalogue indices by ray-entry priority. Workers
/// read LOD metadata and open Metal I/O handles in parallel, then place them
/// into a bounded queue. The cache owns GPU slots, payload I/O, fixed-point
/// conversion, mipmap generation, and residency state.
class AsyncTilePreparer {
public:
  /// Construct a stopped preparer with a bounded prepared-tile hand-off queue.
  AsyncTilePreparer(
      id<MTLDevice> device,
      std::span<const TerrainSource> sources,
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

  /// Record a request and queue one selected terrain LOD at its smallest priority.
  void request(uint32_t source_index, uint32_t lod, float priority);

  /// Return and remove one completed prepared source, or no value when none is ready.
  [[nodiscard]] std::optional<PreparedTile> try_take_prepared();

  /// Block until one prepared tile is available or a worker reports an error.
  void wait_for_prepared();

  /// Rethrow the first worker failure, if a worker has encountered one.
  void rethrow_if_failed() const;

  /// Record that the atlas has installed one prepared source/LOD variant.
  void mark_resident(TileVariant variant);

  /// Record that an atlas eviction makes a source/LOD variant eligible for reloading.
  void mark_evicted(TileVariant variant);

  /// Stop workers, wake all waiters, and join every worker thread.
  void stop_and_join();

  /// Return stable preparation counters after workers have stopped.
  [[nodiscard]] TilePreparationStatistics statistics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace panorama
