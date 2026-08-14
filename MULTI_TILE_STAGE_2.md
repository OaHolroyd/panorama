# Multi-tile ray tracing: stage 2 — GPU wavefront and asynchronous tile cache

## Goal

Move ray-continuation scheduling from the CPU to the GPU while keeping terrain
I/O, tile discovery, decoding, upload, and eviction on the CPU. The GPU traces
all work whose terrain is resident and emits deferred requests for the rest.
This overlaps disk I/O, CPU preparation, GPU upload, and GPU ray tracing.

This stage follows the synchronous, CPU-scheduled correctness baseline in
stage 1 directly. The initial steps below validate the GPU work frontier with
all test tiles resident, then add bounded residency and asynchronous loading.

## Core work item

The unit of GPU scheduling is one unresolved azimuth-column segment:

```text
TileWorkItem
  tile key or resident slot
  slot generation
  azimuth index
  first polar index
  exact entry distance
```

One work item processes all unresolved polar rays in that azimuth column
through one tile. Polar slopes are sorted low to high, so collisions form a
prefix and the kernel can emit at most one unresolved suffix for the next tile.

This preserves the useful work-sharing that independent polar-ray dispatches
lose: all polar rays in a column follow the same XY DDA path.

## GPU frontier passes

Maintain two work-item buffers, `active` and `next`, plus an append counter for
each. One wavefront iteration is:

```text
active resident work items
        |
        v
trace tile-column segments
        |
        +--> terrain hits: write final distance/elevation/normal outputs
        |
        +--> unresolved suffix: append to next work-item buffer
        |
        +--> absent tile: append a tile request and defer the work item
        v
compact and bucket next work items by resident tile slot
        |
        v
swap active and next
```

No kernel may spin-wait for a tile to arrive. A tile is either resident for the
current pass or its work is deferred. Metal threadgroups have no global barrier,
so append, compaction, and bucketing are separate dispatch passes or use a
well-defined multi-pass prefix-sum implementation.

The first version does not need GPU-side tile requests. Load every tile in a
small test scene before tracing, assign each one a resident slot, and run the
frontier until its active work-item buffer is empty. This isolates GPU
scheduling and tile-boundary correctness from cache misses and disk I/O.

## Resident tile atlas

Give each resident tile a fixed slot. A slot contains:

- level-0 vertex elevations;
- the flat maximum-mipmap buffer, or a texture-array slice with mip levels;
- lower-left coordinates relative to the observer;
- cell size, side length, mip-level count, and maximum elevation;
- a generation counter; and
- a readiness flag.

The work item includes the generation counter observed when it was created.
The kernel rejects or defers an item if the slot has been recycled. This
prevents an eviction/reuse bug from tracing a ray through unrelated terrain.

A flat buffer atlas is the first implementation because it matches the current
single-tile kernel. A `texture2d_array` is a later option: all rechunked tiles
share dimensions, and texture-array slices provide cache-friendly 2D storage.
Maximum-pyramid values must be fetched exactly with `read`, never filtered with
`sample`, because interpolation would not be conservative.

## CPU tile manager

The CPU runs independently of the GPU and owns:

- mapping a grid tile key to a GeoTIFF path;
- the maximum-height metadata index;
- asynchronous GeoTIFF read/decode and `LoadedTile` preparation;
- staging/upload allocation;
- resident-slot assignment and eviction; and
- request deduplication and priority.

Maintain states such as:

```text
absent -> requested -> decoding -> prepared -> uploading -> resident -> evictable
```

GPU requests are hints, not the only source of work. The CPU should keep a
prefetch horizon ahead of the frontier using the azimuth wedge, minimum
possible entry distance, direct outgoing neighbours, and the maximum-height
index. This prevents a request from becoming a GPU-visible stall in the common
case.

Evict only tiles that are not referenced by active/deferred work items and not
leased by an in-flight command buffer. Prefer an LRU policy among eligible
tiles, biased against evicting tiles in the current or next frontier shell.

## Host/GPU synchronisation

Use a bounded number of in-flight command buffers and completion handlers to
recycle temporary buffers and tile-slot leases. The host should not call
`waitUntilCompleted` in the normal scheduling loop.

The CPU can prepare more tiles while prior command buffers execute. A completed
handler must return the command buffer's temporary work buffers and release any
resident tile leases only after the GPU has stopped reading them. Completion
handlers must update cache state through an explicitly synchronised host-side
queue; they must not race tile-loading workers.

An `MTLSharedEvent` is appropriate only if explicit CPU/GPU event values make
the design clearer; it has overhead and is not required for the initial
single-device implementation. Command-buffer ordering and completion handlers
are sufficient for the first wavefront cache.

## Missing tiles and termination

The GPU cache may not contain a requested tile for two different reasons:

1. the tile exists but is still being loaded; or
2. no source tile covers that grid key.

The host must communicate the distinction. Existing-but-not-resident work is
deferred. Confirmed missing coverage terminates its unresolved suffix without
writing a fictitious hit.

Similarly, suffixes at or beyond `max_distance` terminate without emitting a
tile request.

## Rollout and validation

Implement this stage as a sequence of independently verifiable changes:

1. **Preload every test tile.** Give each tile a permanent resident slot for
   the duration of a small scene. Do not implement eviction, asynchronous
   loading, or requests yet.
2. **Introduce GPU work items.** Replace stage 1's CPU tile queue with an
   initial GPU work-item buffer containing one observer-tile column segment per
   azimuth. Keep the work-item and output buffers readable for debugging.
3. **Append and compact continuations.** Trace the active work items, append
   their unresolved suffixes, compact the append buffer, swap the frontiers,
   and repeat until it is empty. In the all-resident mode this must reproduce
   stage 1 exactly.
4. **Bucket by resident slot.** Group work items that access the same tile
   slot. This improves locality and makes later per-slot resource access
   explicit, without changing the all-resident result.
5. **Bound the resident cache.** Introduce tile-slot pinning and generation
   checks. Test eviction only when no work item or in-flight command buffer can
   refer to the slot.
6. **Defer absent resident tiles.** Add a GPU request/deferred-work buffer.
   A tile known to exist but not yet resident is deferred; a tile known to be
   absent terminates its suffix as open coverage.
7. **Add asynchronous loading and upload.** Service deduplicated requests on
   background workers, insert completed tiles into free slots, and resubmit
   deferred work without blocking the GPU when other resident work exists.
8. **Add prefetch and eviction policy.** Use the azimuth wedge, tile-entry
   distance, direct outgoing neighbours, and the maximum-height index to keep
   likely tiles resident before the GPU asks for them.

At every step, compare distances, elevations, normals, and termination states
with the stage-1 baseline and the Python reference. Performance improvements
are not a reason to weaken the shared-boundary and float32 hand-off contract.

At every step, compare distances, elevations, normals, and termination states
with the stage-1 baseline and the Python reference. Performance improvements
are not a reason to weaken the shared-boundary and float32 hand-off contract.
