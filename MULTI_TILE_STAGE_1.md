# Multi-tile ray tracing: stage 1 — synchronous correctness baseline

## Goal

Extend the current single-tile Metal ray tracer to traverse a set of terrain
tiles while keeping the tile scheduler on the CPU. This stage deliberately
accepts CPU/GPU synchronisation after each tile so that tile-boundary behaviour
can be compared directly with `panorama-python/ray_tracing.py`.

The result must be correct before any attempt is made to overlap disk I/O,
uploads, or GPU execution.

## Execution model

The CPU owns a priority queue of pending tiles. Each pending tile contains the
unresolved suffix of every azimuth column that reaches it:

```text
pending[tile_key][azimuth_index] = {
    first_polar_index,
    entry_distance,
}
```

Tiles are processed in increasing Manhattan distance from the observer tile.
For straight XY rays, a successor tile is always one grid step farther from
the observer in at least one dimension, so this ordering ensures that every
tile has received all of its predecessor states before it is processed.

For each pending tile:

1. The CPU loads its GeoTIFF into `LoadedTile` and prepares its mipmap.
2. The CPU creates an input list containing its active azimuth-column states.
3. The GPU traces those states through the tile.
4. The CPU waits for the GPU result, reads explicit continuation states, and
   inserts successor states into the pending-tile queue.
5. A missing GeoTIFF terminates its incoming states; missing coverage is open
   sky, not a vertical terrain wall.

The initial implementation may dispatch independent polar rays, but it must
write enough information for the CPU to determine the first unresolved polar
ray for each azimuth column. A later implementation can instead make one GPU
work item represent a whole azimuth-column suffix.

## Required host/device state

Define an explicit continuation result. Do not infer continuation from a zero
distance, because zero is also a possible output sentinel and makes debugging
ambiguous.

```text
TileContinuation
  azimuth_index
  first_unresolved_polar
  entry_distance
  status
```

`status` should distinguish at least:

- all rays resolved in this tile;
- unresolved suffix crosses into another tile;
- maximum distance reached;
- missing coverage; and
- malformed/internal-error output while debugging.

The incoming work item must contain the exact `entry_distance`. A nudged point
is used only to classify ownership of the next tile or cell; it must never
replace the exact ray geometry.

## Boundary contract

An incoming tile segment begins at the maximum mipmap level, except for the
observer's own tile, which begins at level 1. The tile kernel must therefore:

- reconstruct the entry point from the exact hand-off distance;
- nudge that point in the ray direction only to select the owning fine cell;
- align that cell to the selected maximum-mipmap cell;
- retain the exact entry distance as the DDA segment's near boundary;
- descend through mip levels using the parent interval's true near boundary;
- report the exact far boundary when handing an unresolved suffix to a
  neighbouring tile.

At shared tile edges and corners, classify the outgoing tile using the same
cell-scaled, float32-safe nudge used by the Python reference. The tile exit
distance itself must remain unnuded.

## Validation

Compare the Metal output byte-for-byte where practical, or with documented
float32 tolerances otherwise, against the Python reference. Test at least:

- a flat synthetic 2×2 and 3×3 tile grid;
- a tile seam with continuous terrain heights;
- rays leaving through each cardinal edge;
- rays leaving exactly through a corner;
- observer positions on a tile edge and on a tile corner;
- incoming rays which start on an edge at the maximum mipmap level;
- a ray that crosses several tiles without a collision;
- a hit immediately after entering a tile; and
- absent neighbour tiles.

Do not proceed to stage 2 until the final distances and elevations agree with
the reference for these cases.

## Deliberate limitations

- `waitUntilCompleted` after each processed tile is acceptable.
- Tile loading can be synchronous.
- One tile may use one or more dedicated buffers.
- The CPU may inspect every continuation result.

These limitations isolate numerical and ownership errors from concurrency
errors.
