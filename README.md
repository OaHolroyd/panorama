# Panorama

Panorama is a GPU-accelerated terrain ray tracer and renderer for macOS. It
turns digital terrain model (DTM) data into panoramic distance and height maps,
surface-normal diagnostics, and shaded terrain images. The tracing and image
presentation run on the GPU, making it practical to generate large panoramas
automatically or explore the terrain interactively.

## Executables

Building the project produces three executables:

- `panorama-tile-gen` prepares aligned DTM GeoTIFF, raw SRTM HGT, and Arc/Info
  ASC inputs for tracing. ASC files may be loose or retained in ZIP archives.
  It can write conventional rechunked GeoTIFFs or compact, optionally compressed
  Metal tiles intended for fast GPU loading.
- `panorama` is the batch renderer. It traces an angular panorama or a pinhole
  camera view and writes diagnostic PNGs and, optionally, a shaded synthetic
  terrain image.
- `panorama-app` is the interactive viewer. It retains terrain and GPU state
  while the observer looks around, changes rendering settings, or interacts
  with the minimap.

## Getting started

Install Clang/Xcode command-line tools, Metal, GDAL, and `clang-format`, then
build all three programs:

```sh
make
```

### Download terrain data

For Switzerland, download the free [swissALTI3D dataset](https://www.swisstopo.admin.ch/en/height-model-swissalti3d)
from swisstopo. Select the 2 m, Cloud Optimized GeoTIFF (COG) product in the
LV95/LN02 coordinate system and extract the required 1 km tiles into one source
directory, for example `downloads/swissalti3d`. A 0.5 m product is also available but at anything beyond the nearest detail this resolution is excessive so the 2 m version is recommended.

For Great Britain, download the free [OS Terrain 50 grid](https://osdatahub.os.uk/downloads/open/Terrain50)
from the Ordnance Survey Data Hub. Terrain 50 has fairly coarse 50 m spacing;
the more detailed 5 m OS Terrain 5 product is paid. The downloaded Terrain 50
ZIP files can be placed below one input directory and passed directly to
`panorama-tile-gen`, which reads each contained `.asc` with its `.prj`, `.gml`,
and GDAL sidecars without extracting the archive. Extracted ASC packages are
also supported, provided their sidecars remain beside the raster.

All input rasters supplied to one tile-generation run must use the same
projected or geographic CRS, resolution, pixel registration, and aligned sample
grid. Tile generation rechunks the data without reprojecting or resampling it.

For SRTM, place the raw `.hgt` files below one input directory; nested
directories are supported. The filename supplies each tile's one-degree WGS 84
bounds, while its 3601- or 1201-sample side selects one- or three-arcsecond
spacing. HGT elevations are decoded as signed, big-endian 16-bit metres and the
standard `-32768` void value is treated as no-data. The current renderer still
requires one of its supported projected CRSs; accepting native geographic
tiles there is part of the planned multi-source ray-tracing work.

### Prepare tracing tiles

Metal tiles with quantized decimetre elevations without compression provide a
compact, fast-loading representation. First inspect the proposed operation,
then generate the tiles:

```sh
./panorama-tile-gen \
  --input downloads/swissalti3d \
  --output data/swissalti3d-2m-metal \
  --format metal --sample-type uint16 --compression none
```

Use the directory containing the Terrain 50 ZIP or extracted ASC packages as
the input directory to prepare OS data in the same way. Run
`./panorama-tile-gen --help` for GeoTIFF output, chunk-size, grid-origin, and
overwrite options.

Observer eastings and northings are always expressed in the prepared dataset's
projected CRS. The executable defaults describe the Swiss example, so supply
British National Grid coordinates with `--easting`, `--northing`, and
`--elevation` when using OS data. Uint16 Metal tiles remain quantized in the GPU
atlas by default; `--discard-quantized` expands them to Float32 during loading.

The resulting directory can be opened interactively:

```sh
./panorama-app --tile-dir data/swissalti3d-2m-metal
```

or rendered non-interactively:

```sh
./panorama \
  --tile-dir data/swissalti3d-2m-metal \
  --synthetic-output
```

## Interactive viewer

The viewer and CLI generate ray directions/slopes and select terrain LOD on
the GPU, including angular panoramas and Brown–Conrady camera distortion. Pixel footprint and LOD plans are cached across camera rotations;
movement, zoom, resolution or LOD changes update the relevant GPU plan. The CPU
receives per-source LOD decisions for loading and BVH preparation. Missing terrain
and shadows still finish before a frame is published; the viewer displays only
complete frames.

Use `./panorama-app --trace-diagnostics` to log frame wall latency, GPU producer
time, submission count, and whether streaming was needed. Resident frames encode
primary tracing, shadows, colouring and (when the minimap is visible) collision
point projection into one command.
GPU producer time excludes synchronous BVH preparation and streaming work;
wall latency includes them. See [the BVH performance investigation](todo/bvh-performance-investigation.md)
for measurements, cache guidance and a repeatable camera benchmark.
`Camera preparation` lines separately report GPU LOD-plan wall/device time and
cumulative plan/footprint updates. Ray generation is included in the producer's
GPU time on the resident path. Plan preparation is included in frame wall time.

For intermittent viewer stalls, capture `./panorama-app --trace-diagnostics >
diagnostics.log 2>&1` (with your usual options). Frame lines include elapsed
timestamps and cumulative BVH cache counts. An independent monitor prints
`Health` lines every second, including process footprint, Metal allocations,
thermal state, and the current worker/display/minimap stage and its age. These
continue even while the render worker or UI thread is blocked. Submission,
completion, presentation, stale-frame rejection and missing-drawable counts are
cumulative. On the display path, `refreshed` counts snapshots replaced with newer
frames after drawable acquisition; `stale` counts resolution mismatches rejected
before encoding.
`Frame shadows` reports newly cached shadow casters, BVH repair passes, their GPU
time and capacity fallbacks. Warm shadow views should need no repair or terrain I/O.
`Frame phases` separates preparation, primary/shadow streaming repair and producer
waits. `Frame work` reports per-frame BVH builds/evictions, scene size, streaming
passes and terrain I/O. Its detail GPU time covers synchronous traversal;
resident producer GPU time remains on the main `Frame` line. Cache hits count
streaming tile lookups, not individual GPU accesses. `Health` also samples the
innermost terrain stage, and `Slow stage` records inclusive wall durations over
100 ms (nested durations overlap). This instrumentation is enabled only by
`--trace-diagnostics`, apart from the inexpensive per-frame phase timers.
Display revisions distinguish new camera images from repeated
presentations. `idle` means that path is outside its instrumented callback, not
necessarily that the whole UI is responsive. Keep capturing for about ten seconds
after the slowdown begins before closing the app.

The minimap retains the camera cone and visible-terrain coverage. Coverage uses a
compute-generated bitmap displayed by MapKit, with no separate transparent Metal
view. Map panning and zooming reuse the latest collision snapshot. Updates are
bounded to one active job and one replaceable pending job; hiding the map stops
new collision projection and mask work, drops pending results and releases the
visibility buffers once active work drains. Reopening requests a fresh trace even
when the camera is stationary.

With diagnostics enabled, `Minimap mask` reports worker time (including CRS-grid
preparation and image creation) and GPU time. `Minimap publish` reports latency
from requesting the bitmap to handing it to MapKit; this excludes MapKit's own
subsequent drawing/composition. In the minimap `Health` line, `refreshed` counts
collision snapshot encodes, `submitted/completed` counts mask commands,
`presented` counts image publications, and `stale` counts cancelled generations.
These should stop increasing after hiding the map and draining active work.

The BVH backend automatically reuses a resident scene hierarchy to trace across
cached tiles in one GPU pass. Uncached candidates fall back to bounded streaming.
For the default full-detail Swiss view, use `--bvh-cache-mib 2048` to keep its
working set resident, or `--lod-scale 1` to fit coarser distant terrain within
the default cache. Repeated views then avoid per-tile submissions and CPU ray
grouping; cold views and cache misses still incur loading/building work.

The viewer defaults to the `metal-bvh` terrain backend and LOD scale `1.5` to
reduce distant-terrain cache pressure. Use `--lod-scale 0` for full detail.
In Viewer Settings →
Terrain, the Raytracer selector switches between Mipmap and BVH and redraws the
current view. The Raytracer menu provides the same choices. Mipmap uses the
`software` backend; both executables also accept
`--raytracer software|metal-bvh`, `--bvh-block-cells N` (default `4`), and
`--bvh-cache-mib N` (default `512`). The batch
renderer keeps `software` as its default. For example:

```sh
./panorama --raytracer metal-bvh --bvh-cache-mib 512 \
  --max-distance 600000 --lod-scale 0 --synthetic-output
```

Both raytracers share a session-owned inter-tile BVH on devices supporting
Metal ray tracing. It uses conservative manifest elevation bounds to select
candidate tiles; Mipmap then traverses each selected tile's maximum hierarchy,
while BVH uses its detailed surface acceleration. The shared catalogue survives
camera turns, image resizing, and backend switches, and rebuilds after XY or LOD
changes. Mipmap retains grid selection on devices without Metal ray-tracing
support. Coverage gaps still terminate rays. Sharing the catalogue alone did not
improve Mipmap timings in the tested M2 view; performance depends on the camera
and device.

Metal BVH streams full-resolution tiles on demand; LOD is optional. Detailed tile
BVHs and their immutable vertices are cached by tile and LOD. Rays retain their
progress across batches, including when a frame's terrain exceeds the cache. Tiles are scheduled in outward grid shells to avoid
rebuilding the same tile for successive groups of rays within a frame.

`--bvh-cache-mib` bounds the requested Metal storage for detailed BVHs, owned
vertices, block metadata, bounds, and peak build/compaction workspace. It is
**additional to** `--tile-cache-mib`; ray/output buffers, the small catalogue
BVH, batch instance structures (at most 64 tiles), and driver allocation overhead
are separate. A cache must fit at least one tile plus its build workspace;
otherwise the error reports the required bytes. LRU eviction occurs only after
GPU work completes. Statistics report resident bytes, peak reservation, builds,
cache hits, and evictions. A small cache increases construction costs, particularly
between viewer frames.

Detailed bounds use a fixed tile-centred curvature anchor. Metal instance
transforms apply XY translation and Z shear for the current observer, preserving
horizontal ray distance without rebuilding the cached terrain. Observer movement
updates the small catalogue and batch instance hierarchies; LOD changes select
separate cache entries. GPU traversal, tile loading/building, and instance setup
are reported separately beneath inclusive `BVH streaming trace` time.
Actual speed and compaction savings depend on the GPU. Devices without Metal
ray-tracing support can use the software backend. The viewer traces shadows
against the resident BVH scene. Missing shadow casters are requested by the GPU,
loaded into the ordinary detailed BVH cache at the selected LOD, and retained for
later frames. Shadow repair pins the current scene and its new casters, with
exact software streaming as a fallback if they cannot fit within the cache budget.
It repairs incomplete primary or shadow results before publishing the image.
The batch renderer retains the software shadow traversal.

For a headless shadow-cache check with the viewer's 600 km range and 1.5 LOD,
run `obj/release/metal-bvh-test --benchmark-camera TILE_DIR gpu-shadows`.
This logs caster loads, repair passes and terrain I/O through warm pans, movement
and zoom. `make check-bvh` also checks shadow reuse and bounded-cache fallback
against software visibility, using float/quantized terrain and both collision modes.

`make check-bvh` runs Metal API validation and software/BVH comparisons on
generated terrain, including retained and expanded uint16, float samples,
partial blocks, coverage gaps, range clipping, resizing, relocation, backend
switching, cold and resident shadows, producer fallback, and forced cache eviction.
`make check-camera` checks projection/reprojection, angular pixel centres, inverse
lens distortion, LOD bounds and footprint decisions,
checks projection-cache reuse, and exercises complete GPU-camera producers through
streaming repair, resizing, relocation, shadows, and backend changes.
For a headless 1600×900 benchmark including GPU ray preparation, LOD and complete
rendering (without shadows or minimap), run:

```sh
obj/release/metal-bvh-test --benchmark-camera data/swissalti3d-10-level-0-metal-u16-none-lod-point gpu
```

`make check-minimap` compares compute coverage with the former point rasterizer
and a CPU reference under Metal validation. It checks invalid hits, duplicate
opacity, backing dimensions, image lifetime, horizontal-distance reconstruction,
and projection accuracy in all three supported terrain CRSs.

`make check-manifest` validates manifest versions, bounds across all LODs,
raw/compressed tile scans, and generator upgrades of existing version-1 manifests
without rewriting tiles.
The test executable also accepts a Swiss prepared-tile directory, optional range
in metres (default 21000), and BVH cache in MiB (default 512):

```sh
obj/release/metal-bvh-test data/swissalti3d-10-level-0-metal-u16-none-lod-point 600000 64
```

The real-terrain check requires identical hit masks and bounded distance and
elevation errors. Patch-local normals can differ at the existing collision
solver's cell-edge tolerance; away from those edges, the comparison allows only
half-precision rounding. BVH step diagnostics count procedural candidates;
evaluation diagnostics count precise cell tests.

`panorama-tile-gen` now writes version-2 `panorama-terrain-manifest.bin` entries
with minimum and maximum elevations enclosing **all stored LODs**, including
quantization and finite no-data fill values. Re-running the original generation
command without `--overwrite` upgrades an old manifest by scanning existing tile
payloads; it does not regenerate those tiles. Existing version-2 entries are
reused for skipped files. Version-1 or absent manifests remain readable with
conservative culling where bounds are unavailable.

Build the project, then launch the interactive viewer with:

```sh
make
./panorama-app --tile-dir data/swissalti3d-2m-metal
```

Drag with the mouse or use the arrow/WASD keys to change heading and pitch;
scroll to zoom. The trailing tabbed inspector separates viewer controls from
observer positioning. The Viewer tab controls resolution, lighting,
distance/elevation colourmaps and scaling, and optional multiscale feature
outlines.

The Position tab's Movement section can switch from this Browse behaviour to
keyboard Roam mode. In Roam mode, WASD moves relative to the current heading;
turning can use either the arrow keys or pointer motion over the panorama.
Mouse turning replaces click-and-drag rotation while selected and has its own
sensitivity control. Movement can maintain either a fixed height above terrain
or an absolute altitude. Speed and the maximum observer-update rate are
configurable; movement requests are coalesced when terrain rendering completes
more slowly than the selected rate. Press Space to pause navigation and free
the pointer for other controls; a visible badge remains until Space resumes it.
Cruise mode moves forward continuously along the mouse-controlled heading;
its logarithmic speed control spans 3.6 km/h to 36,000 km/h, and W/S adjusts
the speed multiplicatively. It opens paused in Flight mode, which holds
absolute altitude and uses camera pitch to climb or descend; Terrain mode
instead maintains a fixed AGL clearance.
Cruise steering acts as a virtual joystick: the central HUD's fixed boresight
is neutral, cursor displacement controls continuous yaw and pitch rates, and a
small central dead zone prevents drift. Its compass ribbon, pitch ladder, and
artificial horizon show the current view attitude; Aircraft mode also adds a
bank indicator.
The optional Aircraft toggle changes horizontal steering into coordinated
banked turns. Its speed setting becomes a trim speed, while climbs lose
airspeed and dives gain it; pausing restores a wings-level attitude.
If Flight mode meets terrain, forward motion is held. Drag to steer or climb,
use W/S to adjust speed, then press Space to resume.

The minimap and terrain-point inspection are enabled by default; the map
toolbar button hides or reveals them as one feature. Hover either the panorama
or map to preview a point. Right-click the panorama to lock its current point;
left-click the minimap to lock a map point and turn the camera toward it. The
map can always be panned and zoomed. The location button recentres it on the
observer without changing scale, while the scope button toggles following the
panorama mouseover point. Following pauses while the pointer is over the map.
A locked point can be used as the new observer location with **Move here**.
Option-click the minimap, or use its secondary-click menu, to move immediately.
The Position tab also accepts decimal WGS 84 `latitude, longitude`, Swiss LV95
easting/northing, and OS National Grid coordinates such as `NG 90716 59877`,
`NG907598`, or `190716, 859877`. Its coordinate-system menu defaults to Auto,
which uses distinctive syntax and prepared-terrain coverage to resolve the
input. If several interpretations remain plausible, it names them and waits
for an explicit menu selection. Prefixes such as `WGS84`, `LV95`, `BNG`, and
`DATASET` are also accepted in Auto mode. Eye-height controls set the retained
height above the terrain for jumps and vertical adjustments. The expand button
changes map size; the grid button overlays the complete prepared-tile coverage.
If the requested startup observer is outside that coverage, the viewer opens on
a central available tile with the coverage overlay already enabled.

Collapse or reveal the inspector with the `sidebar.right` toolbar button. Run
`./panorama-app --help` for observer, image-size, and field-of-view options.
The Viewfinder colourmap reproduces the indexed distance palette published by
[Viewfinder Panoramas](https://viewfinderpanoramas.org/panoramas.html).

## Development setup

After cloning, enable the repository's development hooks:

```sh
git config core.hooksPath .githooks
```

The pre-commit hook formats staged source files with `.clang-format`, then
stages the formatting changes before the commit is created. It refuses
partially staged source files to avoid including unstaged work accidentally.
