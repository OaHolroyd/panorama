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
its logarithmic speed control spans 1 m/s to 10 km/s, and W/S adjusts the speed
multiplicatively. Terrain height mode maintains a fixed AGL clearance, while
Flight mode uses camera pitch to climb or descend.
If Flight mode meets terrain, forward motion is held while mouse steering and
W/S speed adjustment remain available; steer or climb clear, then press Space
to resume.

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
