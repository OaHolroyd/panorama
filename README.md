# Panorama

Panorama is a GPU-accelerated terrain ray tracer and renderer for macOS. It
turns digital terrain model (DTM) data into panoramic distance and height maps,
surface-normal diagnostics, and shaded terrain images. The tracing and image
presentation run on the GPU, making it practical to generate large panoramas
automatically or explore the terrain interactively.

## Executables

Building the project produces three executables:

- `panorama-tile-gen` prepares aligned DTM GeoTIFF or raw SRTM HGT inputs for
  tracing.
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
from the Ordnance Survey Data Hub and extract the ASCII grid tiles. Terrain 50
has 50 m spacing which is quite coarse; the 5 m OS Terrain 5 data is much better but is unfortunately a paid product. `panorama-tile-gen` reads GeoTIFF input, so convert
each Terrain 50 `.asc` file while assigning the British National Grid CRS:

```sh
mkdir -p downloads/os-terrain-50-geotiff
gdal_translate -a_srs EPSG:27700 \
  downloads/os-terrain-50/NN/nn20.asc \
  downloads/os-terrain-50-geotiff/nn20.tif
```

Repeat the conversion for the tiles covering the area of interest. All input
rasters supplied to one tile-generation run must use the same projected or
geographic CRS, resolution, pixel registration, and aligned sample grid. Tile
generation rechunks the data without reprojecting or resampling it.

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

Use `downloads/os-terrain-50-geotiff` as the input directory to prepare the OS
data in the same way. Run `./panorama-tile-gen --help` for GeoTIFF output,
chunk-size, grid-origin, and overwrite options.

Observer eastings and northings are always expressed in the prepared dataset's
projected CRS. The executable defaults describe the Swiss example, so supply
British National Grid coordinates with `--easting`, `--northing`, and
`--elevation` when using OS data. Use `--retain-quantized` only with Metal tiles
created using `--sample-type uint16`.

The resulting directory can be opened interactively:

```sh
./panorama-app --tile-dir data/swissalti3d-2m-metal --retain-quantized
```

or rendered non-interactively:

```sh
./panorama \
  --tile-dir data/swissalti3d-2m-metal --retain-quantized \
  --synthetic-output
```

## Interactive viewer

Build the project, then launch the interactive viewer with:

```sh
make
./panorama-app --tile-dir data/swissalti3d-2m-metal --retain-quantized
```

Drag with the mouse or use the arrow/WASD keys to change heading and pitch;
scroll to zoom. The trailing Render Settings inspector controls resolution,
lighting, distance/elevation colourmaps and scaling, and optional multiscale
feature outlines.

The map toolbar button enables the minimap and terrain-point inspection as one
feature. Hover either the panorama or map to preview a point. Right-click the
panorama to lock its current point; left-click the minimap to lock a map point
and turn the camera toward it. Option-click the minimap, or use its secondary-
click menu, to move the observer while retaining the current eye height above
the terrain. The map's scope and expand buttons select its focus and size.

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
