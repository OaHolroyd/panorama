# Panorama

Panorama is a GPU-accelerated terrain renderer for macOS. It turns digital
terrain models into shaded panoramas, distance maps, and elevation maps. Explore
terrain in the interactive viewer, with a minimap, location search, peak labels,
and astronomical lighting, or render PNGs from the command line.

This project is inspired by Jonathan de Ferranti's original [panoramas](https://viewfinderpanoramas.org/panoramas.html).

## Build

Requires macOS 26 or later, Xcode with the Metal shader toolchain, and GDAL,
PROJ, and `pkg-config`. After cloning the repository, install the dependencies
and build:

```sh
brew install gdal proj pkg-config
make -j
```

This produces `panorama-app` (interactive viewer), `panorama` (batch renderer),
and `panorama-tile-gen` (terrain preparation).

## Prepare terrain

Download elevation data, for example
[swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d)
(2 m GeoTIFF, LV95/LN02) or
[OS Terrain 50](https://osdatahub.os.uk/downloads/open/Terrain50).
The generator accepts GeoTIFF, SRTM `.hgt`, and Arc/Info `.asc` files, including
ASC packages inside ZIP archives. Keep ASC sidecars beside their rasters; ZIPs
can be used directly.

Place the source files under an input directory and generate tracing tiles:

```sh
./panorama-tile-gen \
  --input downloads/terrain \
  --output data/terrain
```

Each generation run requires inputs with the same coordinate system,
resolution, pixel registration, and aligned sample grid; it does not reproject
or resample them.
The output contains compact `.ptile` terrain tiles and a coverage manifest.

## Run

Open the prepared terrain in the viewer:

```sh
./panorama-app --tile-dir data/terrain
```

Or render a shaded image and diagnostic PNGs into the current directory:

```sh
./panorama --tile-dir data/terrain \
  --latitude 46.1 --longitude 7.7 --elevation 4500 \
  --synthetic-output
```

Choose an observer within your terrain coverage. Latitude and longitude use
WGS 84 decimal degrees; elevation is in metres above sea level. Run any of the
three programs with `--help` for options.

### Multiple datasets

You can combine datasets with different coordinate systems and resolutions.
Prepare each separately using matching tile-size, LOD, and compression settings,
then repeat `--terrain` in priority order. For example, combine detailed
swissALTI3D coverage with SRTM `.hgt` data for the surrounding region:

```sh
./panorama-tile-gen --input downloads/swissalti3d --output data/swissalti3d --compression none
./panorama-tile-gen --input downloads/srtm --output data/srtm --compression none
./panorama-app --terrain data/swissalti3d --terrain data/srtm
```

Swiss terrain takes priority; SRTM fills coverage gaps and extends the view
beyond it. The batch renderer also accepts repeated `--terrain` options.

### Peak labelling

Optional peak search and labels use a UTF-8 CSV with the header
`Lat,Lon,Elevation,Prom,Name`. Put it at `data/gazetteers/peaks.csv` or pass
`--peak-gazetteer PATH` to the viewer. Terrain and gazetteer data are ignored by
Git. Apple Maps search and observer time-zone lookup require internet access.

## Development

See [src/README.md](src/README.md) for the source architecture. Build with
`make DEBUG=1`; run `make check-geographic check-search` for core checks or
`make check-bvh check-camera` for GPU tests.

To enable automatic source formatting on commit:

```sh
brew install clang-format
git config core.hooksPath .githooks
```
