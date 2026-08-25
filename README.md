# Panorama

GPU-accelerated terrain ray tracing and rendering for macOS.

## Development setup

This project requires Clang, Metal, GDAL, and `clang-format`. After cloning,
enable the repository's development hooks:

```sh
git config core.hooksPath .githooks
```

The pre-commit hook formats staged C, C++, Objective-C, Objective-C++, and Metal
source files with `.clang-format`, then stages the formatting changes before
the commit is created. It refuses partially staged source files to avoid
including unstaged work accidentally.

## Interactive viewer

Build the project, then launch the interactive viewer with:

```sh
make
./panorama-app --tile-dir data/swissalti3d-10-level-0-metal-u16-none --retain-quantized
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
