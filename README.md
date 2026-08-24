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

Build the project, then launch the basic fixed-position viewer with:

```sh
make
./panorama-app --tile-dir data/swissalti3d-10-level-0-metal-u16-none --retain-quantized
```

Drag with the mouse or use the arrow/WASD keys to change heading and pitch.
Use the trailing Render Settings inspector to switch between white,
distance-coloured, and elevation-coloured terrain; select a colourmap and fixed
range; or disable surface-normal lighting. Collapse or reveal the inspector by
clicking the `sidebar.right` button at the trailing edge of the window toolbar.
Run `./panorama-app --help` for observer, image-size, and field-of-view options.
