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
