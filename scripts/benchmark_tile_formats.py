#!/usr/bin/env python3
"""Benchmark every prepared Swiss terrain format across tile-cache sizes.

Only ``--tile-dir`` and ``--tile-cache-mib`` are passed to ``panorama``. The
observer position, elevation, ray counts, maximum distance, and worker count
therefore retain the executable's defaults.

By default, all repetitions of one tile-directory/cache configuration run
back-to-back. The first run can therefore populate the filesystem cache before
the following runs measure the hot-cache behaviour. Randomized rounds remain
available explicitly for measuring a mixed working set.

Results are written incrementally as JSON. Each run contains the original
stdout/stderr, process wall time, and every timer line with its indentation
converted into an explicit nesting path.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CACHE_SIZES = (128, 256, 512, 1024)
DEFAULT_REPETITIONS = 5
DEFAULT_RANDOM_SEED = 20260818

TIMER_PATTERN = re.compile(
    r"^(?P<indent> +)(?P<name>[^:]+?)\s*:\s*"
    r"(?P<milliseconds>[0-9]+(?:\.[0-9]+)?) ms "
    r"\((?P<kind>wall|work)\)\s*$"
)
TILE_DIRECTORY_PATTERN = re.compile(
    r"^(?P<dataset>.+)-(?P<power>[0-9]+)-level-0(?:-(?P<format>.+))?$"
)


def parse_arguments() -> argparse.Namespace:
    """Parse sweep controls without duplicating panorama's rendering options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("./panorama"),
        help="panorama executable (default: ./panorama)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory containing prepared terrain directories (default: data)",
    )
    parser.add_argument(
        "--dataset",
        default="swissalti3d",
        help="prepared-directory prefix to benchmark (default: swissalti3d)",
    )
    parser.add_argument(
        "--tile-dir",
        action="append",
        type=Path,
        dest="tile_directories",
        help="benchmark this directory; repeat to override automatic discovery",
    )
    parser.add_argument(
        "--cache-mib",
        nargs="+",
        type=int,
        default=list(DEFAULT_CACHE_SIZES),
        help="tile-cache sizes in MiB (default: 128 256 512 1024)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="number of runs per tile-directory/cache configuration (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "configuration-order seed used by --randomize-rounds "
            f"(default: {DEFAULT_RANDOM_SEED})"
        ),
    )
    parser.add_argument(
        "--randomize-rounds",
        action="store_true",
        help=(
            "run one shuffled repetition of every configuration at a time; "
            "the default groups repetitions by configuration"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: timestamped file below benchmark-results)",
    )
    return parser.parse_args()


def contains_prepared_tiles(directory: Path) -> bool:
    """Return whether a directory contains a supported prepared tile file."""

    supported_suffixes = (
        ".tif",
        ".tiff",
        ".ptile",
        ".ptile.zlib",
        ".ptile.lz4",
        ".ptile.lzma",
        ".ptile.lzbitmap",
    )
    for path in directory.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(supported_suffixes):
            return True
    return False


def tile_directory_sort_key(directory: Path, dataset: str) -> tuple[int, str]:
    """Sort directory names numerically by tile power, then by representation."""

    match = TILE_DIRECTORY_PATTERN.fullmatch(directory.name)
    if match is None or match.group("dataset") != dataset:
        return (sys.maxsize, directory.name)
    return (int(match.group("power")), match.group("format") or "")


def discover_tile_directories(data_directory: Path, dataset: str) -> list[Path]:
    """Find level-0 prepared variants while excluding the original source data."""

    if not data_directory.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")

    candidates: list[Path] = []
    for directory in data_directory.glob(f"{dataset}-*-level-0*"):
        if directory.is_dir() and contains_prepared_tiles(directory):
            candidates.append(directory)
    return sorted(candidates, key=lambda path: tile_directory_sort_key(path, dataset))


def directory_statistics(directory: Path) -> dict[str, int]:
    """Return stable file-count and byte-size metadata for one tile variant."""

    files = [path for path in directory.iterdir() if path.is_file()]
    return {
        "file_count": len(files),
        "stored_bytes": sum(path.stat().st_size for path in files),
    }


def parse_timings(output: str) -> list[dict[str, Any]]:
    """Convert Timer's indentation into explicit hierarchical measurement paths."""

    timings: list[dict[str, Any]] = []
    names_by_level: list[str] = []
    for line in output.splitlines():
        match = TIMER_PATTERN.fullmatch(line)
        if match is None:
            continue

        spaces = len(match.group("indent"))
        level = max(0, spaces // 2 - 1)
        name = match.group("name").rstrip()
        if level < len(names_by_level):
            names_by_level[level] = name
            del names_by_level[level + 1 :]
        else:
            # Timer output normally advances one level at a time. Filling a
            # malformed indentation gap keeps the raw measurement recoverable.
            names_by_level.extend(["<unknown>"] * (level - len(names_by_level)))
            names_by_level.append(name)

        timings.append(
            {
                "path": names_by_level.copy(),
                "name": name,
                "kind": match.group("kind"),
                "milliseconds": float(match.group("milliseconds")),
            }
        )
    return timings


def total_elapsed(timings: list[dict[str, Any]]) -> float | None:
    """Return the renderer's overall wall measurement, if it was printed."""

    for timing in timings:
        if timing["name"] == "Total elapsed" and timing["kind"] == "wall":
            return float(timing["milliseconds"])
    return None


def write_results(path: Path, document: dict[str, Any]) -> None:
    """Atomically publish all completed runs after every subprocess finishes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_revision() -> str | None:
    """Return the checked-out revision when the script runs inside a repository."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def print_summary(document: dict[str, Any]) -> None:
    """Print median reported times while retaining fuller data in the JSON file."""

    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for run in document["runs"]:
        elapsed = run["reported_total_milliseconds"]
        if run["return_code"] != 0 or elapsed is None:
            continue
        key = (run["tile_directory"], run["tile_cache_mib"])
        grouped.setdefault(key, []).append((run["repetition"], elapsed))

    print("\nReported total-time medians (milliseconds):")
    print("tile directory                                      cache   all runs   warm runs")
    for (directory, cache), samples in sorted(grouped.items()):
        all_values = [value for _, value in samples]
        warm_values = [value for repetition, value in samples if repetition > 1]
        warm_text = f"{statistics.median(warm_values):9.3f}" if warm_values else "        -"
        print(
            f"{Path(directory).name:<51} {cache:5d} "
            f"{statistics.median(all_values):10.3f} {warm_text}"
        )


def main() -> int:
    """Run grouped or randomized repetitions and save each result incrementally."""

    arguments = parse_arguments()
    if arguments.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if any(size < 1 for size in arguments.cache_mib):
        raise ValueError("--cache-mib values must be positive")

    executable = arguments.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Panorama executable does not exist: {executable}")

    if arguments.tile_directories:
        tile_directories = [path.resolve() for path in arguments.tile_directories]
    else:
        tile_directories = [
            path.resolve()
            for path in discover_tile_directories(arguments.data_dir, arguments.dataset)
        ]
    if not tile_directories:
        raise RuntimeError(f"No prepared {arguments.dataset} tile directories were found")
    for directory in tile_directories:
        if not directory.is_dir() or not contains_prepared_tiles(directory):
            raise FileNotFoundError(f"No prepared tiles found in {directory}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = arguments.output or Path("benchmark-results") / f"tile-formats-{timestamp}.json"
    configurations = [
        (directory, cache)
        for directory in tile_directories
        for cache in arguments.cache_mib
    ]

    # Grouped repetitions deliberately make runs two through N hot-cache
    # measurements of the same files. The optional schedule preserves the old
    # mixed-working-set benchmark by shuffling every complete round separately.
    run_plan: list[tuple[int, Path, int]] = []
    if arguments.randomize_rounds:
        for repetition in range(1, arguments.repetitions + 1):
            round_configurations = configurations.copy()
            random.Random(arguments.seed + repetition).shuffle(round_configurations)
            run_plan.extend(
                (repetition, directory, cache)
                for directory, cache in round_configurations
            )
    else:
        run_plan.extend(
            (repetition, directory, cache)
            for directory, cache in configurations
            for repetition in range(1, arguments.repetitions + 1)
        )

    document: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": None,
        "executable": str(executable),
        "git_revision": git_revision(),
        "dataset": arguments.dataset,
        "repetitions": arguments.repetitions,
        "tile_cache_mib": arguments.cache_mib,
        "execution_order": "randomized rounds" if arguments.randomize_rounds else "grouped",
        "random_seed": arguments.seed if arguments.randomize_rounds else None,
        "render_options": "panorama defaults",
        "tile_directories": {
            str(directory): directory_statistics(directory) for directory in tile_directories
        },
        "runs": [],
    }
    write_results(output, document)

    total_runs = len(run_plan)
    completed_runs = 0
    failed_runs = 0
    try:
        for repetition, tile_directory, cache_mib in run_plan:
            completed_runs += 1
            command = [
                str(executable),
                "--tile-dir",
                str(tile_directory),
                "--tile-cache-mib",
                str(cache_mib),
            ]
            print(
                f"[{completed_runs}/{total_runs}] repetition {repetition}, "
                f"cache {cache_mib} MiB, {tile_directory.name}",
                flush=True,
            )

            started = time.perf_counter()
            result = subprocess.run(command, capture_output=True, check=False, text=True)
            process_milliseconds = 1_000.0 * (time.perf_counter() - started)
            timings = parse_timings(result.stdout)
            elapsed = total_elapsed(timings)
            if result.returncode != 0:
                failed_runs += 1

            document["runs"].append(
                {
                    "sequence": completed_runs,
                    "repetition": repetition,
                    "tile_directory": str(tile_directory),
                    "tile_cache_mib": cache_mib,
                    "command": command,
                    "return_code": result.returncode,
                    "process_wall_milliseconds": process_milliseconds,
                    "reported_total_milliseconds": elapsed,
                    "timings": timings,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            write_results(output, document)

            status = "failed" if result.returncode != 0 else "completed"
            reported = "unavailable" if elapsed is None else f"{elapsed:.3f} ms"
            print(
                f"  {status}: reported {reported}, process {process_milliseconds:.3f} ms"
            )
    except KeyboardInterrupt:
        print(f"\nInterrupted; {len(document['runs'])} completed runs remain saved in {output}")
        return 130

    document["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_results(output, document)
    print_summary(document)
    print(f"\nSaved {len(document['runs'])} runs to {output}")
    if failed_runs != 0:
        print(f"{failed_runs} runs failed; inspect their stderr fields in the JSON file.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
