#!/usr/bin/env python3
"""Download the SwissALTI3D GeoTIFFs listed in the accompanying URL catalogue.

The output filenames use the existing ``easting-northing.tif`` convention.
Repeated runs skip completed files, while incomplete downloads are written to
``.part`` files and atomically renamed only after they finish successfully.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
TILE_NAME = re.compile(r"swissalti3d_\d{4}_(\d{4}-\d{4})")
CHUNK_SIZE = 1024 * 1024


def destination_for(url: str, output_directory: Path) -> Path:
    """Return the repository's short tile name for one SwissTopo URL."""
    match = TILE_NAME.search(url)
    if match is None:
        raise ValueError(f"Could not derive a tile coordinate from {url!r}")
    return output_directory / f"{match.group(1)}.tif"


def download(url: str, destination: Path, retries: int) -> None:
    """Download one URL to a temporary sibling, then publish it atomically."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
            temporary.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(f"Could not download {url}: {error}") from error
            delay = 2**attempt
            print(f"  retrying in {delay}s: {error}", file=sys.stderr)
            time.sleep(delay)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--retries", type=int, default=3, metavar="N")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.retries < 0:
        raise ValueError("--retries must not be negative")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        line.strip()
        for line in arguments.catalogue.read_text().splitlines()
        if line.strip()
    ]
    downloaded = skipped = 0
    for index, url in enumerate(urls, start=1):
        destination = destination_for(url, arguments.output_dir)
        if destination.is_file() and destination.stat().st_size > 0:
            skipped += 1
            continue
        print(f"[{index}/{len(urls)}] {destination.name}")
        download(url, destination, arguments.retries)
        downloaded += 1

    print(f"Downloaded {downloaded} tile(s); skipped {skipped} existing tile(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"download.py: {error}", file=sys.stderr)
        raise SystemExit(1)
