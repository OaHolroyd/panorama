#!/usr/bin/env python3
"""
data.py

This tool enables the unarchiving/reformatting and also resampling of data from
the following supported sources:
    OS Terrain 50  (osdatahub.os.uk/downloads/open/Terrain50)
    swissALTI3D    (swisstopo.admin.ch/en/geodata/height/alti3d.html)
"""

import argparse
from pathlib import Path
from zipfile import ZipFile
import requests
import numpy as np


def rm_tree(pth):
    """recursively deletes the contents of a directory """
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def download_file(url, root="."):
    """download file from url"""
    file = f"{root}/{url.split('/')[-1]}"
    with requests.get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        with open(file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return file


def gzd_name(source, res, ncols, nrows, y0, x0):
    """returns the gzd name for the data"""
    return f"{source.upper()}{res:02d}-{ncols:03d}-{nrows:03d}-{y0:04d}-{x0:04d}.gzd"


def process_ost50():
    """reformats the OS Terrain 50 data zip"""
    ORIGINAL_DATA_ROOT = "./terr50_gagg_gb"
    OUTPUT_DATA_ROOT = "./ost50"

    # ensure output directory exists
    Path(OUTPUT_DATA_ROOT).mkdir(parents=True, exist_ok=True)

    # unzip the main archive
    with ZipFile(f"{ORIGINAL_DATA_ROOT}.zip", 'r') as z:
        try:
            z.extractall(path=ORIGINAL_DATA_ROOT)
        except FileNotFoundError:
            print(f"ERROR: '{ORIGINAL_DATA_ROOT}.zip' not found")
            return

    # recursively unpack and reformat the nested data
    path = Path(ORIGINAL_DATA_ROOT)
    for p in path.rglob("*"):
        if p.name[-4:] == ".zip":
            ascfile = f"{p.name[0:4].upper()}.asc"
            with ZipFile(p, 'r') as z:
                z.extract(member=ascfile, path=OUTPUT_DATA_ROOT)

            # load from .asc
            ascfile = f"{OUTPUT_DATA_ROOT}/{ascfile}"
            with open(ascfile, encoding='utf-8') as f:
                h = [next(f) for i in range(5)]
            data = np.flipud(np.loadtxt(ascfile, skiprows=5))

            # read header data
            for i in range(5):
                h[i] = int(h[i].split(" ")[1])

            # reformat as .gzd
            gzdfile = f"{OUTPUT_DATA_ROOT}/{gzd_name('OST', h[4], h[0], h[1], h[3]//1000, h[2]//1000)}"
            data.tofile(gzdfile)

            # remove the data as we go
            Path(ascfile).unlink()
            p.unlink()

    # clean up
    rm_tree(path)
    Path(f"{ORIGINAL_DATA_ROOT}.zip").unlink()


def process_swt02():
    """takes the .csv of files, downloads and reformats them"""
    OUTPUT_DATA_ROOT = "./swt02"

    # ensure output directory exists
    Path(OUTPUT_DATA_ROOT).mkdir(parents=True, exist_ok=True)

    # find the .csv file
    path = Path(".")
    csvfile = None
    for p in path.rglob("*"):
        if p.name[-4:] == ".csv":
            csvfile = p.name
            break
    if csvfile is None:
        print("ERROR: csv file not found")
        return

    # download and unzip listed in the .csv
    with open(csvfile, encoding='utf-8') as f:
        for line in f:
            # download and unzip
            file = download_file(line.strip(), root=OUTPUT_DATA_ROOT)
            with ZipFile(file, 'r') as z:
                z.extractall(path=OUTPUT_DATA_ROOT)
                xyzfile = f"{OUTPUT_DATA_ROOT}/{z.namelist()[0]}"

            # load from .xyz
            data = np.loadtxt(xyzfile, skiprows=1)

            # extract coords and reshape
            x0 = int(data[0, 0])//1000
            y0 = int(data[0, 1])//1000
            data = np.flipud(data[:, 2].reshape([500, 500]))

            # reformat as .gzd
            gzdfile = f"{OUTPUT_DATA_ROOT}/{gzd_name('SWT', 2, 500, 500, y0, x0)}"
            data.tofile(gzdfile)

            # remove the data as we go
            Path(file).unlink()
            Path(xyzfile).unlink()
    Path(csvfile).unlink()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', action='store', type=str,
                        help='data source (ost50, swt02)')
    parser.add_argument('-r', '--resample', action='store', type=int,
                        help='resample to the specified resolution (integer)')
    args = parser.parse_args()

    if args.source is not None:
        if args.source == 'ost50':
            process_ost50()
        elif args.source == 'swt02':
            process_swt02()
        else:
            print(f"ERROR: source {args.source} not recognised")

    if args.resample is not None:
        # check if data exists
        # resample
        pass
