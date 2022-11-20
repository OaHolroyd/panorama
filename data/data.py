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
import h5py
import matplotlib.pyplot as plt


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


def process_nzt08():
    """reformats the NZ 8m DEM data zip"""
    ORIGINAL_DATA_ROOT = "./lds-nz-8m-digital-elevation-model-2012-KEA"
    OUTPUT_DATA_ROOT = "./nzt08"

    # ensure output directory exists
    Path(OUTPUT_DATA_ROOT).mkdir(parents=True, exist_ok=True)

    # unzip the main archive
    # with ZipFile(f"{ORIGINAL_DATA_ROOT}.zip", 'r') as z:
    #     try:
    #         z.extractall(path=ORIGINAL_DATA_ROOT)
    #     except FileNotFoundError:
    #         print(f"ERROR: '{ORIGINAL_DATA_ROOT}.zip' not found")
    #         return

    # output size and extent
    dx = 8
    out_size = np.array([10000, 10000], dtype=int)
    out_n = out_size//dx
    # out_x = np.array([1048000, 2098000], dtype=int)
    # out_y = np.array([4718000, 6208000], dtype=int)
    out_x = np.array([1238000, 1278000], dtype=int)
    out_y = np.array([4978000, 5018000], dtype=int)

    # loop over all output cells
    y = np.array([out_y[0], out_y[0]+out_size[1]])
    count = 0
    max_count = ((out_x[1]-out_x[0])//out_size[0])*((out_y[1]-out_y[0])//out_size[1])
    while y[0] < out_y[1]:
        x = np.array([out_x[0], out_x[0]+out_size[0]])
        while x[0] < out_x[1]:
            count += 1
            print(f"{count} of {max_count}")

            # shift y
            y -= 18000

            # find corresponding input cells
            X = (x-1048576)//65536
            Y = (6207960-y)//65536

            # check if there is any data
            files = []
            for i in range(Y[1], Y[0]-1, -1):
                for j in range(X[0], X[1]+1):
                    if 0 <= i and i < 23 and 0 <= j and j < 16:
                        # copy in data from file
                        file = f"{ORIGINAL_DATA_ROOT}/{chr(ord('A')+i)}{chr(ord('A')+j)}.kea"
                        if Path(file).is_file():
                            files.append([i, j, file])
            if len(files) > 0:
                # load and output the data
                data = np.zeros([8192*(Y[0]-Y[1]+1), 8192*(X[1]-X[0]+1)])
                for row in files:
                    f = h5py.File(row[2], 'r')
                    in_data = np.array(f['BAND1']['DATA'])
                    in_data[in_data < -500.0] = 0.0  # replace nodata values
                    data[(row[0]-Y[1])*8192:(row[0]-Y[1]+1)*8192,
                         (row[1]-X[0])*8192:(row[1]-X[0]+1)*8192] = in_data

                # crop
                data = np.flipud(data)
                J = (x[0] - (1048576+X[0]*65536))//8
                I = (y[0] - (6207960-(Y[0]+1)*65536))//8
                data = data[I:I+out_n[1], J:J+out_n[0]]

                # shift y back
                y += 18000

                # output
                gzdfile = f"{OUTPUT_DATA_ROOT}/{gzd_name('NZT', dx, out_n[1], out_n[0], y[0]//1000, x[0]//1000)}"
                data.tofile(gzdfile)
            else:
                # shift y back
                y += 18000

            x += out_size[0]
        y += out_size[1]

    # clean up
    # rm_tree(Path(f"{ORIGINAL_DATA_ROOT}"))
    # Path(f"{ORIGINAL_DATA_ROOT}.zip").unlink()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', action='store', type=str,
                        help='data source (ost50, swt02, nzt08)')
    parser.add_argument('-r', '--resample', action='store', type=int,
                        help='resample to the specified resolution (integer)')
    args = parser.parse_args()

    if args.source is not None:
        if args.source == 'ost50':
            process_ost50()
        elif args.source == 'swt02':
            process_swt02()
        elif args.source == 'nzt08':
            process_nzt08()
        else:
            print(f"ERROR: source {args.source} not recognised")

    if args.resample is not None:
        # check if data exists
        # resample
        pass
