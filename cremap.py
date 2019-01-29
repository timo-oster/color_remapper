#!/usr/bin/env python3
import argparse
import png
import numpy as np
import csv
from matplotlib import pyplot as plt


# map a scalar value to a color from a colormap
def map_to_color(scalar, colormap):
    # search in list to find scalar
    lo = int(0)
    hi = int(colormap.shape[0] - 1)
    while hi - lo > 1:
        c = (hi + lo) // 2
        if colormap[c, 0] <= scalar:
            lo = c
            continue
        else:
            hi = c
            continue

    # import pdb; pdb.set_trace()
    # interpolate color from colormap
    clo = colormap[lo, 1:]
    slo = colormap[lo, 0]
    chi = colormap[hi, 1:]
    shi = colormap[hi, 0]
    t = (scalar - slo) / (shi - slo)
    return (1 - t) * clo + t * chi


# map a color from a colormap to a scalar
# if no color within tolerance tol is found, return none
def map_to_scalar(color, colormap, tol):
    # import pdb; pdb.set_trace()
    # search in list to find color
    for i in range(colormap.shape[0] - 1):
        clo = colormap[i, 1:]
        chi = colormap[i + 1, 1:]
        cd = chi - clo
        cc = color - clo
        t = np.dot(cd, cc) / (np.linalg.norm(cd)**2)
        d = np.linalg.norm(cc - t * cd)
        if t >= 0 and t < 1 and d <= tol:
            slo = colormap[i, 0]
            shi = colormap[i + 1, 0]
            return (1 - t) * slo + t * shi
    return None


# map a color from one color map to another
def map_color(color, cmap_in, cmap_out, tol):
    scalar = map_to_scalar(color, cmap_in, tol)
    if scalar is None:
        return color
    return map_to_color(scalar, cmap_out)


def normalize_cmap(cmap, dtype=None, invert=False):
    """normalize the color values to 0...1 depending on data type and add a
        scalar column with equidistant values if none is present."""
    (rows, cols) = cmap.shape
    # add scalar column if necessary
    if cols == 3:
        cmap = np.concatenate(([[i] for i in range(rows)], cmap), 1)
    # detect datatype
    if dtype is None:
        dtype = 'int' if np.max(cmap[:, 1:]) > 1 else 'float'
    # normalize int values to 0...1
    if dtype == 'int':
        cmap[:, 1:] = cmap[:, 1:] / 255.0

    # invert colormap if necessary
    if invert:
        cmap[:, 0] = -cmap[:, 0]
        cmap = np.flip(cmap, 0)
    # normalize the scale in the first column to 0-1
    min_scal = np.min(cmap[:, 0])
    max_scal = np.max(cmap[:, 0])
    cmap[:, 0] = (cmap[:, 0] - min_scal) / (max_scal - min_scal)
    return cmap

def read_cmap(filename, invert=False, delimiter=None, dtype=None):
    with open(filename) as csvfile:
        # infer csv format from file
        dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=delimiter)
        csvfile.seek(0)
        r = csv.reader(csvfile, dialect)
        dcmap = np.vstack(map(np.double, r))
        # Check for correct number of rows and columns
        (rows, cols) = dcmap.shape
        if cols < 3 or cols > 4:
            print("Error reading csv file \"{}\": detected {} columns. ".format(filename, cols) +
                  "Valid csv files need to have 3 columns (r, g, b) or 4 columns (scalar, r, g, b).")
        if rows < 2:
            print("Error reading csv file \"{}\": detected {} rows. ".format(filename, rows) +
                  "I need at least 2 rows to construct a colormap.")
        # normalize colors and ensure scalar column
        dcmap = normalize_cmap(dcmap, dtype=dtype, invert=invert)
        return dcmap


def read_img(filename):
    # read png image
    r = png.Reader(filename)
    (rows, cols, pngdata, meta) = r.asDirect()
    image_2d = np.vstack(map(np.uint16, pngdata))
    image_3d = np.reshape(image_2d,
                          (cols, rows, meta['planes']))
    return np.double(image_3d) / 255


def write_img(img, filename):
    img_16 = (np.floor(img * 255)).astype(np.uint16)
    png.from_array(img_16.tolist(), mode='RGB').save(filename)


def remap_img(img, cmap_in, cmap_out, tol):
    return np.apply_along_axis(lambda c: map_color(c, cmap_in, cmap_out, tol),
                               2,
                               img)


def main():

    # todo: read (and write) different image file formats?
    # todo: what to do with alpha channel?
    #       - in input image
    #       - in input colormap
    #       - in output colormap
    # todo: how to make it faster?

    # Parameters:
    # - Input file
    # - Output file
    # - Input colormap
    # - Output colormap
    # - Tolerance for reverse color lookup
    # - color format (float or byte)
    # - colormap reading options
    #   - separator
    #   - skip first line
    #   - with or without scalar in column
    #   - with or without alpha channel
    #   - which column for scalar
    # - colormap transformation options
    #   - leave scalars untouched (if any) or normalize to [0 1]

    parser = argparse.ArgumentParser(description="Remap colors of a " +
        "color-mapped image to another color map. \n"+
        "Input and output color maps are specified as csv files with " +
        "three columns for r, g, b and an optional first column specifying " +
        "the position of the color in the color map.")

    parser.add_argument("input", help="Input image")
    parser.add_argument("cmap_in", help="Input colormap (as csv file)")
    parser.add_argument("cmap_out", help="Output colormap (as csv file)")
    parser.add_argument("output", help="Output image", nargs='?',
                        default="out.png")

    parser.add_argument("-t", "--tolerance", type=float,
                        help="Tolerance for reverse color lookup",
                        default=0.01)
    parser.add_argument("-d", "--color-dtype",
                        help="Data type for color values in the csv files"+
                             " (float 0...1 or int 0...255)." +
                             " Estimated automatically by default.",
                        choices=['float', 'int'])
    parser.add_argument("-s", "--separator",
                        help="Separator for elements in the csv file",
                        default=',')
    parser.add_argument("-i", "--invert", help="Invert the output color map",
                        action='store_true')

    args = parser.parse_args()

    img = read_img(args.input)

    # remove alpha channel
    if img.shape[2] > 3:
        img = img[:, :, 0:3]

    cmap_in = read_cmap(args.cmap_in,
                        delimiter=args.separator,
                        dtype=args.color_dtype)

    cmap_out = read_cmap(args.cmap_out,
                         invert=args.invert,
                         delimiter=args.separator,
                         dtype=args.color_dtype)

    img_r = remap_img(img, cmap_in, cmap_out, args.tolerance)

    write_img(img_r, args.output)


if __name__ == "__main__":
    main()
