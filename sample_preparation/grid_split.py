import argparse
import os
from itertools import product

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", type=str, help="Input file", required=True)
parse.add_argument("-o", "--output", type=str, help="Output file", required=True)

args = parse.parse_args()

input_shapefile = args.input
output_dir = args.output

ext = gpd.read_file(input_shapefile)

crs_ = ext.crs

### This is specifically for this case
# get the height using the perimeter of a square equation.
h = (int(ext.geometry.length) / 4) / 10


def make_grid(polygon, edge_size):
    """
    polygon : shapely.geometry
    edge_size : length of the grid cell
    source: https://stackoverflow.com/a/68778560/9948817
    """

    bounds = polygon.bounds
    x_coords = np.arange(bounds[0] + edge_size / 2, bounds[2], edge_size)
    y_coords = np.arange(bounds[1] + edge_size / 2, bounds[3], edge_size)
    combinations = np.array(list(product(x_coords, y_coords)))
    squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(
        edge_size / 2, cap_style=3
    )
    return gpd.GeoSeries(squares[squares.intersects(polygon)], crs=crs_)


grid = make_grid(ext.geometry[0], h)

# margin
for i in range(len(grid)):
    grid.iloc[i] = grid.iloc[i].buffer(-200)

# save the grid to file as shapefile
grid.to_file(os.path.join(output_dir, "data/grid/grid.shp"))
