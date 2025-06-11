"""
Add volume specific analysis regions.
"""

import json

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

import local


def sva_to_delaunay(sva):
    """Convert sva to hulk"""
    members = sva["members"]
    assert len(members) == 1, "only works with len 1 members."
    first = members[0]
    origin = np.array(first["origin"])
    x_vals = np.array(first["nodes_x"]) + origin[0]
    y_vals = np.array(first["nodes_y"]) + origin[1]
    z_vals = np.ones_like(x_vals) + origin[2]
    points = np.array([x_vals, y_vals, z_vals]).T
    extension = np.array(first["direction"]) * first["cylinder_length"]
    new_points = points + extension
    volume_points = np.concatenate([points, new_points], axis=0)
    return Delaunay(volume_points)


def in_sva(df, sva):
    """Determine which events are in the sva."""
    delaunay = sva_to_delaunay(sva)
    xyz = df[["x", "y", "z"]].values
    in_hull = delaunay.find_simplex(xyz) >= 0
    return in_hull


def main():
    """Add columns for each volume of interest."""
    df = pd.read_parquet(local.cat_path_local_info)
    vols = json.load(local.analysis_volume_path.open())
    for name, sva_def in vols.items():
        df[f"vsa_{name}"] = in_sva(df, sva_def)
    df.to_parquet(local.cat_path_with_volumes)


if __name__ == "__main__":
    main()
