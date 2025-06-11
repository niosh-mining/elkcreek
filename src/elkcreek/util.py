"""Utilities for Elk Creek."""

import pandas as pd
from pyproj import CRS, Transformer


def add_local_time_to_df(df, tz, time_column=None):
    """Add a column called local time."""
    col_name = "time" if time_column is None else time_column
    df["local_time"] = (
        pd.to_datetime(df[col_name])
        .dt.tz_localize("UTC")
        .dt.tz_convert(tz)
        .dt.tz_localize(None)
    )
    return df


def add_latitude_longitude_to_df(df, from_crs):
    """Using mine coordinates, add latitude and longitude."""
    crs_lat_lon = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(from_crs, crs_lat_lon)
    xy = df[["x", "y"]].values
    latitude, longitude = transformer.transform(xy[:, 0], xy[:, 1])
    return df.assign(latitude=latitude, longitude=longitude)


def read_event_directory(path):
    """Read all parquet event files in a directory."""
    assert path.exists() and path.is_dir(), f"{path} is not a directory."
    out = []
    for sub in path.glob("*.parquet"):
        out.append(pd.read_parquet(sub))
    df = pd.concat(out).sort_values(by="Date")
    return df
