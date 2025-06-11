"""
Add local time and latitude/longitude to event catalog.
"""

import pandas as pd
import pyproj
from pyproj import CRS, Transformer

import local

pyproj.network.set_network_enabled(
    active=False
)  # Make sure pyproj isn't unnecessarily connecting to the internet


def add_local_time(df, tz=local.time_zone, time_column=None):
    """Add a column called local time."""
    if time_column is None:
        col_name = "time" if "time" in df.columns else "p_time"
    else:
        col_name = time_column
    df["local_time"] = (
        pd.to_datetime(df[col_name])
        .dt.tz_localize("UTC")
        .dt.tz_convert(tz)
        .dt.tz_localize(None)
    )
    return df


def add_latitude_longitude(df, from_crs):
    """Using mine coordinates, add latitude and longitude."""
    crs_lat_lon = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(from_crs, crs_lat_lon)
    xy = df[["x", "y"]].values
    latitude, longitude = transformer.transform(xy[:, 0], xy[:, 1])
    return df.assign(latitude=latitude, longitude=longitude)


def main():
    """Add local time and latitude/longitude to event catalog."""
    crs_elk_creek = CRS.from_wkt(local.wkt_path.read_text())

    df = (
        pd.read_parquet(local.cleaned_cat_path)
        .pipe(add_local_time)
        .pipe(add_latitude_longitude, from_crs=crs_elk_creek)
    )

    df.to_parquet(local.cat_path_local_info)


if __name__ == "__main__":
    main()
