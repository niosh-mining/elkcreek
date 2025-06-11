"""
Add latitude and longitude to the stations.
"""

import pandas as pd
from elkcreek.util import add_latitude_longitude_to_df
from pyproj import CRS

import local


def main():
    """Add latitude and longitude to the stations."""
    crs = CRS.from_wkt(local.wkt_path.read_text())

    df = pd.read_csv(local.station_file).pipe(
        add_latitude_longitude_to_df, from_crs=crs
    )
    df.to_parquet(local.seismic_station_with_lat_lon_path)


if __name__ == "__main__":
    main()
