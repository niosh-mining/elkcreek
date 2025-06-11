"""
Makes the pyrocko inventory
"""

import pandas as pd
from pyrocko import model

import local


def get_surface_stations(df_path=local.seismic_station_with_lat_lon_path):
    """Get only the surface stations as a DF."""
    # Only use surface stations
    df = pd.read_parquet(df_path)
    df = df[~df["underground"]]
    ug_stations = set(df["name"].values)
    return ug_stations


def main():
    """Make the pyrocko inventory."""
    stations = []
    surface_stations = get_surface_stations()
    df = pd.read_parquet(local.seismic_station_with_lat_lon_path)

    for _, row in df.iterrows():
        elevation = row["z"] - local.grond_datum
        channels = []
        if row["name"] not in surface_stations:
            continue
        for chan_code in "ZNE":
            cha = model.Channel(chan_code)
            channels.append(cha)
        sta = model.Station(
            lat=row["latitude"],
            lon=row["longitude"],
            station=row["name"],
            network="EC",
            location="00",
            channels=channels,
            elevation=elevation,
        )

        stations.append(sta)

    model.dump_stations(stations, local.pyrocko_station_path)


if __name__ == "__main__":
    main()

