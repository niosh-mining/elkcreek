"""
Combine the IMS and Rocksigma catalogs.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from elkcreek.events import apparent_volume, local_magnitude, moment_magnitude
from elkcreek.util import read_event_directory
from obsplus.utils import to_datetime64

from local import combined_cat_path, ims_raw, rocksigma_raw

ims_col_map = {
    "Apparent Stress": "apparent_stress",
    "Apparent Volume": "apparent_volume",
    "Corner Frequency S": "corner_frequency",
    "Date": "time",
    "Dynamic Stress Drop": "dynamic_stress_drop",
    "Energy P": "energy_p",
    "Energy S": "energy_s",
    "Energy Total": "energy_total",
    "Imported Tag": "event_status",
    "Location X": "x",
    "Location Y": "y",
    "Location Z": "z",
    "Location Residual": "location_residual",
    "Mag Local": "local_mag",
    "Magnitude Moment": "moment_mag",
    "Moment Total": "moment_total",
    "Num Sensors Triggered": "num_sensors_triggered",
    "Num Sensors Used": "num_sensors_used",
    "Source Radius": "source_radius",
    "SP Energy Ratio": "sp_energy_ratio",
    "SP Moment Ratio": "sp_moment_ratio",
    "Static Stress Drop": "static_stress_drop",
}
rocksigma_col_map = {
    "Date": "time",
    "Location X": "x",
    "Location Y": "y",
    "Location Z": "z",
    "ML": "local_mag",
    "Classification Tag": "event_status",
    "Sensors Used": "num_sensors_used",
    "Location Error": "location_residual",
    "SP Ratio Energy": "sp_energy_ratio",
    "Moment Magnitud": "moment_mag",
    # QUESTION: what formulation does IMS use for source_radius?
    "Source Radius Brune (1970)": "source_radius",
    "Moment": "moment_total",
    "Apparent Stress": "apparent_stress",
    "Dynamic Stress Drop": "dynamic_stress_drop",
    "SP Ratio Moment": "sp_moment_ratio",
    "Static Stres Drop": "static_stress_drop",
    "S Wave Energy": "energy_s",
    "Corner Frequency": "corner_frequency",
    "totalRadiatedEnergy": "energy_total",
    "P Wave Energy": "energy_p",
    "Rupture Velocity": "rupture_velocity",
}
desired_columns = [
    "time",
    "x",
    "y",
    "z",
    "local_mag",
    "moment_mag",
    "event_status",
    "location_residual",
    "moment_total",
    "sp_moment_ratio",
    "energy_p",
    "energy_s",
    "energy_total",
    "sp_energy_ratio",
    "apparent_stress",
    "apparent_volume",
    "corner_frequency",
    "source_radius",
    "static_stress_drop",
    "dynamic_stress_drop",
    "num_sensors_triggered",
    "num_sensors_used",
    "rupture_velocity",  # Not sure how much confidence to put in this one
]


def _prep_df(df: pd.DataFrame, col_map: Mapping) -> pd.DataFrame:
    df = df.rename(columns=col_map)
    df["time"] = to_datetime64(df["time"])
    return df


def prep_ims_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare IMS data."""
    df["rupture_velocity"] = np.nan
    return _prep_df(df, ims_col_map)[desired_columns]


def prep_rocksigma_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare rocksigma catalog."""
    df["num_sensors_triggered"] = df[
        "Sensors Used"
    ]  # Not strictly true, but avoids dtype problems
    df = _prep_df(df, rocksigma_col_map)
    df["apparent_volume"] = apparent_volume(
        df["moment_total"], apparent_stress=df["apparent_stress"] * 10e6
    )
    df["event_status"] = df.event_status.map(
        {
            "e": "RockSigma",
            "n": "Reject",
        }
    )
    return df[desired_columns]


def main():
    """Combine catalogs together."""
    ims_raw_df = read_event_directory(ims_raw)
    rs_raw_df = read_event_directory(rocksigma_raw)

    ims = prep_ims_data(ims_raw_df)
    rs = prep_rocksigma_data(rs_raw_df)

    # Fill the gaps in the RockSigma data with IMS data
    mintime = rs.time.min()
    missing = ims.loc[ims.time < mintime]
    # Wille filled in most of this gap, but accidentally used
    # local time instead of UTC, so there's still a 7-hour chunk missing
    dec2 = ims.loc[
        (ims.time > np.datetime64("2012-12-02"))
        & (ims.time < np.datetime64("2012-12-02T07:00:00"))
    ]

    # Combine the two datasets (and check that it actually worked)
    combined = pd.concat([rs, missing, dec2]).sort_values(by="time")
    expected = {"Auto", "Manual", "RockSigma", "Reject"}
    assert set(combined["event_status"].unique()) == expected

    # Recalculate the magnitudes to make sure they're consistent
    combined["moment_mag"] = moment_magnitude(combined["moment_total"])
    combined["local_mag"] = local_magnitude(
        moment=combined["moment_total"], energy=combined["energy_total"]
    )

    # Output the result
    combined.to_parquet(combined_cat_path, index=False)


if __name__ == "__main__":
    main()
