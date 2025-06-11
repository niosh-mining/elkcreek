"""Things for acting on the catalog"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def apparent_volume(
    moment: float | Iterable[float], **kwargs
) -> float | Iterable[float]:
    """
    Calculate the apparent volume for an event or sequence of events

    Parameters
    ----------
    moment
        Seismic moment of the event(s)
    apparent_stress
        Apparent stress of the event(s). Only used for V_A = M/(2*sigma_A).
    energy
        Radiated seismic energy of the event(s). Must also specify shear_modulus.
         Used for V_A = M^2/(2*mu*E).
    shear_modulus
        Shear modulus at the source(s). Must also specify energy.
        Used for V_A = M^2/(2*mu*E).

    Notes
    -----
    This is units-agnostic, but verify that the units of the input values will
    give you what you expect.
    """
    if "apparent_stress" in kwargs:
        stress: float | Iterable[float] = kwargs.pop("apparent_stress")
        if len(kwargs):
            raise ValueError(f"Unused keyword arguments: {kwargs.keys()}")
        # breakpoint()
        return moment / (2 * stress)
    else:
        energy = kwargs.pop("energy")
        shear_mod = kwargs.pop("shear_modulus")
        if len(kwargs):
            raise ValueError(f"Unused keyword arguments: {kwargs.keys()}")
        return moment**2 / (2 * shear_mod * energy)


def moment_magnitude(moment):
    """Calculate the moment magnitude."""
    return 2 / 3 * np.log10(moment) - 6.03


def local_magnitude(moment, energy):
    """Calculate the local magnitude."""
    return 0.272 * np.log10(energy) + 0.392 * np.log10(moment) - 4.63


def filter_event_df(
    df: pd.DataFrame,
    burst_events: pd.DataFrame,
    max_rocksigma_location_residual: float = 80,
    max_ims_location_residual: float = 30,
    x_range: tuple[float, float] = (9800, 12800),
    y_range: tuple[float, float] = (4200, 6500),
    z_range: tuple[float, float] = (1400, 2100),
) -> pd.DataFrame:
    """Filter out poor quality events"""
    mask = df["event_status"] != "Reject"
    mask = mask & (
        (
            (df["event_status"] == "RockSigma")
            & (df["location_residual"] < max_rocksigma_location_residual)
        )
        | (
            df["event_status"].isin(["Auto", "Manual"])
            & (df["location_residual"] < max_ims_location_residual)
        )
        | (df["location_residual"].isnull())
    )
    mask = mask & ((df["x"] >= x_range[0]) & (df["x"] <= x_range[1]))
    mask = mask & ((df["y"] >= y_range[0]) & (df["y"] <= y_range[1]))
    mask = mask & ((df["z"] >= z_range[0]) & (df["z"] <= z_range[1]))
    # Make sure to explicitly keep the bump events...
    mask = mask | (df["time"].isin(burst_events["time"]))
    return df[mask]


def get_outliers(
    df,
    high_apparent_stress=1,  # MPa
    high_apparent_volume=10**8.5,  # m^3
    large_source_radius=700,  # m
    big_local_mag=2,
):
    """Get outlier and large event catalogs."""
    # Do outliers first
    app_stress_outliers = df.loc[(df.apparent_stress > high_apparent_stress)]
    app_stress_outliers["event_status"] = "High Apparent Stress"
    app_volume_outliers = df.loc[(df.apparent_volume > high_apparent_volume)]
    app_volume_outliers["event_status"] = "High Apparent Volume"
    source_radius_outliers = df.loc[(df.source_radius > large_source_radius)]
    source_radius_outliers["event_status"] = "Big Source Radius"
    outliers = pd.concat(
        [app_stress_outliers, app_volume_outliers, source_radius_outliers]
    )

    # Next do large magnitude events
    big = df.loc[df.local_mag >= big_local_mag]

    return outliers, big
