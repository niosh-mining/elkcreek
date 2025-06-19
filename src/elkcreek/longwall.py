""" Functions for working with the longwall data """

from pathlib import Path

import numpy as np
import pandas as pd


def get_longwall_positions(times: pd.Series, lw_df: pd.DataFrame, end_buffer: int = 7) -> pd.DataFrame:
    """
    Get a dataframe of linearly extrapolated longwall positions.

    Parameters
    ----------
    times
        An array of datetime objects
    lw_df
        The dataframe which specifies the longwall position as a function
        of time.
    end_buffer
        Number of days to add to end range for each panel. This allows events
        Occurring slightly after panel has finished to still be attributed to
        the final longwall position.

    Returns
    -------
    Dataframe of longwall positions
    """
    # init output
    out_cols = {  # Need to explicitly set the dtypes because pandas is obnoxious otherwise
        "headgate_x": np.float64,
        "headgate_y": np.float64,
        "tailgate_x": np.float64,
        "tailgate_y": np.float64,
        "panel": str,
    }
    out = pd.DataFrame(columns=out_cols.keys(), index=times.index)
    buff_dt = np.timedelta64(end_buffer, "D")

    # iterate each panel, filter on events in panel range and extrapolate.
    for panel, panel_df in lw_df.groupby("panel"):
        min_time = panel_df["local_time"].values.min()
        max_time = panel_df["local_time"].values.max() + buff_dt
        in_panel_time = (times >= min_time) & (times <= max_time)
        if not in_panel_time.sum():
            continue
        times_filtered = times[in_panel_time]
        times_ns = times_filtered.astype(np.int64)
        panel_times_ns = panel_df["local_time"].values.astype(np.int64)
        for col in out_cols:
            if col == "panel":
                continue
            interps = np.interp(times_ns, panel_times_ns, panel_df[col])
            out.loc[times_filtered.index.values, col] = interps
        out.loc[times_filtered.index.values, "panel"] = panel
    out.loc[out.panel.isnull(), "panel"] = ""
    return out.astype(out_cols)


def read_longwall_df(pth: Path) -> pd.DataFrame:
    return pd.read_csv(pth).assign(
        local_time=lambda x: pd.to_datetime(x["local_date"])
    )
