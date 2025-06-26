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


def compile_daily_face_positions(
    start: str | np.datetime64,
    end: str | np.datetime64,
    longwall_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Get longwall face positions in one-day intervals

    Parameters
    ----------
    start
        Start of the time window over which to get the face positions
    end
        End of the time window over which to get the face positions
    longwall_df
        DataFrame containing the known face positions

    Returns
    -------
    Daily face positions
    """
    start = np.datetime64(start)
    end = np.datetime64(end)
    t = start
    times = []
    while t < end:
        times.append(t)
        t = t + np.timedelta64(1, "D")
    face_positions = get_longwall_positions(pd.Series(times), longwall_df)
    face_positions["date"] = times
    return face_positions


def get_date_from_face_position(face_positions, reference_positions, ref_point="headgate") -> list[np.datetime64] | np.datetime64:

    xcol = f"{ref_point}_x"
    ycol = f"{ref_point}_y"

    def _grab_date(fp) -> np.datetime64:
        dist = np.sqrt((reference_positions[xcol] - fp[xcol])**2 + (reference_positions[ycol] - fp[ycol])**2)
        return reference_positions.loc[dist.idxmin()].date

    if isinstance(face_positions, pd.DataFrame):
        dates = []
        for _, loc in face_positions.iterrows():
            dates.append(_grab_date(loc))
        return dates
    else:
        return _grab_date(face_positions)



def read_longwall_df(pth: Path) -> pd.DataFrame:
    """ Read a CSV containing information about longwall face positions """
    return pd.read_csv(pth).assign(
        local_time=lambda x: pd.to_datetime(x["local_date"])
    )
