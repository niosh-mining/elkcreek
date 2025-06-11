"""
Add the longwall information/extrapolations.
"""

import numpy as np
import pandas as pd

import local


def get_longwall_positions(times, lw_df, end_buffer=7) -> pd.DataFrame:
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
    """
    # init output
    out_cols = ["headgate_x", "headgate_y", "tailgate_x", "tailgate_y", "panel"]
    out = pd.DataFrame(columns=out_cols, index=times.index)
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
    return out


def l2_normalize(array):
    """Normalize rows by l2 norm."""
    array = (array.values if hasattr(array, "values") else array).astype(np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / norms


def add_distances(df, lw_df):
    """Add Shortest distance to the longwall (xy)"""
    # need to only use rows that have panel locations.
    sub = df[~df["panel"].isnull()]
    # First get a df of the first longwall position in each panel.
    first_pan = lw_df.sort_values("local_time").groupby("panel").first()
    # Expand the first panel df.
    expanded = first_pan.loc[sub["panel"].values]
    first_hgs = expanded[["headgate_x", "headgate_y"]].values
    # Now get vector quantities from dataframe.
    hg_xy = sub[["headgate_x", "headgate_y"]].values.astype(np.float64)
    tg_xy = sub[["tailgate_x", "tailgate_y"]].values.astype(np.float64)
    lw_length = np.linalg.norm(hg_xy - tg_xy, axis=1)
    # tg_norm = l2_normalize(tg_xy - first_tgs)
    lw_normal = l2_normalize(hg_xy - first_hgs)  # vector perp to lw
    lw_parallel = l2_normalize(hg_xy - tg_xy)  # vector parallel to lw
    event_xy = sub[["x", "y"]].values.astype(np.float64)
    origin_xy = sub[["lw_center_x", "lw_center_y"]].values.astype(np.float64)
    event_vector_from_center = event_xy - origin_xy
    # Get distance perpendicular to longwall
    perp_lw_dist = np.einsum("ij,ij->i", event_vector_from_center, lw_normal)
    # Get distance in longwall line to hg and tg
    tg_dist = np.abs(np.einsum("ij,ij->i", event_xy - tg_xy, lw_parallel))
    hg_dist = np.abs(np.einsum("ij,ij->i", event_xy - hg_xy, lw_parallel))
    # Determine if event is outside the panel
    outside_panel = (tg_dist > lw_length) | (hg_dist > lw_length)
    on_tg = outside_panel & (tg_dist < hg_dist)
    on_hg = outside_panel & (tg_dist > hg_dist)
    total_lw_dist = np.abs(perp_lw_dist)
    total_lw_dist[on_tg] = np.linalg.norm(event_xy[on_tg] - tg_xy[on_tg], axis=1)
    total_lw_dist[on_hg] = np.linalg.norm(event_xy[on_hg] - tg_xy[on_hg], axis=1)

    lw_label = pd.Series([""] * len(sub), index=sub.index)
    lw_label[on_tg] = "tailgate"
    lw_label[on_hg] = "headgate"
    lw_label[~outside_panel] = "panel"

    out = pd.DataFrame(index=sub.index).assign(
        on_tailgate_side=on_tg,
        on_headgate_side=on_hg,
        tailgate_distance=np.linalg.norm(event_xy - tg_xy, axis=1),
        headgate_distance=np.linalg.norm(event_xy - hg_xy, axis=1),
        perp_lw_distance=perp_lw_dist,
        total_lw_distance=total_lw_dist,
        longwall_label=lw_label,
    )
    return df.join(out)


def _add_center_panel(df):
    """Add the center panel positions."""
    hgx, hgy = df["headgate_x"], df["headgate_y"]
    tgx, tgy = df["tailgate_x"], df["tailgate_y"]
    out = df.assign(lw_center_x=(hgx + tgx) / 2, lw_center_y=(hgy + tgy) / 2)
    return out


def rotate_2d_array(array, rotation):
    """Apply a rotation around z axis for 2d array."""
    assert array.shape[1] == 2
    rotation = np.array(rotation)
    assert len(rotation) == len(array) or len(rotation) == 1
    old_x = array[:, 0]
    old_y = array[:, 1]

    new_x = old_x * np.cos(rotation) - old_y * np.sin(rotation)
    new_y = old_x * np.sin(rotation) + old_y * np.cos(rotation)
    return np.array([new_x, new_y]).T


def _add_lw_centered_event_positions(df):
    """Add event positions with longwall center (aligned with x axis) as center."""
    origin = df[["lw_center_x", "lw_center_y"]].values
    tailgate = df[["tailgate_x", "tailgate_y"]].values
    rotation_vector = l2_normalize(tailgate - origin)
    rot_angle = np.arctan2(rotation_vector[:, 1], rotation_vector[:, 0])
    event_loc = df[["x", "y"]].values
    # get un-rotated event location with lw center as origin
    event_vect = event_loc - origin
    # rotate so lw is aligned with x axis.
    out = rotate_2d_array(event_vect, -rot_angle)

    return df.assign(
        location_x_longwall_coord=out[:, 0],
        location_y_longwall_coord=out[:, 1],
    )


def add_longwall_info(df, lw_df, time_column="local_time", spatial_columns=("x", "y")):
    """Add the longwall information to a dataframe."""
    lw_df = lw_df.pipe(_add_center_panel)
    positions = get_longwall_positions(df["local_time"], lw_df=lw_df).pipe(
        _add_center_panel
    )
    out = (
        df.join(positions)
        .pipe(add_distances, lw_df=lw_df)
        .pipe(_add_lw_centered_event_positions)
    )
    return out


def main():
    """Add the longwall distance/location."""
    lw_df = pd.read_csv(local.longwall_position_path).assign(
        local_time=lambda x: pd.to_datetime(x["local_date"])
    )
    df = pd.read_parquet(local.cat_path_with_volumes)
    out = add_longwall_info(df, lw_df)
    out.to_parquet(local.cat_path_with_longwall)


if __name__ == "__main__":
    main()
