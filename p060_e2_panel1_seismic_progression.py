""" Get a gridded spatial event count of all of the events """

from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from elkcreek.grid import Grid
from elkcreek.longwall import read_longwall_df, get_longwall_positions, get_date_from_face_position, compile_daily_face_positions
from elkcreek.plot import gridplot, plot_workings, plot_burst, plot_faults, plot_anomalous, plot_scale_bar, set_extents, configure_font_sizes

from p040_spatial_event_count import spatial_event_count
import local


def plot_over_advance_window(
    face_start: pd.Series,
    face_end: pd.Series,
    df: pd.DataFrame,
    bursts: pd.DataFrame,
    ax: plt.Axes,
    extents: dict[str, list[float]],
    reference_positions: pd.DataFrame,
    cbar_ax: str | plt.Axes | None = "same",
    show_burst_face = False
) -> tuple[plt.Axes, Grid]:

    def _plot_face(pos):
        ax.plot((pos.headgate_x, pos.tailgate_x), (pos.headgate_y, pos.tailgate_y), c=local.cp_hex[0])

    # Get the dates corresponding to the face positions
    start = get_date_from_face_position(face_start, reference_positions)
    end = get_date_from_face_position(face_end, reference_positions)
    print(start, end)

    # Get events from the specified date range
    eves = df.loc[(df.time >= start) & (df.time < end)]

    # Create the spatial plot
    ax, grid = spatial_event_count(eves, bursts, ax=ax, extents=extents, spacing=local.stat_grid_spacing, simplified_workings=False, cbar_ax=cbar_ax, show_burst_face=show_burst_face)

    # Add the face positions
    _plot_face(face_start)
    _plot_face(face_end)

    return ax, grid


def panel1_seismicity(df: pd.DataFrame, bursts: pd.DataFrame, longwall: pd.DataFrame) -> plt.Figure:

    # --- Set up the figure
    fig = plt.figure(figsize=(7, 9))
    gs = GridSpec(
        3,
        3,
        hspace=0.03,
        top=0.6,
        bottom=0.04,
    )
    windows = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
    ]

    # Now do the colorbar
    gs_cbar = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2, 1], height_ratios=[0.4, 0.1, 0.4])
    cax = fig.add_subplot(gs_cbar[1])

    # --- Do the thing

    # Generate a subplot spanning each time window
    grids = []
    faces = local.p1_face_pos
    for i in range(len(faces)-1):
        cbar_ax = None if i < (len(faces) - 2) else cax
        ax = windows[i]
        _, grid = plot_over_advance_window(faces.loc[i], faces.loc[i+1], df, bursts, ax=ax, extents=local.map_extents_event2, reference_positions=longwall, show_burst_face=False, cbar_ax=cbar_ax)
        grids.append(grid)

    # Manually set the color bar scale (and add subplot labels)
    vmax = max([x.data.max() for x in grids])
    for tw, a in zip(windows, ["a", "b", "c", "d", "e"]):
        mesh = tw.get_children()[0]
        mesh.set_clim([0, vmax])
        tw.text(-0.13, 1.02, f"({a})", transform=tw.transAxes)

    return fig


def main():
    # Load data
    df = pd.read_parquet(local.final_catalog)
    burst_df = pd.read_csv(local.burst_events, parse_dates=["time"])
    longwall_df = read_longwall_df(local.longwall_position_path)

    # Compile daily face positions over the time window of interest
    daily_face_positions = compile_daily_face_positions("2010-01-01", "2010-05-01", longwall_df)

    # Standardize fonts
    configure_font_sizes(local.font_sizes)

    # Make the figure
    fig = panel1_seismicity(df, burst_df, daily_face_positions)
    fig.savefig(local.e2_panel1_events_path, **local.savefig_params)


if __name__ == "__main__":
    main()
