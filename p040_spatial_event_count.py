""" Get a gridded spatial event count of all of the events """

from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from elkcreek.plot import gridplot, plot_workings, plot_burst, plot_faults, plot_anomalous, plot_scale_bar, set_extents

import local

def spatial_event_count(
    events: pd.DataFrame,
    bursts: pd.DataFrame,
    ax: plt.Axes,
    extents: dict[str, list[float]],
    spacing: float,
    simplified_workings: bool = False,
    vmax: float | None = None,
    scale_bar_params: dict[str, Any] = local.scale_bar_defaults,
    cbar_ax: plt.Axes | str | None = "same",
    show_anomalous: bool = True,
    show_burst_face: bool = True,
):
    """ Plot a count of events across a spatial grid """

    # Get a count of number of events in a grid
    cbar_params = {"orientation": "horizontal"}
    ax, grid = gridplot(
        ax,
        events,
        [extents["x"][0], extents["y"][0]],
        [extents["x"][1], extents["y"][1]],
        spacing=spacing,
        param="x",
        cbar_label="Number of Events",
        statistic="count",
        cbar_ax=cbar_ax,
        cbar_params=cbar_params,
        vmax=vmax,
    )
    # Plot the other geometry
    if not simplified_workings:
        plot_workings(local.dxfs["workings"], ax=ax)
    else:
        plot_workings(local.dxfs["workings_simplified"], ax=ax)
    plot_faults(local.dxfs["faults"], ax=ax, lw=1)
    # Plot the bursts and their damage zones
    color = local.cp_hex[1]
    for time in bursts.time:
        # Grab the key for the dxf file...
        time = str(time).replace(" ", "T").split(".")[0]
        plot_burst(time, local.dxfs["damage"], bursts, color, ax=ax, lw=1.5, include_event=False, include_face=show_burst_face)
    if show_anomalous:
        plot_anomalous(local.dxfs["anomalous"], ax=ax, edgecolor=local.cp_hex[2], alpha=1, lw=1.5)
    # Make it look nice
    plot_scale_bar(ax, **scale_bar_params)
    set_extents(ax, extents)
    return ax, grid


def spatial_count_version1(df, burst_df):
    """ Plot the spatial event count for ALL of the events during the deployment """
    fig = plt.figure(figsize=(7, 5))
    gs = GridSpec(
        2,
        1,
        hspace=0.02,
        height_ratios=[1, 0.05],
    )
    grid_ax = fig.add_subplot(gs[0])

    gs_cbar = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.01, width_ratios=[0.02, 0.95, 0.02])
    cbar_ax = fig.add_subplot(gs_cbar[1])

    spatial_event_count(df, burst_df, grid_ax, local.map_extents_zoomed, local.stat_grid_spacing, simplified_workings=False, cbar_ax=cbar_ax)
    return fig


def spatial_count_version2(df, burst_df):
    """ Plot the spatial event count broken up by panel """
    time_windows = {
        "Panel 1": [local.panel_dates.loc["panel1"].start, local.panel_dates.loc["p1_break"].end],
        "Panel 2": [local.panel_dates.loc["panel2"].start, local.panel_dates.loc["p2b_break"].end],
        "Panel 3": [local.panel_dates.loc["panel3"].start, local.panel_dates.loc["p3b_break"].end],
        "Panel 4": [local.panel_dates.loc["panel4"].start, local.panel_dates.loc["post_mining"].end],
    }

    fig = plt.figure(figsize=(7, 10))
    gs = GridSpec(
        3,
        2,
        hspace=0.02,
        wspace=0.09,
        height_ratios=[1, 1, 0.05],
        top=0.45,
        bottom=0.02,
    )
    axes = {
        "Panel 1": (fig.add_subplot(gs[0, 0]), "a"),
        "Panel 2": (fig.add_subplot(gs[0, 1]), "b"),
        "Panel 3": (fig.add_subplot(gs[1, 0]), "c"),
        "Panel 4": (fig.add_subplot(gs[1, 1]), "d"),
    }
    gs_cbar = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.01, width_ratios=[0.02, 0.95, 0.02])
    cbar_ax = fig.add_subplot(gs_cbar[1])

    scale_bar_params = dict(local.scale_bar_defaults)
    scale_bar_params.update({
        "loc": "upper right",
        "label_top": True,
    })

    grids = []
    added_cbar = False
    for panel, time_range in time_windows.items():
        sub = df.loc[(df.time > time_range[0]) & (df.time <= time_range[1])]
        burst_sub = burst_df.loc[(burst_df.time > time_range[0]) & (burst_df.time <= time_range[1])]

        if added_cbar:
            ax, grid = spatial_event_count(sub, burst_sub, axes[panel][0], local.map_extents_zoomed, local.stat_grid_spacing, simplified_workings=False, cbar_ax=None, scale_bar_params=scale_bar_params)
        else:
            ax, grid = spatial_event_count(sub, burst_sub, axes[panel][0], local.map_extents_zoomed, local.stat_grid_spacing, simplified_workings=False, cbar_ax=cbar_ax, scale_bar_params=scale_bar_params)
            added_cbar = True
        grids.append(grid)

    # Update the color limits to match for each plot
    vmax = max([x.data.max() for x in grids])
    for panel, (ax, subplot) in axes.items():
        mesh = ax.get_children()[0]
        mesh.set_clim([0, vmax])
        ax.set_title(panel)
        ax.text(-0.07, 1.05, f"({subplot})", transform=ax.transAxes)

    return fig


def main():
    # Load data
    df = pd.read_parquet(local.final_catalog)
    burst_df = pd.read_csv(local.burst_events, parse_dates=["time"])

    # Plot the original version of the figure
    version1 = spatial_count_version1(df, burst_df)
    version1.savefig(local.spatial_event_count_all, **local.savefig_params)

    # Plot the version broken up into subplots
    version2 = spatial_count_version2(df, burst_df)
    version2.savefig(local.spatial_event_count_by_panel, **local.savefig_params)


if __name__ == "__main__":
    main()
