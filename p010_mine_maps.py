"""
Plot the main narrative map.
"""

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
from elkcreek.plot import (
    magnitude_scaling,
    plot_burst,
    plot_events,
    plot_faults,
    plot_instrumentation_sites,
    plot_overburden,
    plot_scale_bar,
    plot_stations,
    plot_workings,
    set_extents,
)

import local


def panel_labels(ax):
    """Add labels to panels."""
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "top",
        "fontsize": 10,
        "rotation": -14.5,
    }

    ax.text(11000, 4770, "Panel 1", **text_props)
    ax.text(10450, 5280, "Panel 2", **text_props)
    ax.text(11925, 4900, "Panel 2b", **text_props)
    ax.text(10400, 5675, "Panel 3", **text_props)
    ax.text(11200, 5530, "Panel 3b", **text_props)
    ax.text(10500, 6175, "Panel 4", **text_props)
    ax.text(11650, 5875, "Panel 4b", **text_props)


def event_labels(ax):
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "fontsize": 10,
        "bbox": {"edgecolor": "k", "facecolor": "white", "alpha": 0.8},
    }

    ax.text(11580, 5550, "1", **text_props)
    ax.text(11550, 4950, "2", **text_props)
    ax.text(10800, 5550, "3", **text_props)
    ax.text(11000, 6320, "4", **text_props)
    ax.text(11500, 6150, "5", **text_props)


def instrument_site_labels(ax):
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "fontsize": 10,
        "bbox": {"edgecolor": "k", "facecolor": "white", "alpha": 0.8},
        "rotation": -14.5,
    }

    ax.text(10360, 4750, "A", **text_props)
    ax.text(11660, 4400, "B", **text_props)
    ax.text(10700, 5000, "C", **text_props)
    ax.text(11550, 4850, "D", **text_props)


def sub_instrument_site_labels(ax):
    """Add labels to sub instrument sites (eg can sites for D)"""
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "fontsize": 10,
        "rotation": -14.5,
    }
    peffects = [
        path_effects.Stroke(linewidth=3, foreground="white"),
        path_effects.Normal(),
    ]

    labels = {"d1": (11706.6, 4742.4), "d2": (11735.9, 4734.7)}
    for label, coord in labels.items():
        txt = ax.text(coord[0], coord[1], label, **text_props)
        txt.set_path_effects(peffects)


def burst_location_map(bursts):
    """Create the mine map with overburden, faults, and burst locations"""
    ob_contours = [x * 100 for x in range(11)]
    ob_plot_kwargs = {
        "figsize": (7, 4.5),
    }
    fig, ax = plot_overburden(
        local.dxfs["overburden"],
        ob_contours,
        local.lower_left,
        local.upper_right,
        local.spacing,
        plotting_kwargs=ob_plot_kwargs,
    )
    plot_workings(local.dxfs["workings"], ax=ax)
    plot_faults(local.dxfs["faults"], ax=ax)
    # Plot the bursts and their corresponding damage zones
    for time, color in local.burst_colors.items():
        plot_burst(time, local.dxfs["damage"], bursts, color, ax=ax)

    panel_labels(ax)
    event_labels(ax)
    plot_scale_bar(ax, **local.scale_bar_defaults)
    set_extents(ax, local.map_extents_zoomed)

    fig.tight_layout()
    return fig


def instrumentation_location_map(bursts, stations):
    """Create the mine map with seismic stations and instrument sites"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_workings(local.dxfs["workings"], ax=ax)
    plot_events(bursts, c="#999ccc", ax=ax, s=magnitude_scaling(bursts))
    plot_instrumentation_sites(local.dxfs["instrumentation"], ax=ax)
    plot_stations(
        stations,
        local.station_groups,
        local.color_palette,
        ax=ax,
        legend=dict(loc="upper right"),
    )

    panel_labels(ax)
    instrument_site_labels(ax)
    sb_params = local.scale_bar_defaults.copy()
    sb_params.update(
        {
            "dist": 1000,
            "unit": "1 km",
            "label_override": True,
        }
    )

    plot_scale_bar(ax, **sb_params)
    set_extents(ax, local.map_extents)

    fig.tight_layout()
    return fig


def plot_instruments(ax, df):
    """Plot/label instruments to match other plot colors."""
    bpc = df[df["sensor"].str.startswith("BP")]
    for _, row in bpc.iterrows():
        color = local.bpc_colors[row["sensor"]]
        ax.plot(row["easting"], row["northing"], "o", color=color)
        ax.text(
            row["easting"] + 2,
            row["northing"],
            row["sensor"],
            color=color,
            verticalalignment="center",
            horizontalalignment="left",
            fontsize=12,
        )
    return ax


def event_2_instrument_location_map(bursts, inst_df):
    """Create a zoomed-in map around burst 2 showing cans/BPCs."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_workings(local.dxfs["workings"], ax=ax)
    plot_events(bursts, c="#999ccc", ax=ax, s=magnitude_scaling(bursts))
    plot_instrumentation_sites(
        local.dxfs["instrumentation"],
        ax=ax,
        site_slice=slice(-2, None),
    )

    for time, color in local.burst_colors.items():
        plot_burst(time, local.dxfs["damage"], bursts, color, ax=ax)

    sb_params = local.scale_bar_defaults.copy()
    sb_params.update(
        {
            "dist": 50,
            "unit": "50 m",
            "label_override": True,
            "size_vertical": 1.5,
        }
    )

    plot_scale_bar(ax, **sb_params)
    set_extents(ax, local.map_extents_event_2)

    sub_inst = inst_df[inst_df["location"] == "2N Outby"]
    plot_instruments(ax, sub_inst)

    # Label can sub-sites.
    sub_instrument_site_labels(ax)

    return fig


def main():
    bursts = pd.read_csv(local.burst_events, parse_dates=["time"])
    stations = pd.read_csv(local.station_file)
    inst_df = pd.read_csv(local.instrumentation_file)

    burst_fig = burst_location_map(bursts)
    burst_fig.savefig(local.burst_map, dpi=300)

    inst_fig = instrumentation_location_map(bursts, stations)
    inst_fig.savefig(local.station_map, dpi=300)

    fig = event_2_instrument_location_map(bursts, inst_df)
    fig.savefig(local.event_2_zoomed_in_map, dpi=300)


if __name__ == "__main__":
    main()
