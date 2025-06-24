"""
Plot the main narrative map.
"""

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd
from elkcreek.plot import (
    configure_font_sizes,
    fill_mined_areas,
    magnitude_scaling,
    plot_anomalous,
    plot_bpcs,
    plot_burst,
    plot_events,
    plot_faults,
    plot_instrumentation_sites,
    plot_north_arrow,
    plot_overburden,
    plot_scale_bar,
    plot_stations,
    plot_workings,
    set_extents,
)

import local


def panel_labels(ax):
    """Add panel names"""
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "top",
        "rotation": -14.5,
    }

    ax.text(11000, 4770, "Panel 1", **text_props)
    ax.text(10270, 5328, "Panel 2", **text_props)
    ax.text(11925, 4900, "Panel 2b", **text_props)
    ax.text(10400, 5675, "Panel 3", **text_props)
    ax.text(11200, 5530, "Panel 3b", **text_props)
    ax.text(10500, 6175, "Panel 4", **text_props)
    ax.text(11650, 5875, "Panel 4b", **text_props)


def event_labels(ax):
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "bbox": {"edgecolor": "k", "facecolor": "white", "alpha": 0.8},
    }

    ax.text(11580, 5550, "1", **text_props)
    ax.text(11550, 4950, "2", **text_props)
    ax.text(10800, 5550, "3", **text_props)
    ax.text(10650, 6220, "4", **text_props)
    ax.text(11500, 6150, "5", **text_props)


def instrument_site_labels(ax):
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "rotation": -14.5,
    }
    peffects = [
        patheffects.Stroke(linewidth=3, foreground="white"),
        patheffects.Normal(),
    ]

    def _add_label(x, y, label):
        txt = ax.text(x, y, label, **text_props)
        txt.set_path_effects(peffects)

    _add_label(10461.2343, 4648.9917, "A")
    _add_label(11777.5002, 4306.5869, "B")
    _add_label(10507.5620, 5023.8779, "C")
    _add_label(11633.9232, 4721.0132, "D")


def sub_instrument_site_labels(ax):
    """Add labels to sub instrument sites (eg can sites for D)"""
    text_props = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "fontsize": 10,
        "rotation": -14.5,
    }
    peffects = [
        patheffects.Stroke(linewidth=3, foreground="white"),
        patheffects.Normal(),
    ]

    labels = {"d1": (11706.6, 4742.4), "d2": (11735.9, 4734.7)}
    for label, coord in labels.items():
        txt = ax.text(coord[0], coord[1], label, **text_props)
        txt.set_path_effects(peffects)


def burst_location_map(bursts):
    """Create the mine map with overburden, faults, and burst locations"""
    fig, ax = plt.subplots(1 , figsize=(7, 4.6))

    ob_contours = [x * 100 for x in range(11)]
    plot_overburden(
        local.dxfs["overburden"],
        ob_contours,
        local.lower_left,
        local.upper_right,
        local.ob_grid_spacing,
        ax=ax,
    )
    plot_workings(local.dxfs["workings"], ax=ax)
    fill_mined_areas(local.dxfs["workings_simplified"], ax=ax)
    plot_faults(local.dxfs["faults"], ax=ax)
    # Plot the bursts and their corresponding damage zones
    for time, color in local.burst_colors.items():
        plot_burst(time, local.dxfs["damage"], bursts, color, ax=ax)
    plot_anomalous(
        local.dxfs["anomalous"], ax=ax, edgecolor=local.cp_hex[-1], alpha=1, lw=1.5
    )

    panel_labels(ax)
    event_labels(ax)
    plot_scale_bar(ax, **local.scale_bar_defaults)
    arrow_length = 200
    plot_north_arrow(
        ax,
        local.map_extents_zoomed["x"][1] - 150,
        y=local.map_extents_zoomed["y"][1] - 50 - arrow_length,
        arrow_length=arrow_length,
        text_offset=30,
    )
    set_extents(ax, local.map_extents_zoomed)

    return fig


def instrumentation_location_map(bursts, stations):
    """Create the mine map with seismic stations and instrument sites"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_workings(local.dxfs["workings"], ax=ax)
    fill_mined_areas(local.dxfs["workings_simplified"], ax=ax)
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
    arrow_length = 250
    plot_north_arrow(
        ax,
        local.map_extents["x"][0] + 220,
        y=local.map_extents["y"][1] - 100 - arrow_length,
        arrow_length=arrow_length,
        text_offset=30,
    )
    set_extents(ax, local.map_extents)

    return fig


def sideD_instrument_map(bursts, inst_df):
    """Create a zoomed-in map around burst 2 showing cans/BPCs."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    plot_workings(local.dxfs["workings"], ax=ax)
    plot_events(bursts, c="#999ccc", ax=ax, s=magnitude_scaling(bursts))
    plot_instrumentation_sites(
        local.dxfs["instrumentation"],
        ax=ax,
        site_slice=slice(-2, None),
    )

    sub_inst = inst_df[inst_df["site"] == "D"]
    plot_bpcs(ax, sub_inst, bpc_colors=local.bpc_colors)

    # Label can sub-sites.
    sub_instrument_site_labels(ax)

    for time, color in local.burst_colors.items():
        plot_burst(time, local.dxfs["damage"], bursts, color, ax=ax)

    sb_params = local.scale_bar_defaults.copy()
    sb_params.update(
        {
            "dist": 50,
            "unit": "m",
            "size_vertical": 1.5,
        }
    )

    plot_scale_bar(ax, **sb_params)
    set_extents(ax, local.map_extents_siteD)

    return fig


def main():
    # Load necessary data
    bursts = pd.read_csv(local.burst_events, parse_dates=["time"])
    stations = pd.read_csv(local.station_file)
    inst_df = pd.read_csv(local.instrumentation_file)

    # Set the fonts
    configure_font_sizes(local.font_sizes)

    # Make the plots
    burst_map = burst_location_map(bursts)
    burst_map.savefig(local.burst_map, **local.savefig_params)

    inst_map = instrumentation_location_map(bursts, stations)
    inst_map.savefig(local.station_map, **local.savefig_params)

    siteD_inst_map = sideD_instrument_map(bursts, inst_df)
    siteD_inst_map.savefig(local.siteD_inst_map, **local.savefig_params)


if __name__ == "__main__":
    main()
