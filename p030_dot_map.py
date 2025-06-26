""" Plot a map of all of the events"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.dates as mdates

from elkcreek.plot import plot_workings, plot_events, plot_scale_bar, plot_workings, magnitude_scaling, set_extents, configure_font_sizes, date_normalizer, viridis_cmap

import local


def add_events(df, ax, normalizer, bursts):
    # events
    plot_events(df, c="None", ax=ax, s=1, edgecolors=viridis_cmap(df.time, normalizer), alpha=0.1)
    # "big" events
    big = df.loc[df.local_mag > local.big_mag]
    plot_events(big, c=viridis_cmap(big.time, normalizer), edgecolors="k", s=magnitude_scaling(big)/2, ax=ax)
    # bursts
    bc = local.cp_hex[-2]
    bursts = df.loc[df.time.isin(bursts.time)]
    plot_events(bursts, s=magnitude_scaling(bursts)/2, c=bc, edgecolors="k", ax=ax)


def plan_view(events, ax, normalizer, bursts):
    plot_workings(local.dxfs["workings"], ax=ax)
    add_events(events, ax, normalizer, bursts)
    set_extents(ax, local.map_extents)
    # Scale bar
    sb = local.scale_bar_defaults.copy()
    sb.update({
        "dist": 1000,
        "unit": "1 km",
        "label_override": True,
        "size_vertical": 20,
    })
    plot_scale_bar(ax, **sb)
    # Create a legend for the dot size
    ax.scatter(0, 0, c="k", s=1, label="M<2")
    ax.scatter(0, 0, c="k", s=magnitude_scaling({"local_mag": 2})/2, label="M2")
    ax.scatter(0, 0, c="k", s=magnitude_scaling({"local_mag": 3})/2, label="M3")
    ax.legend(loc="upper right")
    # Subplot label
    ax.text(-0.08, 0.97, "(a)", transform=ax.transAxes)
    ax.set_aspect("equal")


def face_centered_geometry(ax):
    # Add the approximate face position (and gateroads)
    ax.plot([-125, 125],[0,0], c="k")
    # Normal panel width
    ax.plot([-125, -125],[-1000, 1000], c="k")
    ax.plot([125, 125],[-1000, 1000], c="k")
    # Min panel width (panel 3b)
    ax.plot([-70, -70],[-1000, 1000], c="#666", ls="--")
    ax.plot([70, 70],[-1000, 1000], c="#666", ls="--")
    # Add an arrow to indicate mining direction
    ax.arrow(0, 0, 0, 150, head_width=60, head_length=80, fc="#aa0000", ec="#aa0000", zorder=100)
    # Label the headgate and tailgate sides
    ax.text(-290, 630, "HG")
    ax.text(170, 630, "TG")


def face_centered_plot(events, ax, normalizer, bursts):
    face_centered_geometry(ax)
    # Face-centered df...
    fc = events[["time", "location_x_longwall_coord", "location_y_longwall_coord", "local_mag"]].rename(columns={"location_x_longwall_coord": "x", "location_y_longwall_coord": "y"})
    add_events(fc, ax, normalizer, bursts)
    set_extents(ax, local.face_centered_extents)
    # Scale bar
    sb = local.scale_bar_defaults.copy()
    sb.update({
        "dist": 500,
        "unit": "m",
        "label_override": False,
        "loc": "lower right",
    })
    plot_scale_bar(ax, **sb)
    # Subplot label
    ax.text(-0.18, 0.97, "(b)", transform=ax.transAxes)
    ax.set_aspect("equal")


def main():
    # Load the data
    df = pd.read_parquet(local.final_catalog)
    burst_df = pd.read_csv(local.burst_events, parse_dates=["time"])

    configure_font_sizes(local.font_sizes)

    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(
        2,
        2,
        hspace=0.1,
        width_ratios=[2, 1],
        height_ratios=[0.95, 0.05],
    )
    plan_ax = fig.add_subplot(gs[0, 0])
    fc_ax = fig.add_subplot(gs[0, 1])

    # Now do the colorbar
    gs_cbar = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], width_ratios=[0.1, 0.9, 0.1])
    cb_ax = fig.add_subplot(gs_cbar[1])

    normalizer = date_normalizer(df.time)

    plan_view(df, plan_ax, normalizer, burst_df)
    face_centered_plot(df, fc_ax, normalizer, burst_df)

    # Add the color scale to the plot
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap="viridis"), cax=cb_ax, orientation="horizontal", label="Date")
    cbar.ax.xaxis.set_major_locator(mdates.YearLocator())
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(local.dot_map_path, **local.savefig_params)


if __name__ == "__main__":
    main()
