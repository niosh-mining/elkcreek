""" Get a gridded spatial event count of all of the events """

from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from elkcreek.grid import Grid
from elkcreek.longwall import read_longwall_df, get_longwall_positions, get_date_from_face_position, compile_daily_face_positions
from elkcreek.plot import gridplot, plot_workings, plot_burst, plot_faults, plot_anomalous, plot_scale_bar, set_extents, configure_font_sizes

from p060_e2_panel1_seismic_progression import plot_over_advance_window
import local


def plot_map(events, bursts, reference_positions, ax, cbar_ax):
    # Grab the face positions for the two times of interest
    faces = pd.DataFrame([
        reference_positions.loc[reference_positions.date == pd.Timestamp(local.p2_face_start.date())].iloc[0],
        reference_positions.loc[reference_positions.date == pd.Timestamp(local.p2_face_end.date())].iloc[0],
    ])

    plot_over_advance_window(faces.iloc[0], faces.iloc[1], events, bursts, ax=ax, reference_positions=reference_positions, cbar_ax=cbar_ax, extents=local.map_extents_event2)
    ax.text(-0.01, 1.03, "(a)", va="bottom", ha="right", transform=ax.transAxes)


def plot_daily_event_count(events, bursts, reference_positions, ax):
    e2 = bursts.iloc[1].time
    anomalous_date = get_date_from_face_position(local.panel2_anomalous_event2, reference_positions)
    anomalous_days = (anomalous_date - e2).days

    sub = events.loc[(events.time >= local.p2_time_series_start) & (events.time < local.p2_time_series_end)]
    sub["event_count"] = 1
    ts = sub.set_index("time")["event_count"].resample("1d").sum()
    ax.plot((ts.index - e2).days, ts.values, c=local.cp_hex[1], marker="o", markeredgecolor="w")
    ax.axvline(anomalous_days, c=local.cp_hex[-1], ls="--")
    ax.axvline(0, c=local.cp_hex[-2], ls=":")
    ax.set_xlabel("Days From Event 2")
    ax.set_ylabel("Number of Events")
    ax.text(-0.01, 1.02, "(b)", va="bottom", ha="right", transform=ax.transAxes)


def panel2_seismicity(events, bursts, reference_positions):
    # Define the figure
    fig = plt.figure(figsize=(7, 3.5))
    gs = GridSpec(
        2,
        2,
        hspace=0.01,
        height_ratios=[0.9, 0.05],
        width_ratios=[0.8, 1],
        wspace=0.4,
        top=0.75,
    )
    map_ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[1, 0])
    ts_ax = fig.add_subplot(gs[:, 1])

    # Make the plot elements
    plot_map(events, bursts, reference_positions, map_ax, cbar_ax)
    plot_daily_event_count(events, bursts, reference_positions, ts_ax)

    return fig

def main():
    # Load data
    df = pd.read_parquet(local.final_catalog)
    burst_df = pd.read_csv(local.burst_events, parse_dates=["time"])
    longwall_df = read_longwall_df(local.longwall_position_path)

    # Compile daily face positions over the time window of interest
    daily_face_positions = compile_daily_face_positions("2011-01-01", "2011-05-01", longwall_df)

    # Standardize fonts
    configure_font_sizes(local.font_sizes)

    # Make the figure
    fig = panel2_seismicity(df, burst_df, daily_face_positions)
    fig.savefig(local.e2_panel2_events_path, **local.savefig_params)


if __name__ == "__main__":
    main()
