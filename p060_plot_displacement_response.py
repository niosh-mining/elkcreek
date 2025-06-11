"""
Plot the displacement response to event 1 using can load and BPC pressure.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import local


def prep_ax(ax):
    """Prepare the plotting axis."""
    event_time = pd.to_datetime(local.burst_times[1])
    event_time_local = (
        event_time.tz_localize("UTC")
        .tz_convert(local.time_zone)
        .tz_localize(None)
        .to_numpy()
    )
    ax.set_xlim(local.inst_time_range)
    ax.axvline(
        event_time_local,
        color="gray",
        alpha=0.5,
        ls="--",
        lw=2.0,
    )
    date_format = mdates.DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_format)


def plot_cans(ax, df):
    """Plot the cans."""
    for name, sub in df.groupby("site"):
        if len(name) > 1:  # Only plotting string pods near can.
            continue
        time = sub["local_time"]
        in_time = (time >= local.inst_time_range[0]) & (
            time <= local.inst_time_range[-1]
        )
        # convert from tons to tonnes
        time = time[in_time]
        disp = (sub["displacement_m"].values * 1000)[in_time]
        disp = disp - disp.max()
        if in_time.sum() < 1:
            continue

        diff = np.max(disp) - np.min(disp)
        # Skip the stations which went offline after the event.
        if pd.isnull(diff):
            continue
        color = local.can_colors[name]
        ax.plot(time, disp, label=name, color=color)
    ax.legend(ncol=2, columnspacing=0.5)
    ax.set_ylabel("Displacement (mm)")
    ax.set_xlabel("Month-Day")


def main():
    """Make displacement plots."""
    fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(5.0, 2.75))
    prep_ax(ax1)

    df_disp = pd.read_parquet(local.extracted_can_disp_path)
    plot_cans(ax1, df_disp)
    plt.tight_layout()
    fig.savefig(local.displacement_plot_path)


if __name__ == "__main__":
    main()
