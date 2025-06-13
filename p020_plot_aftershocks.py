"""
Make plots of events around coalburst times.
In this case, we are only looking at largest 2 events.
"""

from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import local


def get_fig_axes(axis_count):
    """Return a figure and axes."""
    figsize = (3.5, 5.25)
    fig, axes = plt.subplots(nrows=axis_count, ncols=1, sharex=True, figsize=figsize)
    return fig, axes


def get_sub_df(
    df,
    event_time,
    time_before=np.timedelta64(2, "D"),
    time_after=np.timedelta64(5, "D"),
):
    """Get a subset of the df around event time."""
    ts = pd.Timestamp(event_time).to_numpy()
    t1 = ts - time_before
    t2 = ts + time_after
    time = pd.to_datetime(df["time"])
    in_time = (time >= t1) & (time <= t2)
    sub_time = df[in_time]

    time_from_event = (sub_time["time"] - ts) / np.timedelta64(1, "D")
    sub_time["days_from_event"] = time_from_event
    return sub_time


def plot_mag_ts(ax, twin, sub, color):
    """Plot the magnitude time series."""
    df = sub.sort_values("time")
    x_axis = sub["days_from_event"].values
    mag = df["local_mag"].values

    ax.scatter(x_axis, mag, alpha=0.5, c=color, edgecolors="gray", linewidth=0.14)

    # replot the main event for emphasis
    ax.scatter(0, mag.max(), c=color, edgecolors="gray", linewidth=0.5, s=55)

    event_count = np.arange(len(df)) + 1

    twin.plot(x_axis, event_count, lw=2, ls="--", c="0.6")

    return ax, twin


def modified_omori_cumulative(t, k, c, p):
    """Return the cumulative form of the modified Omori's law."""
    t = np.asarray(t)
    if np.isclose(p, 1.0):
        return k * np.log((t + c) / c)
    else:
        return (k / (1 - p)) * ((t + c) ** (1 - p) - c ** (1 - p))
    return k / (c + t)


def fit_omori(df, event_time):
    """Try fitting omori's law."""
    t_rel = pd.to_datetime(df["time"]).values - event_time
    after_event = t_rel >= np.timedelta64(0)
    time_after_event = t_rel[after_event] / np.timedelta64(1, "h")
    cum = np.arange(len(time_after_event))

    out = curve_fit(modified_omori_cumulative, time_after_event, cum)
    out = {"k": out[0][0], "c": out[0][1], "p": out[0][2]}

    return out


def plot_omori(ax, df, event_time, params):
    """Plot omori's curve."""
    t_rel = pd.to_datetime(df["time"]).values - pd.to_datetime(event_time).to_numpy()
    after_event = t_rel > np.timedelta64(0)
    time_after_event = t_rel[after_event] / np.timedelta64(1, "h")
    start_count = len(df) - len(time_after_event)

    predicted = modified_omori_cumulative(time_after_event, **params)
    time_to_plot = df[after_event]["days_from_event"]

    ax.plot(
        time_to_plot,
        predicted + start_count,
        label="Omori's law",
        color="red",
        alpha=0.45,
        lw=2.5,
    )


def set_title(ax, sub_time_df, fit_params, label):
    """Create/set title."""
    # create the title
    mag = sub_time_df["local_mag"].max()
    title = (
        f"{label.title().replace("_", " ")} (M={mag:.1f}) "
        f"c:{fit_params['c']:0.1f} "
        f"k:{int(np.round(fit_params['k']))}"
    )
    ax.set_title(title, fontdict={"fontsize": 10.0})
    return ax


def set_ylabels(axes, twin_axes):
    """Set the y labels of the plot."""
    axes[-1].set_xlabel("Days from Event")
    assert len(axes) % 2 == 1, "need odd number of axis."
    middle_ax = axes[len(axes) // 2]
    middle_twin = twin_axes[len(axes) // 2]

    middle_ax.set_ylabel("Magnitude")
    middle_twin.set_ylabel("Cumulative Event Count")


def rectify_y_axis(axes):
    """Find the min/max values and set the ylims all the same."""
    ylims = [ax.get_ylim() for ax in axes]
    min_val = min([x[0] for x in ylims])
    max_val = max([x[1] for x in ylims])
    for ax in axes:
        ax.set_ylim(min_val, max_val)


def add_subplot_labels(axes):
    """Add the labels to each subplot."""
    for ax, letter in zip(axes, ascii_lowercase):
        label = f"({letter})"
        ax.text(
            0.03, 1.17, label, transform=ax.transAxes, fontsize=11, va="top", ha="right"
        )


def main():
    """Make plots of the aftershock from bursts, plot omoris."""
    event_times_utc = [
        (t, f"event_{num+1}") for num, t in enumerate(local.burst_times)
    ][1:-1]

    event_colors = [local.burst_colors[x[0]] for x in event_times_utc]

    df = pd.read_parquet(local.final_catalog)
    fig, axes = get_fig_axes(len(event_times_utc))
    twin_axes = [x.twinx() for x in axes]

    zip_iter = zip(event_times_utc, axes, twin_axes, event_colors)
    for (etime_str, label), ax, twin_ax, color in zip_iter:
        event_time = pd.to_datetime(etime_str).to_numpy()
        # Get sub df to include events in volume.
        expected_label = f"vsa_{label}"
        sub_vol_df = df[df[expected_label]]
        # Get sub df in specified time around event.
        sub_time_df = get_sub_df(sub_vol_df, event_time).sort_values("time")
        # plot event
        plot_mag_ts(ax, twin_ax, sub_time_df, color)

        fit_params = fit_omori(sub_time_df, event_time)
        plot_omori(twin_ax, sub_time_df, event_time, fit_params)
        set_title(ax, sub_time_df, fit_params, label)

    set_ylabels(axes, twin_axes)
    rectify_y_axis(axes)
    rectify_y_axis(twin_axes)
    fig.tight_layout(w_pad=0.2)
    add_subplot_labels(axes)
    fig.savefig(local.omoris_plot_path, transparent=True, dpi=300)


if __name__ == "__main__":
    main()
