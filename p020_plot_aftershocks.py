"""
Make plots of events around coalburst times.
In this case, we are only looking at largest 2 events.
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import local

event_times_utc = {
    "2011-02-17T22:47:20": "event_2",
    "2011-10-19T05:24:16": "event_3",
}


def get_fig_axes():
    """Return a figure and axes."""
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7 * 1.3, 3 * 1.1))
    return fig, axes


def get_sub_df(
        df,
        event_time,
        time_before=np.timedelta64(2, "D"),
        time_after=np.timedelta64(3, "D"),
):
    """Get a subset of the df around event time."""
    ts = pd.Timestamp(event_time).to_numpy()
    t1 = ts - time_before
    t2 = ts + time_after
    time = pd.to_datetime(df['time'])
    in_time = (time >= t1) & (time <= t2)
    sub_time = df[in_time]

    return sub_time


def plot_mag_ts(ax, sub, label):
    """Plot the magnitude time series."""
    df = sub.sort_values("time")
    time = df['time'].values.astype('datetime64[ns]')
    mag = df['local_mag'].values

    ax.scatter(time, mag, alpha=0.3, c="tab:blue")

    myfmt = mdates.DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(myfmt)

    event_count = np.arange(len(df)) + 1

    twin = ax.twinx()
    twin.plot(time, event_count, lw=2, ls='--', c='0.5')

    # These parameters are a bit add-hoc just to get the twin axis
    # to line up correctly.
    twin.set_ylim(0, 900)
    twin.set_yticks([200, 400, 600, 800])

    year = pd.to_datetime(time.max()).year
    # label axes
    ax.set_xlabel(f"Date ({year:d})")

    mag = np.round(mag.max(), 1)
    title = f"{label.replace("_", " ").title()} ($M$={mag:0.1f})"
    ax.set_title(title)

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

    t_rel = pd.to_datetime(df['time']).values - pd.to_datetime(event_time).to_numpy()
    after_event = t_rel >= np.timedelta64(0)
    time_after_event = t_rel[after_event] / np.timedelta64(1, "h")
    cum = np.arange(len(time_after_event))

    out = curve_fit(modified_omori_cumulative, time_after_event, cum)
    out = {"k":out[0][0], "c": out[0][1], "p": out[0][2]}

    return out


def plot_omori(ax, df, time_from_event, params):
    """plot omori's curve."""
    event_time = np.datetime64(time_from_event)
    t_rel = pd.to_datetime(df['time']).values - pd.to_datetime(event_time).to_numpy()
    after_event = t_rel > np.timedelta64(0)
    time_after_event = t_rel[after_event] / np.timedelta64(1, "h")
    start_count = len(df) - len(time_after_event)


    predicted = modified_omori_cumulative(time_after_event, **params)
    time_to_plot = pd.to_datetime(df[after_event]['time']).to_numpy()

    ax.plot(
        time_to_plot, predicted+start_count, label="Omori's law",
        color='red', alpha=0.25,lw=2.5,

    )



def main():
    df = pd.read_parquet(local.final_catalog)
    fig, axes = get_fig_axes()

    for (event_time, label), ax in zip(event_times_utc.items(), axes):
        time = pd.to_datetime(event_time).to_numpy()
        # Get sub df to include events in volume.
        expected_label = f"vsa_{label}"
        sub_vol_df = df[df[expected_label]]
        # Get sub df in specified time around event.
        sub_time_df = get_sub_df(sub_vol_df, time).sort_values("time")
        # plot event, then do labels.
        ax, twin_ax = plot_mag_ts(ax, sub_time_df, label)

        fit_params = fit_omori(sub_time_df, event_time)
        print(f"found the following omori params for {label}: {fit_params}")
        plot_omori(twin_ax, sub_time_df, event_time, fit_params)

        if ax is axes[-1]:
            twin_ax.set_ylabel("Cumulative Event Count")
        else:
            ax.set_ylabel("Magnitude")
            twin_ax.set_yticklabels([])

    fig.tight_layout(w_pad=0.2)
    fig.savefig(local.omoris_plot_path)



if __name__ == "__main__":
    main()

