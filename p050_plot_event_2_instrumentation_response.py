"""
Plot the instrumentation response to event 2.
"""

from collections import defaultdict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import local

# Define the time shift for the can loads.
_shift_large = np.timedelta64(19, "h") + np.timedelta64(10, "m")
_shift_small = np.timedelta64(60, "m")
can_shifts = {
    "A": _shift_large,
    "B": _shift_large,
    "C": _shift_large,
    "D": _shift_large,
    "E": _shift_large,
    "F": _shift_large,
    "G": _shift_small,
    "H": _shift_small,
    "I": _shift_small,
    "J": _shift_small,
}
can_colors = {"d2": "tab:blue", "d1": "tab:orange"}


def filter_date(df, t1=local.inst_time_range[0], t2=local.inst_time_range[-1]):
    """Filter date to be between the times requested."""
    time = df["local_time"]
    return df[(time >= t1) & (time <= t2)]


def plot_bpc(ax, df):
    """Plot bpc data."""
    for name, sub in df.groupby("name"):
        sub = sub.set_index("local_time")["pressure_pa"].sort_index()
        sub = sub.loc[local.inst_time_range[0] : local.inst_time_range[-1]] / 1e6
        sub = sub - np.nanmin(sub)  # Normalize to pre-burst pressure.
        color = local.bpc_colors[name]
        ax.plot(sub.index.values, sub.values, label=name, color=color)
    ax.legend()
    ax.set_ylabel("Pressure (MPa)")


def prep_ax(ax, label):
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
    # Add subplot label
    ax.text(
        0.03, 1.17, label, transform=ax.transAxes, fontsize=11, va="top", ha="right"
    )


def plot_can_load(ax, df):
    """Plot the cans."""
    group_key = dict(df.groupby("can")["group"].first())
    groups = defaultdict(list)
    for name, sub in df.groupby("can"):
        sub = sub.set_index("local_time")["load_tons"].sort_index()
        time = sub.index.values
        # Need to apply shift because data logger got off.
        shift = can_shifts.get(name, np.timedelta64(0, "s"))
        time = time + shift
        in_time = (time >= local.inst_time_range[0]) & (
            time <= local.inst_time_range[-1]
        )
        if in_time.sum() == 0:
            continue
        # convert from tons to tonnes
        time = time[in_time]
        load = (sub.values / 1.102)[in_time]
        load = load - np.nanmin(load)  # normalize to pre-burst pressure.
        ser = pd.Series(data=load, index=time)
        ser.name = name
        group = group_key[name]
        groups[group].append(ser)

    for group in sorted(groups):
        ser_list = groups[group]
        color = can_colors[group]
        ser = pd.concat(ser_list, axis=1).mean(axis=1)
        ax.plot(ser.index, ser.values, label=group, color=color)

    ax.legend()
    ax.set_ylabel("Load (tonnes)")


def plot_can_displacement(ax, df):
    """Plot the cans."""
    groups = defaultdict(list)

    for (name, group), sub in df.groupby(["site", "group"]):
        time = sub["local_time"]
        in_time = (time >= local.inst_time_range[0]) & (
            time <= local.inst_time_range[-1]
        )
        if in_time.sum() < 1:
            continue

        # convert from tons to tonnes
        time = time[in_time]
        disp = (sub["displacement_m"].values * 100)[in_time]
        disp = disp - disp.max()

        # Skip the stations which went offline after the event.
        if pd.isnull(np.max(disp) - np.min(disp)):
            continue

        ser = pd.Series(data=disp, index=time)
        ser.name = name
        group_name = f"{group}_rf" if name.startswith("RF") else f"{group}_can"
        groups[group_name].append(ser)

    for group in sorted(groups):
        ser_list = groups[group]
        color = can_colors[group.split("_")[0]]
        ls = "-" if "can" in group else "--"
        label = (group.replace("_", " (") + ")").replace(" (can)", "")
        ser = pd.concat(ser_list, axis=1).mean(axis=1)
        ax.plot(ser.index, ser.values, label=label, ls=ls, color=color)

    ax.legend(loc="lower left")
    ax.set_ylabel("Displacement (cm)")
    ax.set_xlabel("Month-Day")


def main():
    """Plot the event pressure response."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(4.6, 5.0))
    prep_ax(ax1, "(a)")
    prep_ax(ax2, "(b)")
    prep_ax(ax3, "(c)")

    df_can = pd.read_parquet(local.extracted_support_can_path)
    df_bpc = pd.read_parquet(local.extracted_bpc_data_path).loc[
        lambda x: x["name"].str.startswith("BP")
    ]
    df_disp = pd.read_parquet(local.extracted_can_disp_path)
    plot_bpc(ax1, df_bpc)
    plot_can_load(ax2, df_can)
    plot_can_displacement(ax3, df_disp)

    plt.subplots_adjust(hspace=0.3)
    # plt.tight_layout()
    fig.savefig(local.inst_response_event_2_plot_path)


if __name__ == "__main__":
    main()
