"""
Plot the instrumentation response to event 2.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from elkcreek.util import convert_timezone
from elkcreek.plot import configure_font_sizes

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
can_colors = {"d2": local.cp_hex[1], "d1": local.cp_hex[-1]}


def filter_date(df, t1=local.inst_time_range[0], t2=local.inst_time_range[-1]):
    """Filter date to be between the times requested."""
    time = df["local_time"]
    return df[(time >= t1) & (time <= t2)]


def plot_bpc(ax, df, burst_time):
    """Plot bpc data."""
    for name, sub in df.groupby("name"):
        sub = sub.set_index("local_time")["pressure_pa"].sort_index()
        sub = sub.loc[local.inst_time_range[0] : local.inst_time_range[-1]] / 1e6
        sub = sub - np.nanmin(sub)  # Normalize to pre-burst pressure.
        color = local.bpc_colors[name]
        x_axis = (sub.index.values - burst_time) / np.timedelta64(1, "D")
        ax.plot(x_axis, sub.values, label=name, color=color)
    ax.legend()
    ax.set_ylabel("Pressure (MPa)")


def prep_ax(ax, label):
    """Prepare the plotting axis."""
    # Add subplot label
    ax.text(
        0.03, 1.17, label, transform=ax.transAxes, va="top", ha="right"
    )


def plot_can_load(ax, df, burst_time):
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
        x_axis = (time - burst_time) / np.timedelta64(1, "D")
        ser = pd.Series(data=load, index=x_axis)
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


def plot_can_displacement(ax, df, burst_time):
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

        x_axis = (time - burst_time) / np.timedelta64(1, "D")

        ser = pd.Series(data=disp, index=x_axis)
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
    ax.set_xlabel("Days from Event 2")


def main():
    """Plot the event pressure response."""

    configure_font_sizes(local.font_sizes)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(3.5, 5.1))
    burst_time_utc = np.datetime64(local.burst_times[1])
    burst_time_local = convert_timezone(burst_time_utc, "UTC", "US/Mountain")

    prep_ax(ax1, "(a)")
    prep_ax(ax2, "(b)")
    prep_ax(ax3, "(c)")

    df_can = pd.read_parquet(local.extracted_support_can_path)
    df_bpc = pd.read_parquet(local.extracted_bpc_data_path).loc[
        lambda x: x["name"].str.startswith("BP")
    ]
    df_disp = pd.read_parquet(local.extracted_can_disp_path)
    plot_bpc(ax1, df_bpc, burst_time_local)
    ax1.set_title("BPC Pressure")
    plot_can_load(ax2, df_can, burst_time_local)
    ax2.set_title("Can Load")
    plot_can_displacement(ax3, df_disp, burst_time_local)
    ax3.set_title("Convergence")

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    fig.savefig(local.inst_response_event_2_plot_path, **local.savefig_params)


if __name__ == "__main__":
    main()
