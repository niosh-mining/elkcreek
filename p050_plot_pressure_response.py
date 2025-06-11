"""
Plot the pressure response to event 1 using can load and BPC pressure.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import local


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
    ax.set_ylabel("Cell Pressure (MPa)")


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


def plot_cans(ax, df, no_shift={"G", "I"}):
    """Plot the cans."""
    for name, sub in df.groupby("can"):
        sub = sub.set_index("local_time")["load_tons"].sort_index()
        time = sub.index.values
        # Need to apply shift because data logger got off.
        if name not in no_shift:
            time = time + np.timedelta64(19, "h") + np.timedelta64(10, "m")
        else:
            time = time + np.timedelta64(60, "m")

        in_time = (time >= local.inst_time_range[0]) & (
            time <= local.inst_time_range[-1]
        )
        if in_time.sum() == 0:
            continue
        # convert from tons to tonnes
        time = time[in_time]
        load = (sub.values / 1.102)[in_time]
        load = load - np.nanmin(load)  # normalize to pre-burst pressure.

        color = local.can_colors[name]
        ax.plot(time, load, label=name, color=color)
    ax.legend(ncol=2, columnspacing=0.5)
    ax.set_ylabel("Can Load (tonnes)")
    ax.set_xlabel("Month-Day")


def main():
    """Plot the event pressure response."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.0, 4.5))
    prep_ax(ax1)
    prep_ax(ax2)

    df_can = pd.read_parquet(local.extracted_support_can_path)

    df_bpc = pd.read_parquet(local.extracted_bpc_data_path).loc[
        lambda x: x["name"].str.startswith("BP")
    ]

    plot_bpc(ax1, df_bpc)
    plot_cans(ax2, df_can)
    plt.subplots_adjust(hspace=0.1)
    fig.savefig(local.can_and_bpc_event_2_pressure_path)


if __name__ == "__main__":
    main()
