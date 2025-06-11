"""
Clean the event catalog and add local time.
"""

import numpy as np
import pandas as pd
from elkcreek.events import filter_event_df, get_outliers

import local


def add_burst_number(df, burst_times=local.burst_times):
    """
    Add a column indicating the burst number.

    The events that are not bursts (most of them) will get a -1.
    """
    event_times = df["time"].values
    burst_inds = np.ones(len(event_times), dtype=np.int64) * -1
    burst_times = np.array(burst_times).astype(np.datetime64)
    for ind, time in enumerate(burst_times):
        diff = np.abs(event_times - time) / np.timedelta64(1, "s")
        argmin = np.argmin(diff)
        tdiff = np.min(diff)
        assert tdiff < 1.0, "Possible mis-association, tdiff too great."
        burst_inds[argmin] = ind
    return df.assign(burst_number=burst_inds)


def main():
    """Clean the catalog."""
    bursts = pd.read_csv(local.burst_events, parse_dates=["time"])
    # Read and filter events.
    out = (
        pd.read_parquet(local.combined_cat_path)
        .pipe(filter_event_df, burst_events=bursts, **local.event_filter_params)
        .pipe(add_burst_number, burst_times=local.burst_times)
    )

    outliers, big_events = get_outliers(out, **local.outlier_params)
    outliers.to_csv(local.outlier_path)
    big_events.to_csv(local.big_events_path)

    out.to_parquet(local.cleaned_cat_path)


if __name__ == "__main__":
    main()
