"""
Convert picks to pyrocko format.
"""

import obsplus
from obsplus.constants import NSLC
from pyrocko import marker, model

import local
from d020_make_pyrocko_inv import get_surface_stations

POLARITY_MAP = {"positive": 1, "negative": -1, "": None}


def main():
    """Convert picks to pyrocko format."""
    df = obsplus.picks_to_df(local.burst_events_qml_path)
    df = df[df["evaluation_status"] != "rejected"].drop_duplicates(
        subset=["station", "event_id", "phase_hint"]
    )
    surface_stations = get_surface_stations()
    events = model.load_events(str(local.pyrocko_catalog_path))
    markers = []
    for event in events:
        emarker = marker.EventMarker(event=event)
        markers.append(emarker)

        eid = event.name  # event.name[10:20]
        sub = df[df["event_id"].str.contains(eid)]
        for _, row in sub.iterrows():
            nslc = [tuple(row[x] for x in NSLC)]
            if row["station"] not in surface_stations:
                continue
            pick = marker.PhaseMarker(
                nslc_ids=nslc,
                tmin=row["time"].timestamp(),
                tmax=row["time"].timestamp(),
                event=event,
                phasename=row["phase_hint"],
                polarity=POLARITY_MAP[row["polarity"]],
            )
            markers.append(pick)
    marker.save_markers(markers, local.pyrocko_pick_path)


if __name__ == "__main__":
    main()
