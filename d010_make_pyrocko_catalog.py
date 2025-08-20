"""
Make a catalog using pyrocko.
"""

import obspy
from pyrocko import model
from pyrocko.obspy_compat.base import to_pyrocko_events

import local


def main():
    """Create a pyrocko catalog."""
    cat = obspy.read_events(local.burst_events_qml_path)
    obspy_event_map = {str(x.resource_id): x for x in cat}
    pev = to_pyrocko_events(cat)
    good_events = []
    for event in pev:
        # Correct name to just be the event id.
        event.name = event.name.split("-smi")[0].split("/")[-1]
        # readjust depth to datuum
        event.depth = local.grond_datum + event.depth
        # Pyrocko doesn't seem to convert mags correctly so manually add here.
        rid = f"smi:local/{event.name}"
        obspy_mag = obspy_event_map[rid].preferred_magnitude()
        # if obspy_mag is None:
        #     breakpoint()
        event.magnitude = obspy_mag.mag
        event.magnitude_type = obspy_mag.magnitude_type
        good_events.append(event)

    model.dump_events(good_events, local.pyrocko_catalog_path)


if __name__ == "__main__":
    main()
