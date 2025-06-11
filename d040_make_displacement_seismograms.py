"""
Make displacement seismograms for grond.

The seismograms are originally in m/s as output by the IMS software.
"""

import obspy

import local
from d020_make_pyrocko_inv import get_surface_stations


def main():
    """Make the displacement seismograms."""
    base = local.pyrocko_displacement_mseeds_path
    base.mkdir(exist_ok=True)
    surface_stations = get_surface_stations()

    for path in local.burst_waveform_path.glob("*.mseed"):
        st = obspy.read(path)
        traces = [tr for tr in st if tr.stats.station in surface_stations]
        st = obspy.Stream(traces).detrend("linear").integrate()

        new = base / path.name
        st.write(new, "mseed")


if __name__ == "__main__":
    main()

