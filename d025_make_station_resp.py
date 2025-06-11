"""
Make the station xml.
"""

import matplotlib.pyplot as plt
import obspy
from obspy.core.inventory import (
    Channel,
    InstrumentSensitivity,
    Inventory,
    Network,
    Station,
)
from obspy.core.inventory.response import PolesZerosResponseStage, Response
from pyrocko import model

import local

# A PAZ stage that simply integrates from m/s to m

# Define a stage with no poles or zeros, unity gain
pz = PolesZerosResponseStage(
    stage_sequence_number=1,
    stage_gain=1.0,
    stage_gain_frequency=1.0,
    input_units="M",  # or whatever units your data already has
    output_units="M",  # same as input to avoid transformation
    normalization_factor=1.0,
    normalization_frequency=1.0,
    pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
    zeros=[0 + 1j],
    poles=[0 + 1j],
)

# InstrumentSensitivity (required!)
sensitivity = InstrumentSensitivity(
    value=1.0, frequency=10.0, input_units="M", output_units="M"
)

# Wrap in a Response object
response = Response(response_stages=[pz], instrument_sensitivity=sensitivity)


def pyrocko_stations_to_obspy_station_xml(pyrocko_stations):
    """Convert pyrocko stations to obspyxml with response info."""
    stations = []
    for station in pyrocko_stations:
        channels = []
        for channel_code in "ENZ":
            channels.append(
                Channel(
                    code=channel_code,
                    location_code="00",
                    latitude=station.lat,
                    longitude=station.lon,
                    elevation=station.elevation,
                    depth=0,
                    response=response,
                )
            )
        station = Station(
            code=station.station,
            channels=channels,
            latitude=station.lat,
            longitude=station.lon,
            elevation=station.elevation,
        )
        stations.append(station)
    net = Network(code="EC", stations=stations)
    return Inventory(networks=[net])


def test_response_removal_does_little(
    inv,
):
    """Simply apply the response and plot before/after for qualitative check."""
    wf_path = local.pyrocko_displacement_mseeds_path
    st = obspy.read(wf_path / "event_1.mseed")
    sub = st.select(station="LAY")
    sub2 = sub.copy().remove_response(inv, output="DISP")

    plt.plot(sub2[0].data, label="removed_respo")
    plt.plot(sub[0].data, label="original")
    plt.show()


def main():
    """Make pyrocko station file."""
    pyrocko_stations = model.load_stations(local.pyrocko_station_path)
    inv = pyrocko_stations_to_obspy_station_xml(pyrocko_stations)

    # Uncomment the following line to test the inventory
    # test_response_removal_does_little(inv)

    inv.write(local.pyrocko_station_xml, "stationxml")


if __name__ == "__main__":
    main()
