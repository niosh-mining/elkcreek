The raw data consists of the following:

dxfs:
 - Elk_Creek_map.dxf: Map of the Elk Creek mine including face advance, seismic station locations, in-mine instrumentation, etc. Most of the layers are also separated out as individual files to make parsing with python faster.
 - anomalous_zones.dxf: Outlines of the anomalous zones observed in the 2N gateroad.
 - faults.dxf: Mapped faults.
 - overburden.dxf: Overburden contours. Note that the overburden contours were originally created using imperial units and later converted to metric. As a result, all of the data points are in mine coordinates in meters, but the text labels themselves still show feet.
 - placeholder_workings.dxf: Simplified panel outlines.
 - production.dxf: Monthly face positions.
 - sig_events.dxf: Locations, damage outlines, and face positions for the five coal bursts.
 - topo.dxf: Topography contours. As with the overburden, the contours were originally generated using imperial units, so while the data points are in the mine coordinates in meters, the text labels are in feet.
 - ug_instrumentation.dxf: Locations of underground instrumentation sites.
 - workings.dxf: The mine workings.

earth_models:
 - homogeneous.csv: Homogeneous velocity model for moment tensor inversion.

events:
 - burst_waveforms: Miniseed files for the waveforms of the 5 coal bursts.
 - ims_events: The original (JMTS) seismic catalog, broken up by time due to file size restrictions.
 - rocksigma_events: The BEMIS-reprocessed seismic catalog, broken up by time due to file size restrictions.
 - burst_events.csv: Summary information for the five coal bursts.
 - burst_events.qml: QuakeML containing the pick information ofr the five coal bursts.

instrumentation:
 - borehole_pressure_cells_and_string_pots: Data for the different borehole pressure cells and string potentiometers installed underground.
 - closures_and_mpbx_extensometers: Data for the different closure and multi-point borehole extensiometer measurements underground.
 - face_positions:
   - Hourly distance from BPC instrument lines to face.xlsx: Distance from the mining face to the different in-mine instrumentation sites, determined on an hourly basis.
   - Summary of Face Positions from Production Records.xlsx: Face positions for Panels 1 and 2 in gate road coordinates.
 - ground_condition_surveys: Ground condition surveys conducted in the 1N and 2N gateroads.
 - observation_notes:
   - Notes on Observations of Conditions and Caving.pdf: Observations on the caving that occurred towards the end of mining of Panel 1.
 - sonic_probes: Data from the sonic probes installed underground.
 - support_cans: Data collected at the support Cans.
 - pyrocko_templates: Template configuration files for Pyrocko.
 - elk_creek_coordinates.wkt: Coordinate conversion from mine coordinates to Latitude/Longitude.
 - face_positions.csv: Monthly face positions.
 - instrumentation.csv: An inventory of all of the underground instrumentation (excluding seismic stations).
 - station_list.csv: An inventory of all of the seismic stations.
 - volumes.json: Spatial volumes of interest for examining the seismicity.


The waveform data for all of the events can be found [here](https://data.cdc.gov/browse?q=elk+creek+seismic+waveforms&sortBy=alpha&pageSize=20).