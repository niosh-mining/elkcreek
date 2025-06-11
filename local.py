"""Common variables and paths"""

from pathlib import Path

import numpy as np
import seaborn as sns

here = Path(__file__).absolute().parent
raw_data = here / "data" / "raw"
data_dir = here / "data" / "derived"
data_dir.mkdir(exist_ok=True, parents=True)

# Instrumentation locations
station_file = raw_data / "station_list.csv"
instrumentation_file = raw_data / "instrumentation.csv"
bpc_path = raw_data / "instrumentation" / "borehole_pressure_cells_and_string_pots"
support_can_path = raw_data / "instrumentation" / "support_cans"

# Seismic event stuff
event_stuff = raw_data / "events"
ims_raw = event_stuff / "ims_events"
rocksigma_raw = event_stuff / "rocksigma_events"
burst_events = event_stuff / "burst_events.csv"
burst_events_qml_path = event_stuff / "burst_events.qml"
burst_waveform_path = event_stuff / "burst_waveforms"

# DXFs
dxf_dir = raw_data / "dxfs"
dxfs = {
    "workings": dxf_dir / "workings.dxf",
    "workings_simplified": dxf_dir / "placeholder_workings.dxf",
    "advance": dxf_dir / "production.dxf",
    "overburden": dxf_dir / "overburden.dxf",
    "topo": dxf_dir / "topo.dxf",
    "damage": dxf_dir / "sig_events.dxf",
    "faults": dxf_dir / "faults.dxf",
    "instrumentation": dxf_dir / "ug_instrumentation.dxf",
}

wkt_path = raw_data / "elk_creek_coordinates.wkt"

# Volumes to analyze
analysis_volume_path = raw_data / "volumes.json"

# Monthly positions of longwall
longwall_position_path = raw_data / "face_positions.csv"

# Stuff for source inversion.
gf_model_path = raw_data / "earth_models"
pyrocko_template_path = raw_data / Path("pyrocko_templates")


# --- Derived data
# Catalogs
combined_cat_path = data_dir / "a010_combined_catalog.parquet"
cleaned_cat_path = data_dir / "a020_cleaned_catalog.parquet"
outlier_path = data_dir / "a020_outlier_events.csv"
big_events_path = data_dir / "a020_big_events.csv"
cat_path_local_info = data_dir / "a030_catalog_with_local_info.parquet"
cat_path_with_volumes = data_dir / "a040_catalog_with_geometry.parquet"
cat_path_with_longwall = data_dir / "a050_catalog_with_longwall.parquet"

# In case more steps are added, make a variable with the final catalog.
final_catalog = cat_path_with_longwall


# Instrumentation data
extracted_bpc_data_path = data_dir / "b010_extracted_bpc_data.parquet"
extracted_support_can_path = data_dir / "b020_extracted_support_can_data.parquet"
extracted_can_disp_path = data_dir / "b020_extracted_can_displacement.parquet"


# Seismic station transformation
seismic_station_with_lat_lon_path = (
    data_dir / "c010_seismic_station_with_lat_lon.parquet"
)

# Grond source inversion stuff
pyrocko_catalog_path = data_dir / "d010_pyrocko_catalogs.pf"
pyrocko_station_path = data_dir / "d020_pyrocko_stations.pf"
pyrocko_station_xml = data_dir / "d025_stations.xml"
pyrocko_pick_path = data_dir / "d030_pyrocko_picks.pf"
# pyrocko_nd_model_path = data_dir / "d040_pyrocko_models"
pyrocko_displacement_mseeds_path = data_dir / "d040_displacement_burst_waveforms"
gf_store_path = data_dir / "d050_gf_store"
grond_configs_path = data_dir / "d060_grond_configs"
grond_config_csv_path = data_dir / "d060_grond_config.csv"

grond_run_path = data_dir / "d070_grond_runs"
grond_report_path = here / "report"


# --- Plot files
plots = Path("plots")
plots.mkdir(exist_ok=True)
pairplots = plots / "source_parameter_pairplots"
pre_filtering = pairplots / "pre_filtering"
post_filtering = pairplots / "post_filtering"
burst_map = plots / "p010_burst_events.png"
station_map = plots / "p010_station_map.png"
event_2_zoomed_in_map = plots / "p010_event_2_zoomed_in.png"
can_map = plots / "p010_can_map.png"

inst_response_event_2_plot_path = plots / "p050_event_2_inst_response.png"

moment_tensor_plot_path = plots / "d080_moment_tensor_plots"
moment_tensor_plot_path.mkdir(exist_ok=True)


# --- Variables
station_groups = {
    "Original Stations": ["LOXs", "LOXbh", "RDP", "RRL", "PEB"],
    "Expanded": [
        "NOX",
        "EOR",
        "TNG",
        "LAY",
        "FEN",
        "RR2",
        "BNW",
        "WMW",
        "WMC",
        "WME",
        "TNE",
        "NMN",
        "NMS",
    ],
    "Relocated for Panel 3": ["BBW", "NWF", "FMI", "LGS", "LGN"],
    "Relocated for Panel 4": ["WAW", "SNW", "PDC", "SFE", "SCC", "SCD"],
    "Advanced with Face": [
        "TNW",
        "TNC",
        "TEE",
        "FEE",
        "FNW",
        "FNC",
        "FNS",
        "FNE",
        "SCT",
        "SCB",
    ],
}
burst_times = [
    "2010-12-04T00:37:10",
    "2011-02-17T22:47:20",
    "2011-10-19T05:24:16",
    "2012-12-02T18:44:37",
    "2013-01-02T10:11:58",
]
time_zone = "US/Mountain"  # Local timezone at the mine.

# instrumentation plot time range
inst_time_range = (np.datetime64("2011-02-16"), np.datetime64("2011-02-21"))

# Parameters for filtering events.
event_filter_params = dict(
    max_rocksigma_location_residual=80,
    # Semi-arbitrary, but they calculate their location errors differently
    max_ims_location_residual=30,
    x_range=(9800, 12800),
    y_range=(4200, 6500),
    z_range=(1400, 2100),
)

# Parameters for outliers
outlier_params = dict(
    high_apparent_stress=1,  # MPa
    high_apparent_volume=10**8.5,  # m^3
    large_source_radius=700,  # m
    big_local_mag=2,
)

# For plotting
color_palette = sns.xkcd_palette(
    [
        "purple",
        "ocean blue",
        "green blue",
        "dull yellow",
        "dark pink",
        "orange yellow",
        "shit brown",
    ]
)
burst_colors = {x: y for x, y in zip(burst_times, color_palette.as_hex())}

scale_bar_defaults = {
    "dist": 500,
    "unit": "m",
    "loc": "lower left",
    "frameon": True,
    "pad": 0.5,
    "size_vertical": 10,
    "font_properties": {"size": 10},
    "sep": 3,
    "zorder": 500,
}

bpc_colors = {
    "BP1": "tab:blue",
    "BP2": "tab:orange",
    "BP3": "tab:green",
}

can_colors = {
    "A": "tab:blue",
    "B": "tab:orange",
    "C": "tab:green",
    "D": "tab:red",
    "E": "tab:purple",
    "F": "tab:brown",
    "G": "tab:pink",
    "H": "tab:gray",
    "I": "tab:olive",
    "J": "tab:cyan",
}


map_extents = {
    "x": [9300, 13100],
    "y": [3800, 7000],
    "z": [1500, 2300],
}
map_extents_zoomed = {
    "x": [9700, 12800],
    "y": [4500, 6500],
    "z": [1500, 2300],
}
map_extents_event_2 = {
    "x": [11600, 11775],
    "y": [4700, 4825],
}


# --- Source inversion parameters

# The top of the model used by Grond. Set to be above all station elevations.
grond_datum = 3000

# Total distances passed calculated station/event distance to do GFs for.
DISTANCE_BUFFER = 0.75

# GF Sample spacing
GF_SAMPLE_SPACING = 20

# Frequency range to use for Grond misfits.
GROND_FREQ_RANGE = (1, 6)

# Duration if none provided
GROND_AFTER_PICK_TIME = 0.3
GROND_BEFORE_PICK_TIME = 0.0

# Range over which the depth can shift
DEPTH_RANGE = (-200, 200)

# Range over which the magnitude can shift
MAG_RANGE = (-1, 1)

# Range over which the north component can shift
NORTH_SHIFT_RANGE = (-100, 100)

# Range over which the east component can shift
EAST_SHIFT_RANGE = (-100, 100)

# The amount of time the origin is allowed to shift
ORIGIN_TIME_RANGE = (-0.5, 0.5)

# The ranges for the duration of the source time function
ST_DURATION_RANGE = (0.1, 0.5)

# The base parameters
P_VELOCITY = 3_048  # m/s
S_VELOCITY = 2_042  # m/s
DENSITY = 2.3

COAL_SEAM_ELEVATION = 1780  # m above sea level, found by averaging UG stations.
SHIFTED_COAL_SEAM_DEPTH = grond_datum - COAL_SEAM_ELEVATION

# The elevation ranges for which events are accepted.
EVENT_ELEVATION_RANGE = (SHIFTED_COAL_SEAM_DEPTH + 200, SHIFTED_COAL_SEAM_DEPTH - 200)

# An approximate azimuth/plunge of a normal of the coal seam
coal_normal_azimuth = 14
coal_normal_plunge = 88

# Plotting grid
spacing = 100  # m
lower_left = [map_extents_zoomed["x"][0], map_extents_zoomed["y"][0]]
upper_right = [map_extents_zoomed["x"][1], map_extents_zoomed["y"][1]]
