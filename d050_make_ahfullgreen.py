"""
Make the full green's function response using Pyrocko's ahfullgreen backend.
"""

import os
import time
from contextlib import contextmanager
from functools import cache

import numpy as np
import pandas as pd
from jinja2 import BaseLoader, Environment
from obsplus.utils.geodetics import SpatialCalculator
from pyrocko import gf, model
from pyrocko.apps.fomosto import fomo_wrapper_module

import local
from local import EVENT_ELEVATION_RANGE

EXTRA = """
--- !pf.AhfullgreenConfig
stf: !pf.AhfullgreenSTFImpulse {}
"""


@cache
def get_config_template(template_path=local.pyrocko_template_path):
    """Get the template for the configuration."""
    config_str = (template_path / "ahfullgreen_config").read_text()
    renderer = Environment(loader=BaseLoader()).from_string(config_str)
    return renderer


@contextmanager
def timeit_context(name):
    """Time some function."""
    print(f"starting {name}")  # noqa
    t1 = time.time()
    yield
    t2 = time.time()
    print(f"finished {name}, duration={(t2 - t1):0.2f} seconds")  # noqa


def render_config_template(
    sta_depth,
    source_depth_min,
    source_depth_max,
    distance_min,
    distance_max,
    p_velocity,
    s_velocity,
    density,
    sampling_rate=100.0,
    model_name="",
):
    """Render the config template."""
    temp = get_config_template()
    temp_str = temp.render(
        station_depth=f"{sta_depth:0.2f}",
        source_depth_min=f"{source_depth_min:0.2f}",
        source_depth_max=f"{source_depth_max:0.2f}",
        distance_min=f"{distance_min:0.2f}",
        distance_max=f"{distance_max:0.2f}",
        p_velocity=f"{p_velocity:0.2f}",
        s_velocity=f"{s_velocity:0.2f}",
        density=f"{density:0.2f}",
        gf_id=model_name,
        sampling_rate=sampling_rate,
    )
    return temp_str


def write_file(base_path, config_str):
    """Write the needed files to directory."""
    base_path.mkdir(parents=True, exist_ok=True)
    with (base_path / "config").open("w") as fi:
        fi.write(config_str)
    extra_path = base_path / "extra"
    extra_path.mkdir(exist_ok=True)
    with (extra_path / "ahfullgreen").open("w") as fi:
        fi.write(EXTRA)


def run_fomosto(path):
    """Run fomosto to generate GFs."""
    store = gf.Store(path)
    with timeit_context("travel time tables"):
        store.make_travel_time_tables(
            force=True,
        )
    module, _ = fomo_wrapper_module(store.config.modelling_code_id)
    with timeit_context("making GF store"):
        module.build(
            path,
            force=True,
            nworkers=os.cpu_count(),
            continue_=None,
            step=None,
            iblock=None,
        )
    with timeit_context("Calculating take off angles"):
        store.make_takeoff_angle_tables(force=True)


def _iter_to_df(iterable_thing, mapping):
    """Create a dataframe by extracting things from iterable."""
    out = pd.DataFrame(index=range(len(iterable_thing)))
    for new_name, old_name in mapping.items():
        out[new_name] = [getattr(x, old_name) for x in iterable_thing]
    return out


def pyrocko_stations_to_df(stations):
    """Convert a list of pyrocko stations to a dataframe."""
    name_map = {
        "latitude": "lat",
        "longitude": "lon",
        "elevation": "elevation",
        "network": "network",
        "station": "station",
    }
    return _iter_to_df(stations, name_map)


def pyrocko_events_to_df(events):
    """Convert pyrocko catalog to dataframe."""
    name_map = {
        "latitude": "lat",
        "longitude": "lon",
        "depth": "depth",
        "time": "time",
        "magnitude": "magnitude",
        "event_id": "name",
    }
    return _iter_to_df(events, name_map)


def main():
    """Make Green's function store."""
    gf_space = local.GF_SAMPLE_SPACING
    stations = model.load_stations(local.pyrocko_station_path)
    inv_df = pyrocko_stations_to_df(stations)
    events = model.load_events(str(local.pyrocko_catalog_path))
    event_df = pyrocko_events_to_df(events)

    # coal_depth_m = local.grond_datum - local.COAL_SEAM_ELEVATION
    event_depth_min = EVENT_ELEVATION_RANGE[-1]
    event_depth_max = EVENT_ELEVATION_RANGE[0]

    sta_depths = -inv_df["elevation"]
    sta_depth_mean = np.round(sta_depths.mean())

    # Need to use command: fomosto sat {model_name} to get takeoff angles.
    # raise NotImplemented("Not implemented yet.")

    out = SpatialCalculator()(
        inv_df.set_index("station"),
        event_df.set_index("event_id"),
    )

    # Get min max distances from events
    dmin = np.floor(out["distance_m"].min() // gf_space) * gf_space
    dmax = np.ceil(out["distance_m"].max() // gf_space) * gf_space
    dist_min = np.max([0, dmin - local.DISTANCE_BUFFER * 1_000])
    dist_max = dmax + local.DISTANCE_BUFFER * 1_000

    pvel, svel, den = local.P_VELOCITY, local.S_VELOCITY, local.DENSITY

    config_str = render_config_template(
        sta_depth_mean,
        event_depth_min,
        event_depth_max,
        dist_min,
        dist_max,
        model_name="ahfullgreen",
        p_velocity=pvel,
        s_velocity=svel,
        density=den,
        sampling_rate=500,
    )
    write_file(local.gf_store_path / "ahfullgreen", config_str)
    run_fomosto(local.gf_store_path / "ahfullgreen")


if __name__ == "__main__":
    main()
