"""
Make the config file for Grond based on pick content.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from functools import cache
from hashlib import md5
from itertools import product
from pathlib import Path

import grond
import numpy as np
import obsplus
import obspy
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pyrocko.model.event import load_events
from pyrocko.model.station import load_stations

collections.Sequence = Sequence

import local  # noqa

# ALL_CHANNELS = ['Z', 'E', 'N']
VERTICAL_CHANNELS = [
    "Z",
]
HORIZONTAL_CHANNELS = ["E", "N"]
# REVERSE_CHANNEL_MAP = {v:i for i, v in local.channel_map.items()}

AUTOSHIFT_PENALITY = 0.0001


class GrondConfig(BaseModel):
    """
    Parameters to vary for grond runs.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    store_id: str = "homogenous"
    # Config for target groups
    use_cc_target: bool = True
    use_amp_target: bool = True
    # use_waveform_target: bool
    use_polarity_target: bool = True
    use_envelope_target: bool = True
    # Config for source search
    fixed_hypocenter: bool = True
    fixed_depth: bool = False
    # Frequency range
    frequency_range: tuple[float, float] = local.GROND_FREQ_RANGE

    base_path: Path = local.grond_configs_path
    event_path: Path = local.burst_events_qml_path
    polarity_weight: float = 0.1
    cc_weight: float = 1.0
    amp_weight: float = 0.01
    used_stations: set[str]

    @property
    def config_id(self):
        """Unique id for this combination of params."""
        json = self.model_dump()
        json.pop("event_id")
        some_str = str(json)
        hash_str = md5(some_str.encode("utf-8")).hexdigest()
        return hash_str

    @property
    def event_name(self):
        """Short version of event id."""
        return self.event_id.split("/")[-1]

    @property
    def name_template(self):
        """Get the template string."""
        name_template = f"{self.config_id}" + "_${event_name}"
        return name_template

    @property
    def path(self):
        """Get the path the config file will be saved to."""
        file_name = self.name_template.format(event_name=self.event_name) + ".gronf"
        out_path = local.grond_configs_path / file_name
        return out_path

    def make_cc_target(self, pick, duration, max_shift=None, weight=1):
        """Make a target for cross correlation."""
        # Note I just picked a sensible max_shift duration from looking at filtered wfs.
        # duration = amplitude['time_end']
        phase = pick["phase_hint"]
        misfit_kwargs = _get_base_misfit_params(
            pick, duration_after=duration, freqs=self.frequency_range
        )
        net_sta = f"{pick['network']}.{pick['station']}"

        max_shift = duration / 2 if max_shift is None else max_shift
        channels = VERTICAL_CHANNELS if "Z" in pick.channel else HORIZONTAL_CHANNELS

        waveform_misfit = grond.WaveformMisfitConfig(
            domain="cc_max_norm",
            tautoshift_max=np.round(max_shift, 4),
            autoshift_penalty_max=AUTOSHIFT_PENALITY,
            norm_exponent=1,
            **misfit_kwargs,
        )
        wtg = grond.WaveformTargetGroup(
            include=[net_sta],
            weight=weight,
            channels=channels,
            path=f"cc.{phase.upper()}.{pick['station']}",
            misfit_config=waveform_misfit,
            interpolation="nearest_neighbor",
            store_id=self.store_id,
        )
        return wtg

    def make_amp_target(self, pick, duration, weight=1):
        """Make the amplitude target."""
        phase = pick["phase_hint"]
        misfit_kwargs = _get_base_misfit_params(
            pick, duration_after=duration, freqs=self.frequency_range
        )
        net_sta = f"{pick['network']}.{pick['station']}"
        channels = VERTICAL_CHANNELS if "Z" in pick.channel else HORIZONTAL_CHANNELS
        waveform_misfit = grond.WaveformMisfitConfig(
            domain="frequency_domain",
            tautoshift_max=np.round(duration * 0.25, 4),
            autoshift_penalty_max=AUTOSHIFT_PENALITY,
            norm_exponent=1,
            **misfit_kwargs,
        )
        wtg = grond.WaveformTargetGroup(
            path=f"fd.{phase.upper()}.{pick['station']}",
            include=[net_sta],
            weight=weight,
            misfit_config=waveform_misfit,
            interpolation="nearest_neighbor",
            channels=channels,
            store_id=self.store_id,
        )
        return wtg

    def make_pick_target_group(self):
        """Make the pick target group to account for polarities."""
        out = grond.PhasePickTargetGroup(
            store_id=self.store_id,
            normalisation_family="picks",
            path="pick.phase",
            pick_synthetic_traveltime="{stored:any_p}",
            pick_phasename="P",
            weight_traveltime=0,
            weight_polarity=self.polarity_weight,
        )
        return out

    def make_waveform_target_groups(self, picks):
        """Make target groups."""
        targets = []
        if (~pd.isnull(picks["duration"])).any():
            picks = picks[~pd.isnull(picks["duration"])]
        else:
            picks["duration"] = local.GROND_AFTER_PICK_TIME
        for _, pick in picks.iterrows():
            weight = 1 / len(picks)
            cc_weight = weight * self.cc_weight
            amp_weight = weight * self.amp_weight
            duration = pick["duration"]
            if self.use_cc_target:
                targets.append(self.make_cc_target(pick, duration, weight=cc_weight))
            if self.use_amp_target:
                targets.append(self.make_amp_target(pick, duration, weight=amp_weight))
        if self.use_polarity_target:
            targets.append(self.make_pick_target_group())
        return targets

    def make_grond_config(self):
        """Make the grond config file."""
        picks = get_picks_with_duration(self.event_path, self.event_id)
        if self.used_stations:
            picks = picks[picks["station"].isin(self.used_stations)]

        event = get_event()[self.event_name]
        wtgs = self.make_waveform_target_groups(picks)
        ds_config = get_dataset_config()
        eng_config = get_engine_config()
        prob_config = get_problem_config(
            event,
            self.name_template,
            fixed_depth=self.fixed_depth,
            fixed_hypocenter=self.fixed_hypocenter,
        )
        analyzer_configs = get_analyser_configs()
        opt_config = get_optimizer_config()
        config = get_config(
            wtgs,
            ds_config,
            eng_config,
            prob_config,
            analyzer_configs,
            opt_config,
            self.event_name,
        )
        file_contents = "%YAML 1.1\n" + config.dump()
        return file_contents

    def save(self):
        """Save the grond config to disk."""
        contents = self.make_grond_config()
        self.path.parent.mkdir(exist_ok=True, parents=True)
        with self.path.open("w") as fi:
            fi.write(contents)

    def to_series(self):
        """Convert the config object to a series."""
        ser = pd.Series(self.model_dump())
        ser.name = self.config_id
        return ser


def _get_base_misfit_params(
        pick,
        duration_after=local.GROND_AFTER_PICK_TIME,
        duration_before=0,
        freqs=local.GROND_FREQ_RANGE,
):
    phase_name = pick["phase_hint"].lower()
    stored_name = "{%s}" % f"stored:any_{phase_name}"
    misfit = dict(
        fmin=freqs[0],
        fmax=freqs[1],
        pick_synthetic_traveltime=stored_name,
        pick_phasename=phase_name.upper(),
        tmin=stored_name + f"-{duration_before:0.2f}",
        tmax=stored_name + f"+{duration_after:0.2f}",
    )
    return misfit


@cache
def get_picks_with_duration(path, eid):
    """Get only the accepted picks and amplitudes."""
    path = Path(path)
    if path.is_file():
        events = obspy.read_events(path)
    elif path.is_dir():
        events = obsplus.EventBank(path)
    else:
        raise ValueError("Bad input.")

    obspy_id = eid if eid.startswith("smi:local/") else f"smi:local/{eid}"
    sub_catalog = events.get_events(eventid=obspy_id)
    assert len(sub_catalog)
    event = sub_catalog[0]

    picks_all = obsplus.picks_to_df(event)
    picks = picks_all[picks_all["evaluation_status"] != "rejected"]

    amplitudes = (
        obsplus.amplitudes_to_df(event)
        .loc[lambda x: x["pick_id"].isin(picks["resource_id"])]
        .loc[lambda x: x["evaluation_status"] != "rejected"]
    )
    picks = picks.set_index("resource_id")
    amps = amplitudes.set_index("pick_id")
    # Add durations
    picks["duration"] = np.nan

    picks.loc[amps.index, "duration"] = amplitudes["time_end"].values

    return picks


@cache
def get_event(path=local.pyrocko_catalog_path):
    """Get the event with given name."""
    events = load_events(str(path))
    out = {x.name: x for x in events}
    return out


def get_dataset_config():
    """Create the dataset configuration."""
    dsconfig = grond.DatasetConfig(
        events_path=str(local.pyrocko_catalog_path),
        stations_path=str(local.pyrocko_station_path),
        extend_incomplete=True,
        waveform_paths=[str(local.pyrocko_displacement_mseeds_path)],
        picks_paths=[str(local.pyrocko_pick_path)],
        responses_stationxml_paths=[str(local.pyrocko_station_xml)],
    )
    return dsconfig


def get_engine_config():
    """Create the engine configuration."""
    conf = grond.EngineConfig(
        gf_stores_from_pyrocko_config=False,
        gf_store_superdirs=[str(local.gf_store_path)],
    )
    return conf


def get_problem_config(event, name_template, fixed_hypocenter=True, fixed_depth=False):
    """Get pyrocko configuration for problem."""
    mag = event.magnitude
    coal_depth = local.SHIFTED_COAL_SEAM_DEPTH
    move_north = f"{local.NORTH_SHIFT_RANGE[0]} .. {local.NORTH_SHIFT_RANGE[1]}"
    move_east = f"{local.EAST_SHIFT_RANGE[0]} .. {local.EAST_SHIFT_RANGE[1]}"

    dep1, dep2 = local.DEPTH_RANGE
    fixed_dep = f"{int(coal_depth)} .. {int(coal_depth)}"
    move_dep = f"{int(coal_depth + dep1)} .. {int(coal_depth + dep2)}"
    time = f"{local.ORIGIN_TIME_RANGE[0]} .. {local.ORIGIN_TIME_RANGE[1]} | add"
    duration = f"{local.ST_DURATION_RANGE[0]} .. {local.ST_DURATION_RANGE[1]}"

    ranges = dict(
        # Only let magnitudes go up/down by 1 unit.
        magnitude=f"{mag - 1:0.2f} .. {mag + 1:0.2f}",
        time=time,
        # Only let coal seam move up/down 200,
        depth=fixed_dep if fixed_depth else move_dep,
        # Dont allow any shift in lateral location
        north_shift="0 .. 0" if fixed_hypocenter else move_north,
        east_shift="0 .. 0" if fixed_hypocenter else move_east,
        # Duration of source time function
        duration=duration,
        # Apparently we aren't supposed to touch these
        rmnn="-1.41421 .. 1.41421",
        rmee="-1.41421 .. 1.41421",
        rmdd="-1.41421 .. 1.41421",
        rmne="-1 .. 1",
        rmnd="-1 .. 1",
        rmed="-1 .. 1",
    )

    out = grond.CMTProblemConfig(
        ranges=ranges,
        distance_min=0.0,
        mt_type="full",
        norm_exponent=1,
        name_template=name_template,
    )
    return out


def get_analyser_configs():
    """Get the analyzer manual_configs."""
    from grond.analysers.target_balancing.analyser import TargetBalancingAnalyserConfig

    out = [
        TargetBalancingAnalyserConfig(niterations=1_000),
    ]
    return out


def get_optimizer_config():
    """Get the optimizer config."""
    # n_itter_1 = 2_000
    # n_itter_2 = 30_000
    # nbootstrap = 100

    n_itter_1 = 2_000 * 2
    n_itter_2 = 30_000 * 1.5
    nbootstrap = 100 * 1.5

    uniform_sampler = grond.UniformSamplerPhase(niterations=n_itter_1)
    directed_sampler = grond.DirectedSamplerPhase(
        niterations=n_itter_2,
        scatter_scale_begin=2.0,
        scatter_scale_end=1.2,
    )
    opt = grond.HighScoreOptimiserConfig(
        nbootstrap=nbootstrap,
        sampler_phases=[uniform_sampler, directed_sampler],
    )
    return opt


def get_config(
        wtgs,
        ds_config,
        eng_config,
        prob_config,
        analyser_configs,
        opt_config,
        event_name,
):
    """Get a configuration file."""
    out = grond.Config(
        rundir_template=str(local.grond_run_path) + "/${problem_name}.grun",
        path_prefix=str(local.here),
        dataset_config=ds_config,
        target_groups=wtgs,
        engine_config=eng_config,
        problem_config=prob_config,
        analyser_configs=analyser_configs,
        optimiser_config=opt_config,
        event_names=[event_name],
    )
    return out


def main():
    """Make configuration for grond."""
    events = load_events(str(local.pyrocko_catalog_path))[1:2]
    # Need to use all when ready.
    event_ids = np.array([x.name for x in events])

    # Load used stations
    stations = load_stations(str(local.pyrocko_station_path))
    used_stations = {x.station for x in stations}

    # Create possible configs from given inputs
    possible_configs = dict(
        event_id=event_ids,
        store_id=["ahfullgreen"],
        frequency_range=[(5, 15)],
        use_cc_target=[True],
        use_amp_target=[False],
        use_polarity_target=[True],
        fixed_hypocenter=[True],
        fixed_depth=[True],
    )

    configs = []
    for data in product(*possible_configs.values()):
        _config_params = {i: v for i, v in zip(possible_configs.keys(), data)}
        config = GrondConfig(
            used_stations=used_stations,
            **_config_params,
        )
        config.save()
        configs.append(config)

    df = (
        pd.DataFrame([config.to_series() for config in configs])
        .reset_index()
        .rename(columns={"index": "run_id"})
        .groupby("run_id")
        .first()
        .drop(columns=["event_id"])
    )
    df.to_csv(local.grond_config_csv_path)


if __name__ == "__main__":
    main()
