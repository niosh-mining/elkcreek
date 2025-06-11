"""
Utilities for working with Grond.
"""

from collections import defaultdict

from pyrocko import model


def _read_best_mean_from_report(path):
    """Read the pyrocko events in the report folder for best/mean solution."""
    best_path = path / "event.solution.best.yaml"
    mean_path = path / "event.solution.mean.yaml"

    assert best_path.exists()
    assert mean_path.exists()

    out = {
        "best": model.load_events(str(best_path))[0],
        "mean": model.load_events(str(mean_path))[0],
    }
    return out


def read_resulting_events(report_path, result_type="best"):
    """
    Read the pyrocko events from the directory.

    Output is {run_id: {event_id: {best: event, mean: event}}}
    """
    out = defaultdict(lambda: defaultdict(dict))
    exclude_dirs = {"css", "js", "templates"}
    run_paths = (
        x for x in report_path.glob("*") if x.is_dir() and x.name not in exclude_dirs
    )
    for run_path in run_paths:
        event_paths = (x for x in run_path.glob("*") if x.is_dir())
        for event_path in event_paths:
            expected_str = f"*.{result_type}.yaml"
            my_path = list(event_path.glob(expected_str))
            assert len(my_path) == 1
            events = model.load_events(str(my_path[0]))
            out[run_path.name][event_path.name] = events
    return out
