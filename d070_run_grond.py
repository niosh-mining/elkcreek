"""
Run each of the grond manual_configs.
"""

import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from subprocess import run

from grond.apps.grond import command_go, command_report

import local


def run_grond(config_path, run_path):
    """Run a Grond process."""
    # random delay to avoid locking up squirrel database.
    results_name = config_path.name.replace("$", "").replace(".gronf", ".grun")
    expected_path = run_path / results_name
    if expected_path.exists():
        return

    wait = random.randint(0, 100) / 20
    time.sleep(wait)
    uri = config_path.name.split("_")[0]
    eid = config_path.name.split("_")[-1].replace("$", "").split(".")[0]
    command_go([str(config_path), "--status", "quiet"])
    run_dir = list(local.grond_run_path.glob(f"{uri}_{eid}*"))
    assert len(run_dir) == 1, "no run directory found!"

    # And generate report
    command_report([str(run_dir[0])])


def main():
    """Run Grond in parallel for each config/event."""
    # Note: the version of Grond I developed this on didn't properly perform
    # multiprocessing. This is no longer true, but it was easier to leave the code
    # the same.
    executor = ProcessPoolExecutor(max_workers=20)
    # path = local.grond_configs_path
    config_path = local.grond_configs_path
    run_path = local.grond_run_path
    runs = []

    for path in config_path.rglob("*.gronf"):
        # uncomment for debugging
        # run_grond(path, run_path)
        future = executor.submit(run_grond, path, run_path)
        runs.append(future)

    # This just forces all futures to complete.
    for fut in as_completed(runs):
        pass

    time.sleep(10)
    run(f"grond report -so --parallel=10 {run_path}/*", shell=True)


if __name__ == "__main__":
    main()
