"""
Run the moment decomposition based on Rigby 2024.

Note: Alex Rigby from IMS provided much of this code.
"""

from elkcreek.grond import read_resulting_events
from elkcreek.plot import MomentTensorCDCPlotter

import local


def main():
    """Get resulting moment tensors."""
    base_path = local.moment_tensor_plot_path
    best = read_resulting_events(local.grond_report_path, "best")
    ensemble = read_resulting_events(local.grond_report_path, "ensemble")

    run_ids = set(best) & set(ensemble)
    for run_id in run_ids:
        event_ids = set(best[run_id]) & set(ensemble[run_id])
        for event_id in event_ids:
            best_event = best[run_id][event_id][0]
            event_ensemble = ensemble[run_id][event_id]
            cdc = MomentTensorCDCPlotter(
                best_event=best_event,
                event_ensemble=event_ensemble,
                crush_azimuth=local.coal_normal_azimuth,
                crush_plunge=local.coal_normal_plunge,
            )
            fig = cdc()
            path = base_path / f"{run_id}_{event_id}.png"
            fig.savefig(path, dpi=350)


if __name__ == "__main__":
    main()
