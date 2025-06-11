"""
Run this file to redo all processing and figure creation.
"""

import importlib
import re
import time
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

pattern = r"^[A-Za-z]\d{3}_"


def collect_jobs(base_path):
    """Collect the names of modules to import and run."""
    for path in sorted(base_path.glob("*.py")):
        name = path.stem
        if re.match(pattern, name):
            yield name


def run_job(module_name):
    """Run the job and return the time it took."""
    now = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    duration = time.time() - now
    return duration


class ElkProgress:
    """Simple class for handling progress display."""

    refresh_rate = 5

    def __init__(self, jobs):
        self.jobs = jobs

    def get_progress(self):
        """Create a progress bar for given tasks."""
        progress = Progress(
            SpinnerColumn(style="yellow"),
            TextColumn("[bold blue]{task.fields[task_name]}"),
            TimeElapsedColumn(),
        )
        task_ids = {}
        for task in self.jobs:
            task_id = progress.add_task("", task_name=task, start=False)
            task_ids[task] = task_id

        return progress, task_ids

    def get_completed_text(self, task_name, duration):
        """Get text for the completed task."""
        return (
            f"[green]✔ {task_name}[/green] completed in "
            f"[bold]{duration:.2f}[/bold] seconds"
        )

    def __call__(self):
        """Run all the jobs."""
        completed_texts = []
        progress, task_ids = self.get_progress()
        with Live(Group(progress), refresh_per_second=self.refresh_rate) as live:
            for task_name, task_id in task_ids.items():
                progress.start_task(task_id)
                duration = run_job(task_name)
                progress.remove_task(task_id)
                text = self.get_completed_text(task_name, duration)
                completed_texts.append(text)
                live.update(Group(*completed_texts, progress))


class SimpleProgress:
    """A simple progress display to make debugging easier."""

    def __init__(self, jobs):
        self.jobs = jobs

    def __call__(self):
        """Run the progress monitor"""
        start = time.now()
        for job in self.jobs:
            print(f"Starting on {job}")  # noqa
            duration = run_job(job)
            print(f"Finished {job} in {duration:.2f} seconds")  # noqa
        print("Finsihed all jobs in {time.time() - start:.02f} seconds.")  # noqa


if __name__ == "__main__":
    print("Running all Elk Creek Processing ....")  # noqa
    here = Path(__file__).parent
    jobs = sorted(collect_jobs(here))
    ElkProgress(jobs)()
    # SimpleProgress(jobs)()
