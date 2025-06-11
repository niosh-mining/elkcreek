import random
import time

from rich.console import Group
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Simulated indeterminate tasks
tasks = [
    "Loading data",
    "Processing images",
    "Training model",
    "Validating results",
    "Saving outputs",
]

# Track start and end times
task_durations = {}

# Create progress bars
progress = Progress(
    SpinnerColumn(style="yellow"),
    TextColumn("[bold blue]{task.fields[task_name]}"),
    TimeElapsedColumn(),
)

completed_texts = []


def run_task(task_name):
    start_time = time.time()
    # Simulate a task taking between 1 and 5 seconds
    time.sleep(random.uniform(1, 5))
    end_time = time.time()
    duration = end_time - start_time
    return duration


with Live(Group(progress), refresh_per_second=10) as live:
    task_ids = {}
    for task_name in tasks:
        task_id = progress.add_task("", task_name=task_name, start=False)
        task_ids[task_name] = task_id

    for task_name in tasks:
        task_id = task_ids[task_name]
        progress.start_task(task_id)
        duration = run_task(task_name)
        progress.remove_task(task_id)
        completed_texts.append(
            f"[green]✔ {task_name}[/green] completed in [bold]{duration:.2f}[/bold] seconds"
        )
        # Update live display with completed tasks below the spinner
        live.update(Group(progress, *completed_texts))
