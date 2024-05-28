from pathlib import Path
from codecarbon import EmissionsTracker

def create_emissions_tracker(
    experiment_id: str,
    output_dir: Path,
    project_name: str = __file__,
) -> callable:
    def track_emissions_decorator(task_id):
        def wrapper(func):
            def inner(*args, **kwargs):
                tracker = EmissionsTracker(
                    project_name=project_name,
                    experiment_id=experiment_id,
                    output_dir=output_dir,
                    output_file="emissions.csv",
                )
                tracker.start_task(func.__name__, task_id)
                result = func(*args, **kwargs)
                tracker.stop()
                return result
            return inner
        return wrapper
    return track_emissions_decorator