from dataclasses import asdict
from pathlib import Path
import time

from codecarbon import EmissionsTracker
import pandas as pd

from .logger_utils import get_logger


logging = get_logger(__name__)


# Singleton class to track emissions, responsible for tracking emissions for the entire project
class SingletonEmissionsTracker:
    _instance = None
    task_results = {}
    most_recently_started_task = None

    def __new__(
        cls,
        experiment_id: str,
        output_dir: Path,
        log_level: str = "info",
        project_name: str = __file__,
    ):
        if cls._instance is None:
            cls._instance = EmissionsTracker(
                project_name=project_name,
                experiment_id=experiment_id,
                output_dir=output_dir,
                log_level=log_level,
                output_file="emissions.csv",
                gpu_ids=[0],
                tracking_mode="process",
            )
        return cls._instance

    @staticmethod
    def log_task_results():
        emissions_dict = {}
        for task_id, result in SingletonEmissionsTracker.task_results.items():
            if result is not None:
                emissions = result.emissions
            else:
                emissions = None
            emissions_dict[task_id] = emissions
        logging.info(f"Emissions: {emissions_dict}")

    @staticmethod
    def create_dataframe_from_task_results():
        data = []
        for task, result in SingletonEmissionsTracker.task_results.items():
            if result is not None:
                result_dict = asdict(result)
                result_dict["Task"] = task
                data.append(result_dict)
            else:
                data.append(
                    {
                        "Task": task,
                        "Duration": None,
                        "Energy Consumed": None,
                        "Emissions": None,
                        "Country Name": None,
                        "Country ISO Code": None,
                        "Region": None,
                        "Timestamp": None,
                    }
                )
        df = pd.DataFrame(data)

        # Reorder columns to make 'Task' the first column
        df = df.reindex(columns=["Task"] + [col for col in df.columns if col != "Task"])
        return df

    @staticmethod
    def update_current_task(task_id):
        SingletonEmissionsTracker.most_recently_started_task = task_id
        logging.info(f"Updated the current task to {task_id}")

    @staticmethod
    def track_emissions_decorator(task_id):
        def wrapper(func):
            def inner(*args, **kwargs):
                tracker = SingletonEmissionsTracker._instance
                logging.info(f"Starting task {task_id} at {time.time()}")
                try:
                    tracker.start_task(task_id)
                    result = func(*args, **kwargs)
                except ZeroDivisionError:
                    logging.error(f"ZeroDivisionError occurred during task {task_id}")
                    result = None
                finally:
                    # Initialize the key with a default value
                    SingletonEmissionsTracker.task_results[task_id] = None
                    # Then try to assign the result of stop_task(task_id) to the key
                    try:
                        SingletonEmissionsTracker.task_results[task_id] = (
                            tracker.stop_task()
                        )
                    except Exception as e:
                        logging.error(f"An error occurred while stopping the task: {e}")
                    logging.info(f"Stopped task {task_id} at {time.time()}")
                return result

            return inner

        return wrapper

    @staticmethod
    def start_task(task_id, max_attempts=10):
        logging.info(f"Attempting to start task {task_id} at {time.time()}")
        tracker = SingletonEmissionsTracker._instance

        for attempt in range(max_attempts):
            try:
                tracker.start_task(task_id)
            except Exception as e:
                logging.error(
                    f"An error occurred while starting the task {task_id} on attempt {attempt + 1}: {e}"
                )
                if attempt + 1 == max_attempts:
                    logging.error(
                        f"Failed to start task {task_id} after {max_attempts} attempts"
                    )
                    break
            else:
                SingletonEmissionsTracker.update_current_task(
                    task_id
                )  # Update the current task only if start_task is successful
                logging.info(f"Started task {task_id}")  # Log the task ID
                break

    @staticmethod
    def stop_specfic_task(task_id):
        logging.info(f"Attempting to stop the task {task_id} at {time.time()}")
        tracker = SingletonEmissionsTracker._instance
        try:
            SingletonEmissionsTracker.task_results[task_id] = tracker.stop_task(task_id)
            logging.info(f"Stopped task {task_id} at {time.time()}")
        except Exception as e:
            logging.error(f"An error occurred while stopping the task: {e}")

    @staticmethod
    def stop_current_task():
        logging.info(
            f"Attempting to stop the task {SingletonEmissionsTracker.most_recently_started_task} at {time.time()}"
        )
        tracker = SingletonEmissionsTracker._instance
        task_id = (
            SingletonEmissionsTracker.most_recently_started_task
        )  # Get the current task
        try:
            SingletonEmissionsTracker.task_results[task_id] = tracker.stop_task(task_id)
            logging.info(f"Stopped task {task_id} at {time.time()}")
            SingletonEmissionsTracker.most_recently_started_task = None
        except Exception as e:
            logging.error(f"An error occurred while stopping the task: {e}")

    @staticmethod
    def start_tracker():
        tracker = SingletonEmissionsTracker._instance
        try:
            tracker.start()
            logging.info("Started the emissions tracker.")
        except Exception as e:
            logging.error(
                f"An error occurred while starting the emissions tracker: {e}"
            )

    @staticmethod
    def stop_tracker():
        tracker = SingletonEmissionsTracker._instance
        tracker.stop()
        return tracker

    @staticmethod
    def get_task_results():
        return SingletonEmissionsTracker.task_results
