import abc
import logging

logger = logging.getLogger(__name__)


class Task(abc.ABC):
    """Abstract base class for defining tasks."""

    def __init__(self, config: dict):
        self._config = config
        self._progress_source_list: list[float] = []  # Reference to strategy's list
        # Store checkpoint size for migration calculations
        self.checkpoint_size_gb: float = config.get("checkpoint_size_gb", 50.0)
        self.reset()

    def set_progress_source(self, task_done_time_list: list[float]):
        """Set the external list used as the source of truth for progress."""
        self._progress_source_list = task_done_time_list

    def get_current_progress_seconds(self) -> float:
        """Calculate current progress in seconds from the source list."""
        return sum(self._progress_source_list or [])

    def get_remaining_duration_seconds(self) -> float:
        """Calculate remaining duration in seconds."""
        # Use the new seconds-based total duration method
        return max(
            0.0, self.get_total_duration_seconds() - self.get_current_progress_seconds()
        )

    @abc.abstractmethod
    def reset(self):
        """Reset the internal state of the task (if any)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_done(self) -> bool:
        """Return True if the task is completed based on progress source."""
        raise NotImplementedError

    def get_total_duration_hours(self) -> float:
        """Return the total compute duration required for the task in hours."""
        return self.get_total_duration_seconds() / 3600.0

    @abc.abstractmethod
    def get_total_duration_seconds(self) -> float:
        """Return the total compute duration required for the task in seconds."""
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return the configuration dictionary for the task."""
        return self._config

    @abc.abstractmethod
    def get_info(self) -> dict:
        """Return current state information about the task."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self._config})"

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return a user-friendly string representation."""
        raise NotImplementedError


class SingleTask(Task):
    """A task consisting of a single compute duration requirement."""

    def __init__(self, config: dict):
        """Initialize SingleTask.

        Args:
            config: Dictionary containing 'duration' (float) in hours.
        """
        assert (
            "duration" in config
            and isinstance(config["duration"], (int, float))
            and config["duration"] > 0
        ), "SingleTask config must contain a positive numeric 'duration' key."
        # Store duration internally in seconds
        self._duration_seconds = float(config["duration"]) * 3600.0
        super().__init__(config)

    def reset(self):
        """No internal state to reset for SingleTask progress."""
        pass

    @property
    def is_done(self) -> bool:
        """Check if progress from source list meets duration."""
        return self.get_remaining_duration_seconds() <= 1e-3

    def get_total_duration_hours(self) -> float:
        """Return the total duration required in hours."""
        return self._duration_seconds / 3600.0

    def get_total_duration_seconds(self) -> float:
        """Return the total duration required in seconds."""
        return self._duration_seconds

    def get_info(self) -> dict:
        """Return simplified progress derived from source list."""
        return {
            "task_type": "SingleTask",
            "Done(seconds)": self.get_current_progress_seconds(),
            "Remaining(seconds)": self.get_remaining_duration_seconds(),
            "task_is_done": self.is_done,
            # Optional: Keep target for context?
            "Target(seconds)": self.get_total_duration_seconds(),
        }

    def __str__(self) -> str:
        # Display uses hours
        return f"{self.__class__.__name__}({self.get_total_duration_hours():.1f}h)"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("--- Single Task Example ---")
    task_config = {"duration": 10.0}
    s_task = SingleTask(task_config)
    s_progress_list: list[float] = []  # Add type hint
    s_task.set_progress_source(s_progress_list)
    logger.info(f"Task: {s_task}, Done: {s_task.is_done}, Info: {s_task.get_info()}")
    s_progress_list.append(18000.0)  # 5 hours
    logger.info(f"Task: {s_task}, Done: {s_task.is_done}, Info: {s_task.get_info()}")
    s_progress_list.append(21600.0)  # 6 hours
    logger.info(f"Task: {s_task}, Done: {s_task.is_done}, Info: {s_task.get_info()}")
    s_progress_list.clear()
    logger.info(f"Task: {s_task}, Done: {s_task.is_done}, Info: {s_task.get_info()}")

    logger.info("\n--- Chained Task Example ---")
    chained_config = {
        "sub_tasks": [
            {"duration": 5.0},
            {"duration": 8.0},
        ]
    }
    c_task = ChainedTask(chained_config)
    c_progress_list: list[float] = []  # Add type hint
    c_task.set_progress_source(c_progress_list)
    logger.info(
        f"Task: {c_task}, TotalDuration: {c_task.get_total_duration_hours()}h, Done: {c_task.is_done}"
    )
    logger.info(f"Info: {c_task.get_info()}")
    c_progress_list.append(10800.0)  # 3 hours
    logger.info(f"Task: {c_task}, Done: {c_task.is_done}")
    logger.info(f"Info: {c_task.get_info()}")
    c_progress_list.append(14400.0)  # 4 hours (total 7 hours)
    logger.info(f"Task: {c_task}, Done: {c_task.is_done}")
    logger.info(f"Info: {c_task.get_info()}")
    c_progress_list.append(28800.0)  # 8 hours (total 15 hours)
    logger.info(f"Task: {c_task}, Done: {c_task.is_done}")
    logger.info(f"Info: {c_task.get_info()}")
    c_progress_list.clear()
    logger.info(f"Task: {c_task}, Done: {c_task.is_done}")
    logger.info(f"Info: {c_task.get_info()}")
