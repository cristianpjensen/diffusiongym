"""Utility classes and functions for Value Matching."""


class Report:
    """A simple class to keep track of statistics during training and logging them during training.

    It keeps track of every update and it keeps track of running statistics. The running statistics
    are reset at every log (call to __str__).
    """

    def __init__(self) -> None:
        self.stats: dict[str, list[float]] = {}
        self.running_means: dict[str, tuple[float, int]] = {}

    def update(self, **kwargs: float) -> None:
        """Update statistics with new values."""
        for key, value in kwargs.items():
            if key not in self.stats:
                self.stats[key] = []

            self.stats[key].append(value)

            running_mean, num_updates = self.running_means.get(key, (0, 0))
            self.running_means[key] = (running_mean + value, num_updates + 1)

    def __str__(self) -> str:
        strings = []
        for key, (running_mean, num_updates) in self.running_means.items():
            avg_value = running_mean / num_updates
            strings.append(f"{key}: {avg_value:.3f}")

        # Reset
        self.running_means = {}

        return ", ".join(strings)
