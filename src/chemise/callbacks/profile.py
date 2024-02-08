"""JAX profile callback."""
from __future__ import annotations

import jax
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chemise.callbacks.abc_callback import Callback

if TYPE_CHECKING:
    from chemise.traning import BasicTrainer


@dataclass
class Profile(Callback):
    """This profile callback.

    Args:
        profile_dir: Directory to write profile too
        steps: a tuple the step number to start and stop profiling from
    """
    # Output dir
    profile_dir: str
    # Steps to profile
    steps: int | tuple[int, int] = 10

    _running: bool = False
    # Use an internal step count, accessing the state step count is slow (500ms vs 1ms).
    _step_count: int = 0
    _group_id: int = 0

    def __post_init__(self):
        """Set the steps number correctly if only given an int."""
        if isinstance(self.steps, int):
            self.steps = (self.steps, self.steps + self.steps)

    def on_train_batch_start(self, trainer: BasicTrainer):
        """Profile entry point.

        Args:
          trainer: The training object
        """
        if self._step_count == self.steps[0]:
            logging.info("Starting Profiler, output dir: %s", self.profile_dir)
            jax.profiler.start_trace(self.profile_dir)
            self._running = True

    def on_train_batch_end(self, trainer: BasicTrainer):
        """Profile exit point.

        Args:
          trainer: The training object
        """
        self._step_count += 1
        if self._step_count > self.steps[1] and self._running:
            trainer._group_id += 1
            logging.info("Stopping Profiler")
            jax.profiler.stop_trace()
            self._running = False
