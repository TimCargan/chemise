from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from chemise.callbacks.abc_callback import Callback

if TYPE_CHECKING:
    from chemise.traning import BasicTrainer

@dataclass
class Profile(Callback):
    """
    This profile callback

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
        if isinstance(self.steps, int):
            self.steps = (self.steps, self.steps+self.steps)

    def on_train_batch_start(self, trainer: BasicTrainer):
        if self._step_count == self.steps[0]:
            logging.info("Starting Profiler, output dir: %s", self.profile_dir)
            jax.profiler.start_trace(self.profile_dir)
            self._running = True

        self.step_trace= jax.profiler.StepTraceAnnotation("train", step_name=f"train {self._step_count}",
                                         step_num=self._step_count, group_id=self._group_id)
        self.step_trace.__enter__()

    def on_train_batch_end(self, trainer: BasicTrainer):
        self._step_count += 1
        if self._step_count > self.steps[1] and self._running:
            self._group_id += 1
            logging.info("Stopping Profiler")
            jax.profiler.stop_trace()
            self._running = False

        self.step_trace.__exit__()