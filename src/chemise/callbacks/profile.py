from __future__ import annotations

import logging
from dataclasses import dataclass
import jax

from chemise.callbacks.abc_callback import Callback
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

    def __post_init__(self):
        if isinstance(self.steps, int):
            self.steps = (self.steps, self.steps)

    def on_train_batch_start(self, trainer: BasicTrainer):
        if trainer.state.step == self.steps[0]:
            logging.info("Starting Profiler, output dir: %s)", self.profile_dir)
            jax.profiler.start_trace(self.profile_dir)
            self._running = True

    def on_train_batch_end(self, trainer: BasicTrainer):
        if trainer.state.step > self.steps[1] and self._running:
            logging.info("Stopping Profiler")
            jax.profiler.stop_trace()
            self._running = False
