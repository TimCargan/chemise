from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from absl import logging
from flax.training import checkpoints as cp

from chemise.callbacks.abc_callback import Callback

if TYPE_CHECKING:
    from chemise.traning.basic_trainer import BasicTrainer


@dataclass
class Checkpointer(Callback):
    ckpt_dir: str
    keep: int = 1
    overwrite: bool = False
    keep_every_n_steps: int = None
    intra_train_freq: int = None
    auto_restore: bool = False

    _step_count: int = 0

    def set_step_number(self, step: int):
        self._step_count = step

    def on_fit_start(self, trainer: BasicTrainer):
        if self.auto_restore:
            logging.warning("Restoring checkpoint at start of run")
            trainer.state = cp.restore_checkpoint(self.ckpt_dir, trainer.state)

    def on_train_batch_end(self, trainer: BasicTrainer):
        self._step_count += 1
        if self.intra_train_freq and self._step_count % self.intra_train_freq == 0:
            self.save(trainer)

    def on_epoch_end(self, trainer: BasicTrainer):
        self.save(trainer)

    def save(self, trainer: BasicTrainer):
        orbax_checkpointer = None #orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
        cp.save_checkpoint(target=trainer.state, step=trainer.state.step,
                           ckpt_dir=self.ckpt_dir, overwrite=self.overwrite,
                           keep=self.keep, keep_every_n_steps=self.keep_every_n_steps,
                           orbax_checkpointer=orbax_checkpointer)
    @staticmethod
    def restore(trainer: BasicTrainer, ckpt_dir: Path | str):
        print(f"Restore from {ckpt_dir}")
        orbax_checkpointer = None #orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
        trainer.state = cp.restore_checkpoint(ckpt_dir=ckpt_dir, target=trainer.state,
                                              orbax_checkpointer=orbax_checkpointer)
        return trainer
