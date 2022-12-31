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

    def on_fit_start(self, trainer: BasicTrainer):
        if self.auto_restore:
            logging.warning("Restoring checkpoint at start of run")
            trainer.state = cp.restore_checkpoint(self.ckpt_dir, trainer.state)

    def on_train_start(self, trainer):
        self.train_c = 0

    def on_train_batch_end(self, trainer: BasicTrainer):
        self.train_c += 1
        if self.intra_train_freq and self.train_c % self.intra_train_freq:
            cp.save_checkpoint(target=trainer.state, step=trainer.state.step,
                               ckpt_dir=self.ckpt_dir, overwrite=self.overwrite,
                               keep=self.keep, keep_every_n_steps=self.keep_every_n_steps)

    def on_epoch_end(self, trainer: BasicTrainer):
        cp.save_checkpoint(target=trainer.state, step=trainer.state.step,
                           ckpt_dir=self.ckpt_dir, overwrite=self.overwrite,
                           keep=self.keep, keep_every_n_steps=self.keep_every_n_steps)

    @staticmethod
    def restore(trainer: BasicTrainer, ckpt_dir: Path | str):
        trainer.state = cp.restore_checkpoint(ckpt_dir=ckpt_dir, target=trainer.state)
        return trainer
