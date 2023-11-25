from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import orbax.checkpoint
from absl import logging
from flax.training import orbax_utils
from jax import numpy as jnp

from chemise.callbacks.abc_callback import Callback

if TYPE_CHECKING:
    from chemise.traning.basic_trainer import BasicTrainer


@dataclass
class Checkpointer(Callback):
    ckpt_dir: str
    keep: int = 1
    overwrite: bool = False
    keep_every_n_steps: int = None
    keep_time_interval: datetime.timedelta = None
    intra_train_freq: int = None
    intra_train_freq_time: datetime.timedelta = None
    auto_restore: bool = False  # Restore the most recent ckpt when training begins
    keep_epoch_steps: bool = False  # Keep the step number for every epoch
    epoch_keep_period: int = None  # Freq of epoch steps save
    save_epochs: bool = True  # Save epochs as their own checkpoints with a prefix epoch
    save_ckpt_on_epoch_end: bool = True  # Save a step ckpt on epoch end
    _step_count: int = 0
    _epoch: int = 0
    _save_args = None  # A mapping to let the checkpoint manager know how to compress the ckpt
    _last_ckpt_time = None

    def __post_init__(self):
        mgr_options = orbax.checkpoint.CheckpointManagerOptions(
            create=True, max_to_keep=self.keep, keep_time_interval=self.keep_time_interval,
            keep_period=self.keep_every_n_steps, step_prefix='ckpt')
        self.checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        self.ckpt_mgr = orbax.checkpoint.CheckpointManager(self.ckpt_dir, self.checkpointer, mgr_options)

        epoch_dir = f"{self.ckpt_dir}/epoch"
        mgr_options = orbax.checkpoint.CheckpointManagerOptions(create=True, keep_period=self.epoch_keep_period, step_prefix='epoch')
        self.epoch_ckpt_mgr = orbax.checkpoint.CheckpointManager(epoch_dir, self.checkpointer, mgr_options)

    def set_step_number(self, step: int):
        self._step_count = step

    def on_fit_start(self, trainer: BasicTrainer):
        # Set the save args on fit begin to reduce the number of calls
        self._save_args = orbax_utils.save_args_from_target(trainer.state)
        self._last_ckpt_time = datetime.datetime.now()
        if self.auto_restore:
            logging.warning("Restoring checkpoint at start of run")
            step = self.ckpt_mgr.latest_step()
            trainer.state = self.ckpt_mgr.restore(step, items=trainer.state)
            self.set_step_number(int(jnp.max(trainer.state.step)))

    def on_train_batch_end(self, trainer: BasicTrainer):
        self._step_count += 1 # Use of an internal step count to Dev to Host call
        if ((self.intra_train_freq and self._step_count % self.intra_train_freq == 0) or
                (self.intra_train_freq_time and datetime.datetime.now() - self._last_ckpt_time > self.intra_train_freq_time)):
            self.save(trainer)

    def on_epoch_end(self, trainer: BasicTrainer):

        if self.keep_epoch_steps:
            # This is a hack and I should find a better way
            step = int(jnp.max(trainer.state.step))
            self.ckpt_mgr._options.save_on_steps.append(step)

        if self.save_epochs:
            self.epoch_ckpt_mgr.save(self._epoch, trainer.state, save_kwargs={'save_args': self._save_args}, force=True)

        if self.save_ckpt_on_epoch_end:
            self.save(trainer)

        self._epoch += 1

    def save(self, trainer: BasicTrainer, force: bool = False):
        step = int(jnp.max(trainer.state.step))
        self.ckpt_mgr.save(step, trainer.state, save_kwargs={'save_args': self._save_args}, force=force)
        self._last_ckpt_time = datetime.datetime.now()

    @staticmethod
    def restore(trainer: BasicTrainer, ckpt_dir: Path | str, step_prefix: str = "ckpt", use_restore_kwargs: bool = True):
        print(f"Restore from {ckpt_dir}")
        logging.warning(f"Restore from {ckpt_dir}")
        ckpter = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        mgr_options = orbax.checkpoint.CheckpointManagerOptions(step_prefix=step_prefix)
        ckpt_mgr = orbax.checkpoint.CheckpointManager(ckpt_dir, ckpter, mgr_options)

        step = ckpt_mgr.latest_step()
        # TODO: find out more about the kwargs
        restore_args = orbax_utils.restore_args_from_target(trainer.state, mesh=None)
        restore_kwargs = {'restore_args': restore_args} if use_restore_kwargs else None
        trainer.state = ckpt_mgr.restore(step, items=trainer.state, restore_kwargs=restore_kwargs)
        return trainer
