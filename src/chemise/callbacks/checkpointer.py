from __future__ import annotations

import datetime
import orbax
from absl import logging
from dataclasses import dataclass
from flax.training import orbax_utils
from jax import numpy as jnp
from pathlib import Path
from typing import TYPE_CHECKING

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
    keep_time_interval: datetime.timedelta = None
    _step_count: int = 0
    _save_args = None  # A mapping to let the checkpoint manager know how to compress the ckpt

    def __post_init__(self):
        mgr_options = orbax.checkpoint.CheckpointManagerOptions(
            create=True, max_to_keep=self.keep, keep_time_interval=self.keep_time_interval,
            keep_period=self.keep_every_n_steps, step_prefix='ckpt')
        self.ckpt_mgr = orbax.checkpoint.CheckpointManager(self.ckpt_dir,
                                                           orbax.checkpoint.Checkpointer(
                                                               orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)

    def set_step_number(self, step: int):
        self._step_count = step

    def on_fit_start(self, trainer: BasicTrainer):
        # Set the save args on fit begin to reduce the number of calls
        self._save_args = orbax_utils.save_args_from_target(trainer.state)
        if self.auto_restore:
            logging.warning("Restoring checkpoint at start of run")
            step = self.ckpt_mgr.latest_step()
            trainer.state = self.ckpt_mgr.restore(step, items=trainer.state)
            # trainer.state = cp.restore_checkpoint(self.ckpt_dir, trainer.state)

    def on_train_batch_end(self, trainer: BasicTrainer):
        self._step_count += 1
        if self.intra_train_freq and self._step_count % self.intra_train_freq == 0:
            self.save(trainer)

    def on_epoch_end(self, trainer: BasicTrainer):
        self.save(trainer)

    def save(self, trainer: BasicTrainer):
        # Need to find out what this does
        step = int(jnp.max(trainer.state.step))
        self.ckpt_mgr.save(step, trainer.state, save_kwargs={'save_args': self._save_args})
        # orbax_checkpointer = None #orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
        # cp.save_checkpoint(target=trainer.state, step=trainer.state.step,
        #                    ckpt_dir=self.ckpt_dir, overwrite=self.overwrite,
        #                    keep=self.keep, keep_every_n_steps=self.keep_every_n_steps,
        #                    orbax_checkpointer=orbax_checkpointer)

    @staticmethod
    def restore(trainer: BasicTrainer, ckpt_dir: Path | str, step_prefix: str = "ckpt", use_restore_kwargs: bool = True):
        print(f"Restore from {ckpt_dir}")
        logging.warning(f"Restore from {ckpt_dir}")
        ckpter = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        mgr_options = orbax.checkpoint.CheckpointManagerOptions(step_prefix=step_prefix)
        ckpt_mgr = orbax.checkpoint.CheckpointManager(ckpt_dir, ckpter, mgr_options)

        step = ckpt_mgr.latest_step()
        # TDOD: find out more about the kwargs
        restore_args = orbax_utils.restore_args_from_target(trainer.state, mesh=None)
        restore_kwargs = {'restore_args': restore_args} if use_restore_kwargs else None
        trainer.state = ckpt_mgr.restore(step, items=trainer.state, restore_kwargs=restore_kwargs)
        # orbax_checkpointer = None #orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
        # trainer.state = cp.restore_checkpoint(ckpt_dir=ckpt_dir, target=trainer.state,
        #                                       orbax_checkpointer=orbax_checkpointer)
        return trainer
