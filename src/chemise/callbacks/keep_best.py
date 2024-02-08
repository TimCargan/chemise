from __future__ import annotations

import datetime
from dataclasses import dataclass
from flax.training.train_state import TrainState
from pathlib import Path
from typing import TYPE_CHECKING

import orbax.checkpoint
from absl import logging
from flax.training import orbax_utils
from jax import numpy as jnp
import jax
from jaxtyping import Float, Bool, Array

from chemise.callbacks.abc_callback import Callback, EarlyStopping

if TYPE_CHECKING:
    from chemise.traning.basic_trainer import BasicTrainer



@dataclass
class KeepBest(Callback):

    best_state: TrainState = None
    best_value: Float[Array, ""] = None
    monitor_metric: str = "val_loss"
    improvement_size: float = 0.0
    minimise: bool = True
    patience_steps: int = 0
    _step_count: int = 0
    _es: Bool[Array, ""] = None
    def on_epoch_end(self, trainer: BasicTrainer):
        cur_value = trainer.train_hist["train"][-1][self.monitor_metric]
        cur_value = cur_value if self.minimise else cur_value * -1  # Negate cur value if we are maximising the target
        if self.best_value is None:
            local_state = jax.tree_map(lambda l: jax.device_get(l), trainer.state)
            self.best_state = local_state
            self.best_value = cur_value
            self._es = jnp.zeros_like(self.best_value).astype(bool)

        def _eval(best_value, cur_value, best_state, cur_state, early_stop_mask):
            """Outer eval to run the cond for the early stopping mask.

             This is needed to support the case where some models in a vmap contex run out of patience before others
             and so would trigger an early stopping"""
            def __eval(best_value, cur_value, best_state, cur_state):
                """Inner eval to run the cond for if there is an improvement"""
                best_state, best_value = jax.lax.cond((best_value - cur_value) > self.improvement_size,
                                                      lambda: (cur_state, cur_value),
                                                      lambda: (best_state, best_value))

                diff = cur_state.step - best_state.step
                es = self.patience_steps < diff
                return best_value, best_state, es

            return jax.lax.cond(early_stop_mask,
                                lambda bv, _, bs, *__: (bv, bs, early_stop_mask),  # If true
                                __eval,                                            # If false
                                best_value, cur_value, best_state, cur_state)      # Params



        _eval = jax.vmap(_eval) if len(cur_value.shape) > 0 else _eval

        self.best_value, self.best_state, self._es = _eval(self.best_value, cur_value,
                                                           self.best_state, trainer.state, self._es)
        self.best_state = jax.tree_map(lambda l: jax.device_get(l), self.best_state)

        if self.patience_steps > 0:
            if jnp.all(self._es):
                raise EarlyStopping("Patience expired in keep best. Note that patince_steps is train steps not number of epochs")
