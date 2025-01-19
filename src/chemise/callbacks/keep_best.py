"""The callback to track the best version of the model and keep it at the end of each epoch."""
from __future__ import annotations


from functools import partial

import jax
from dataclasses import dataclass
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float
from typing import TYPE_CHECKING

from chemise.callbacks.abc_callback import Callback, EarlyStopping

if TYPE_CHECKING:
    from chemise.traning.basic_trainer import BasicTrainer


@dataclass(unsafe_hash=True)
class KeepBest(Callback):
    """KeepBest callback.

    Track the progress of the model(s) as the train and keep the best version learned so far.

    Args:
        monitor_metric: The metric to monitor
        improvement_size: The amount the score must change to count as an improvement between two models
        minimise: If the target value should be increasing of decreasing
        patience_steps: The number of steps to wait for an improvement before stopping training
        replace: Replace the state in the trainer when fit is complete with the best values
    """
    best_state: TrainState = struct.field(default=None, compare=False)
    best_value: Float[Array, ""] = struct.field(default=None, compare=False)
    monitor_metric: str = "val_loss"
    improvement_size: float = 0.0
    minimise: bool = True
    patience_steps: int = 0
    replace: bool = False
    reset_on_start: bool = True

    _es: Bool[Array, ""] = struct.field(default=None, compare=False)

    def on_fit_start(self, trainer):
        self.best_state = jax.device_get(trainer.state)
        if self.reset_on_start:
            # Resets the best value and patience counter so the callback can be reused
            self.best_value = None
            self.patience_steps = 0

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 3, 5))
    def _eval(self, best_value, cur_value, best_state, cur_state, early_stop_mask):
        """Outer eval to run the cond for the early stopping mask.

        This is needed to support the case where some models in a vmap contex run out of patience before others
        and so would trigger an early stopping
        """

        def __eval(best_value, cur_value, best_state, cur_state):
            """Inner eval to run the cond for if there is an improvement."""
            best_state, best_value = jax.lax.cond((best_value - cur_value) > self.improvement_size,
                                                        lambda: (cur_state, cur_value),  # If improved
                                                        lambda: (best_state, best_value))  # If not improved

            diff = cur_state.step - best_state.step
            es = self.patience_steps < diff
            return best_value, best_state, es

        res = jax.lax.cond(early_stop_mask,
                           lambda bv, _, bs, *__: (bv, bs, early_stop_mask),  # If true
                           __eval,  # If false
                           best_value, cur_value, best_state, cur_state)  # Params

        res = jax.device_get(res)
        return res

    def on_epoch_end(self, trainer: BasicTrainer):
        """Run the keep best on the epoch end.

        Args:
            trainer: The trainer object
        """
        cur_value = trainer.train_hist["train"][-1][self.monitor_metric]
        cur_value = cur_value if self.minimise else cur_value * -1  # Negate cur value if we are maximising the target
        if self.best_value is None:
            local_state = jax.device_get(trainer.state)
            self.best_state = local_state
            self.best_value = cur_value
            self._es = jnp.zeros_like(self.best_value).astype(bool)

        _eval = jax.vmap(self._eval) if len(cur_value.shape) > 0 else self._eval
        self.best_value, self.best_state, self._es = _eval(self.best_value, cur_value,
                                                           self.best_state, trainer.state, self._es)

        if self.patience_steps > 0:
            if jnp.all(self._es):
                raise EarlyStopping("Patience expired in keep best. Note that patince_steps is train steps not number of epochs")

    def on_fit_end(self, trainer):
        """Replace the trainers state with the best state if replace is set.

        Args:
            trainer: The trainer object
        """
        if self.replace:
            trainer.state = self.best_state