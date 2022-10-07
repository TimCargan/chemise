from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging, flags
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from jax import lax
from jaxtyping import Num, Array
from tensorflow import data as tfd  # Only for typing

from chemise.callbacks.abc_callback import StepCallback
from chemise.traning import BasicTrainer
from chemise.traning.prefetch import get_batch_size, Prefetch_dev

FLAGS = flags.FLAGS

Result = dict[str, Num[Array, ""]]
State_Result = Tuple[TrainState, Result]
Features = dict[str, Num[Array, "..."]]
Batch = Tuple[Features, Features]
Rand_Dict = dict[str, jax.random.PRNGKeyArray]
P_Func = Callable[[TrainState, Batch, Rand_Dict], State_Result]


@dataclass(unsafe_hash=True)
class VectorTrainer(BasicTrainer):

    # @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    @partial(jax.pmap, static_broadcasted_argnums=(0,), in_axes=(None, 0, 0, 0, 0), axis_name="batch")
    @partial(jax.vmap, in_axes=(None, 0, 1, None, 1))
    def p_train_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict, mask: bool) -> State_Result:
        """
        Train for a single step.
        TODO:
         - support multiple output / multi loss via dicts akin to keras
        Notes:
            In order to keep this a pure function, we don't update the `self.state` just return a new state
        """
        new_state, metrics = self._p_train_step(state, batch, rngs, mask)
        new_state = lax.cond(jnp.any(mask), lambda on: on[0], lambda on: on[1], (new_state, state))
        return new_state, metrics


    def p_apply_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict = None) -> Tuple[Features, ...]:
        """
        Apply model to a batch of data returning
        :param state: model state object
        :param batch: Batch tuple to predict with
        :param rngs: dict of rngs for use in the model
        :return: tuple of [X, Y, Y_hat]
        """
        return self._p_apply_step(state, batch, rngs)

    @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    @partial(jax.vmap, in_axes=(None, 0, 0, None))
    def p_test_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        """
        Perform a prediction step and calculate metrics for a given batch
        :param state: model state object
        :param batch: Batch tuple to predict with
        :param rngs: dict of rngs for use in the model
        :return: [State, dict metrics]
        """
        return self._p_test_step(state, batch, rngs)

    @staticmethod
    def get_first_el(data: tfd.Dataset):
        first, _ = next(data.take(1).as_numpy_iterator())
        return first

    @partial(jax.pmap, static_broadcasted_argnums=(0,), in_axes=(None, None, 0, 0), axis_name="batch")
    @partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def jax_if_merge(self, mask, old, new):
        old_new = (old, new)
        results = lax.cond(mask, lambda on: on[0], lambda on: on[1], old_new)
        return results

    def _stateful_step_runner(self, data: tfd.Dataset, step_fn: P_Func, hist: list, callback: StepCallback,
                              training=True) -> None:
        """
        A standard step call, helpful to reduce code in the main train loops
        :param training: 
        :param data: data to iterate over
        :param step_fn: the step function to call, must be
        :param hist:
        :param callback: StepCallback object
        :return:
        """
        callback.start_cb(self)
        d_iter = data.as_numpy_iterator()
        d_iter = Prefetch_dev(d_iter, buffer_size=FLAGS.prefetch_buffer).iter(with_meta=True, batch_dims=2)
        # Replicate state to all devices, use this ref over self.state to reduce / broadcast calls
        r_state = replicate(self.state)
        raw_rngs = self._make_rngs()
        rngs = replicate(raw_rngs)
        dev_batch_size = get_batch_size(r_state)

        step = int(np.max(self.state.step))
        while True:
            callback.step_start_cb(self)
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                if not (batch := next(d_iter, None)):
                    break

                batch, mask = batch
                np_mask = np.array(mask)

                s = get_batch_size(batch)
                _r_state = r_state if s == dev_batch_size else self.slice(r_state, s)
                _rngs = rngs if s == dev_batch_size else self.slice(rngs, s)

                r_state, r_metrics = step_fn(_r_state, batch, _rngs, np_mask)

                # un-replicate so callbacks and metrics work
                self.state, metrics = unreplicate((r_state, r_metrics))

                # un-replicate and re-broadcast for state
                r_state = r_state if s == dev_batch_size else replicate(self.state)

                # Mask out bad results with Nans
                # model_ok_state = jax.tree_util.tree_map(lambda x: x[0, 0], r_state)
                # bad_batch = jax.tree_util.tree_map(lambda x: x[0, :, 2], batch)
                # output = self._p_train_step(model_ok_state, bad_batch, {'lstm_cell': jax.random.PRNGKey(0)},
                #                             mask[0, :, 2])
                if not np.all(mask):
                    # model_ok_state = jax.tree_util.tree_map(lambda x: x[0, 0], r_state)
                    # bad_batch = jax.tree_util.tree_map(lambda x: x[0, :, 2], batch)
                    # output = self._p_dtrain_step(model_ok_state, bad_batch, {'lstm_cell': jax.random.PRNGKey(0)}, mask[0, :, 2])
                    pass
                #     nan_mask = np.array([1.0 if m else np.NAN for m in mask])
                #     metrics = jax.tree_util.tree_map(lambda x: x * nan_mask, metrics)

                hist.append(metrics)
                step += 1
                callback.step_end_cb(self)

        logging.info("Internal step count: %d", step)
        callback.end_cb(self)

    def epoc_metrics(self):
        logging.info(f"Vector Steps: {self.state.step}")
        return super(VectorTrainer, self).epoc_metrics()


def merge_trees(ls):
    tree = jax.tree_util.tree_structure(ls[0])
    flat = [jax.tree_util.tree_flatten(d)[0] for d in ls]
    stacked = _merge(flat)
    return jax.tree_util.tree_unflatten(tree, stacked)

@jax.jit
def _merge(flat):
    flat_n = [[n[l] for n in flat] for l in range(len(flat[0]))]
    stacked = [jnp.stack(x) for x in flat_n]
    return stacked
