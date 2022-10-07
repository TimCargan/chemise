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
    @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    @partial(jax.vmap, in_axes=(None, 0, 0, None))
    def p_train_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        """
        Train for a single step.
        TODO:
         - support multiple output / multi loss via dicts akin to keras
        Notes:
            In order to keep this a pure function, we don't update the `self.state` just return a new state
        """
        new_state, metrics = self._p_train_step(state, batch, rngs)
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

    @partial(jax.jit, static_argnums=(0,2))
    def _slice(self, r_f_state, s):
        return [x[:s] for x in r_f_state]
    def slice(self, r_state, s):
        leaves, treedef = jax.tree_util.tree_flatten(r_state)
        leaves = self._slice(leaves, s)
        return treedef.unflatten(leaves)

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
        last_state = self.state
        r_state = replicate(self.state)
        raw_rngs = self._make_rngs()
        rngs = replicate(raw_rngs)
        dev_batch_size = get_batch_size(r_state)

        step_shape = np.shape(self.state.step)
        to_populate = [True] * step_shape[0]
        saved_states = [None] * step_shape[0]

        step = int(np.max(self.state.step))
        while True:
            callback.step_start_cb(self)
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                if not (batch := next(d_iter, None)):
                    break

                batch, mask = batch
                np_mask = np.array(mask)
                # np_mask = np_mask.reshape((-1,1))
                # If a mask bit has flipped, save the current state for that vector dim,
                # for now we are lazy and just save it all, don't have to worry about edge cases etc etc
                toggled = np.logical_xor(mask, to_populate)

                # if np.any(toggled) and training:
                #     merge_state = False
                #     _states = [None] * step_shape[0]
                #     for i, cond in enumerate(toggled):
                #         if cond:
                #             if to_populate[i]:
                #                 # Mask swapped from True to False so data is now bad for this run
                #                 saved_states[i] = last_state
                #                 to_populate[i] = False
                #                 logging.debug("Saving state %d", i)
                #             if mask[i]:
                #                 # Swap from False to True, so we need to merge the states back in
                #                 merge_state = True
                #                 to_populate[i] = True
                #                 _states[i] = saved_states[i]
                #                 saved_states[i] = None
                #
                #     # if reload state
                #     if merge_state:
                #         logging.debug(f"Reloading using mask: {mask}")
                #         states = [s if s is not None else last_state for s in _states]
                #         v_states = [jax.tree_util.tree_map(lambda x: x[i], v) for i, v in enumerate(states)]
                #         self.state = merge_trees(v_states)
                #         r_state = replicate(self.state)

                # if (s := get_batch_size(batch)) < dev_batch_size:
                s = get_batch_size(batch)
                _r_state = self.slice(r_state, s)
                _rngs = self.slice(rngs, s)  # jax.tree_util.tree_map(lambda x: x[:s], rngs)
                new_r_state, r_metrics = step_fn(_r_state, batch, _rngs)

                if s < dev_batch_size:
                    _state = unreplicate(new_r_state)  # un-replicate and re-broadcast for state
                    new_r_state = replicate(_state)

                # else:
                #     new_r_state, r_metrics = step_fn(r_state, batch, rngs)

                r_state = self.jax_if_merge(mask, new_r_state, r_state)
                # un-replicate so callbacks and metrics work
                self.state, metrics = unreplicate((r_state, r_metrics))

                # Mask out bad results with Nans
                if np.all(mask):
                    nan_mask = np.array([1.0 if m else np.NAN for m in mask])
                    metrics = jax.tree_util.tree_map(lambda x: x * nan_mask, metrics)

                hist.append(metrics)
                step += 1
                callback.step_end_cb(self)

        # Merge state
        states = [s if s is not None else self.state for s in saved_states]
        v_states = [jax.tree_util.tree_map(lambda x: x[i], v) for i, v in enumerate(states)]
        self.state = merge_trees(v_states)

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
