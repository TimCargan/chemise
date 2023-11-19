from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import logging, flags
from flax.jax_utils import replicate
from flax.training.train_state import TrainState
from jax import lax
from jaxtyping import Num, Array
from tensorflow import data as tfd  # Only for typing

from chemise.traning import BasicTrainer
from chemise.traning.basic_trainer import Batch, Rand_Dict, State_Result, Features, Result

FLAGS = flags.FLAGS


@dataclass(unsafe_hash=True)
class VectorTrainer(BasicTrainer):
    batch_dims: int = 2

    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0, 0), axis_name="batch")
    @partial(jax.vmap, in_axes=(None, 0, 1, None))
    def p_train_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        """
        Train for a single step.
        TODO:
         - support multiple output / multi loss via dicts akin to keras
        Notes:
            In order to keep this a pure function, we don't update the `self.state` just return a new state
        """
        mask = jnp.any(s[0]) if (s := batch[3:4]) else True

        state, metrics = lax.cond(mask,
                                  lambda s: self._p_train_step(s, batch, rngs),
                                  lambda s: (s, dict(loss=self.loss_fn(batch[1], np.NAN).sum(),
                                                     **self.metrics_fn(batch[1], np.NAN)))
                                  , state)

        #
        # new_state, metrics = self._p_train_step(state, batch, rngs)
        # mask = jnp.any(s[0]) if (s := batch[2:3]) else True
        # new_state = lax.cond(mask, lambda on: on[0], lambda on: on[1], (new_state, state))
        #
        # @jax.jit
        # def nan_array(x):
        #     return x * np.NAN
        # metrics = lax.cond(mask, lambda x: x, lambda x: jax.tree_util.tree_map(nan_array, x), metrics)

        return state, metrics

    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0, 0, None), axis_name="batch")
    @partial(jax.vmap, in_axes=(None, 0, 1, None, None))
    def p_apply_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict = None, c: int = 0) -> Tuple[Features, ...]:
        """
        Apply model to a batch of data returning
        :param state: model state object
        :param batch: Batch tuple to predict with
        :param rngs: dict of rngs for use in the model
        :return: tuple of [X, Y, Y_hat]
        """
        mask = jnp.any(s[0]) if (s := batch[3:4]) else True
        results = lax.cond(mask,
                           lambda s: self._p_apply_step(s, batch, rngs, c),
                           lambda s: (*batch, batch[1]["pred"] * np.NAN)
                           , state)

        return results

    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0, 0), axis_name="batch")
    @partial(jax.vmap, in_axes=(None, 0, 1, None))
    def p_test_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        """
        Perform a prediction step and calculate metrics for a given batch
        :param state: model state object
        :param batch: Batch tuple to predict with
        :param rngs: dict of rngs for use in the model
        :return: [State, dict metrics]
        """
        mask = jnp.any(s[0]) if (s := batch[3:4]) else True

        state, metrics = lax.cond(mask,
                                  lambda s: self._p_test_step(s, batch, rngs),
                                  lambda s: (s, dict(loss=self.loss_fn(batch[1], np.NAN).sum(),
                                                     **self.metrics_fn(batch[1], np.NAN)))
                                  , state)
        return state, metrics

    def epoc_metrics(self):
        logging.info(f"Vector Steps: {self.state.step}")
        return super(VectorTrainer, self).epoc_metrics()

    def step(self, batch: Batch) -> Tuple[Features, Num[Array, ""], Result]:
        params = self.state
        # x, y = batch[:2]
        rngs = replicate(self._make_rngs())
        # y_pred = self.state.apply_fn({'params': }, batch[0], rngs=rngs)
        # p_loss = self.loss_fn(y, y_pred)
        # met = self.metrics_fn(y, y_pred)
        # params, batch: Batch, rngs: Rand_Dict = None, global_batch: int = 1
        v = jax.vmap(super(VectorTrainer, self)._p_train_step, in_axes=(0, 1, None), axis_name="plant")
        p = jax.pmap(v, in_axes=(None, None, 0), axis_name="batch")
        res = p(params, batch, rngs)
        return res

def stack_vec_datasets(ds:list[tfd.Dataset], vec_axes:int=0, add_mask:bool=False):
    lens = [d.cardinality() for d in ds]
    max_len = max(lens)
    padded_ds = []
    for d in ds:
        zero = jax.tree_util.tree_map(lambda x: np.zeros(shape=x.shape, dtype=x.dtype.as_numpy_dtype), d.element_spec)
        pad = tfd.Dataset.from_tensors(zero)
        pad = pad.map(lambda *x: (*x, [False])) if add_mask else pad
        pad = pad.cache()

        l = d.cardinality()
        d = d.map(lambda *x: (*x, [True])) if add_mask else d
        len_diff = max_len - l
        if len_diff > 0:
            d = d.concatenate(pad.repeat(len_diff))
        padded_ds.append(d)

    def stack_els(*ls):
        tree = jax.tree_util.tree_structure(ls[0])
        flat = [jax.tree_util.tree_flatten(d)[0] for d in ls]
        flat_n = [[n[l] for n in flat] for l in range(len(flat[0]))]
        stacked = [tf.concat(x, axis=vec_axes) for x in flat_n]
        return jax.tree_util.tree_unflatten(tree, stacked)

    zipped = tfd.Dataset.zip((*padded_ds,))
    stacked = zipped.map(stack_els, num_parallel_calls=tfd.AUTOTUNE, deterministic=False)
    return stacked

def stack_datasets(ds:list[tfd.Dataset], pad_to_batch:int=None, add_mask:bool=True):
    """
    Pack a list of datasets into a single dataset with
    :param ds:
    :return:
    """
    d = ds[0]
    zero = jax.tree_util.tree_map(lambda x: np.zeros(shape=x.shape, dtype=x.dtype.as_numpy_dtype), d.element_spec)
    pad = tfd.Dataset.from_tensors(zero)
    pad = pad.map(lambda *x: (*x, [False])) if add_mask else pad
    pad = pad.cache()

    lens = [tc if (tc := d._hack_cardinality) is not None else d.cardinality() for d in ds]
    assert min(lens) > 0, "Cannot stack batches where cardinality is unknown. Use the hack `.card` if know after ops"
    max_len = max(lens)
    if pad_to_batch:
        r = max_len % pad_to_batch
        max_len = max_len + (pad_to_batch - r)

    padded_ds = []
    for d in ds:
        l = d.cardinality()
        d = d.map(lambda *x: (*x, [True]), num_parallel_calls=tfd.AUTOTUNE, deterministic=False) if add_mask else d
        len_diff = max_len - l
        if len_diff > 0:
            d = d.concatenate(pad.repeat(len_diff))
        padded_ds.append(d)

    def stack_els(*ls):
        tree = jax.tree_util.tree_structure(ls[0])
        flat = [jax.tree_util.tree_flatten(d)[0] for d in ls]
        flat_n = [[n[l] for n in flat] for l in range(len(flat[0]))]
        stacked = [tf.stack(x) for x in flat_n]
        return jax.tree_util.tree_unflatten(tree, stacked)

    zipped = tfd.Dataset.zip((*padded_ds,))
    stacked = zipped.map(stack_els, num_parallel_calls=tfd.AUTOTUNE, deterministic=False)
    return stacked
