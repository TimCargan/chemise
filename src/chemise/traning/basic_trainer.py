from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable, Any, Tuple, List, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging, flags
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from jaxtyping import Num, Array, Bool
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from tensorflow import data as tfd  # Only for typing

from chemise.callbacks.abc_callback import Callback, CallbackRunner, StepCallback
from chemise.traning.prefetch import Prefetch
from chemise.utils import mean_reduce_dicts, make_metric_string, seconds_pretty, get_batch_size

flags.DEFINE_bool("interactive", default=False, help="Run in interactive mode. e.g print graphs", short_name='i')
flags.DEFINE_float("refresh_per_second", default=0.2, help="Frequency in Hz to redraw in interactive mode")
flags.DEFINE_integer("prefetch_buffer", default=3, help="Number of batches to prefetch to the GPU")
flags.DEFINE_boolean("sanity_check", default=True, help="Run sanity check on input data")

FLAGS = flags.FLAGS

Result = dict[str, Num[Array, ""]]
State_Result = Tuple[TrainState, Result]
Features = dict[str, Num[Array, "..."]]
Batch = Tuple[Features, Features] | Tuple[Features, Features, Bool[Array, "..."]]
Rand_Dict = dict[str, jax.random.PRNGKeyArray]
P_Func = Callable[[TrainState, Batch, Rand_Dict], State_Result]


def empty_train_hist():
    return {"epochs": [], "train": []}


def make_default_layout() -> Layout:
    layout = Layout(name="root")
    layout.split(
        Layout(name="graph", ratio=1),
        Layout(name="progbar", size=5),
    )
    return layout


def add_device_batch(data: tfd.Dataset) -> tfd.Dataset:
    platform = jax.default_backend()
    d_count = jax.device_count(platform)
    logging.log_first_n(logging.INFO, "Running on %s with %d devices", 1, platform, d_count)
    logging.debug("Adding device batch of size (%d) to dataset: %s", d_count, data.element_spec)
    data = data.batch(d_count, drop_remainder=True).prefetch(2)
    return data


@jax.jit
def _sanity_error(data: Features) -> (bool, dict):
    """
    Determine if all values for each key in a dict of Num's are the same
    Note: This has the opposite result to `sanity_check` which returns True if the data is different
    :param data:
    :return: flag - True if are key has all the same values, dict of keys with v True if all values are the same
    """
    feats = {f"{k}": jnp.all(v == jnp.reshape(v, (-1))[0], axis=None) for k, v in data.items()}
    is_good = reduce(operator.or_, feats.values(), False)
    return is_good, feats


def _sanity_check(data: Batch):
    """
    Check to see if the input and label data looks correct, i.e. not all the same value
    :param data:
    :return: bool - True if all values different, dict - input keys and a bool set to True if value is all the same
    """
    data = jax.device_put(data)
    r_inputs, inputs = _sanity_error(data[0])
    inputs = {f"I_{k}": np.reshape(data[0][k], (-1,))[..., 0] for k, v in inputs.items() if v}
    r_labels, labels = _sanity_error(data[1])
    labels = {f"O_{k}": np.reshape(data[1][k], (-1,))[..., 0] for k, v in labels.items() if v}
    is_good = np.logical_not(np.logical_or(r_inputs, r_labels))
    return is_good, dict(**inputs, **labels)


def sanity_check(el: Batch):
    logging.debug("Sanity Check run")
    pass_sanity, input_errors = _sanity_check(el)
    if pass_sanity:
        logging.info("Sanity check passed: %s", input_errors)
    else:
        logging.warning("Sanity check failed, The following keys all have the same value: %s", input_errors)


@jax.tree_util.Partial
def no_metrics_fn(y, y_hat):
    return {}


def no_on_dev_shape(tree, *args) -> list:
    return [tree]


@dataclass(unsafe_hash=True)
class BasicTrainer:
    """
    Implement boilerplate helper methods to fit basic models similar to what we get in keras
    This class can manage all the state for the various jax / flax object needed to run basic NN training
    We want to have a few basic methods
     - Fit
     - Predict
     - TODO: Transform - same as predict but just add the predictions the input data
    """
    state: TrainState = field(compare=False)
    loss_fn: Callable[[Num[Array, "..."] | dict[str, Num[Array, "..."]], Num[Array, "..."]], Num[Array, "..."]]
    metrics_fn: Callable[[Num[Array, "..."] | dict[str, Num[Array, "..."]], Num[Array, "..."]], Result] = no_metrics_fn
    callbacks: [Callback] = field(default_factory=list, compare=False)
    train_hist: dict[str, list[Any]] = field(default_factory=empty_train_hist, compare=False)
    train_window: Layout = field(default_factory=make_default_layout, compare=False)

    on_dev_shape: Callable[[Batch, int, bool], list[Batch]] = no_on_dev_shape

    batch_dims: int = 1

    rng_keys: List[str] = field(default_factory=list, compare=False)
    seed: int = 0
    _next_prng: jax.random.PRNGKeyArray = field(default=None, compare=False)

    def __post_init__(self):
        self._next_prng = jax.random.PRNGKey(self.seed)

    def _next_rand(self) -> jax.random.PRNGKeyArray:
        """
        Stateful generate a new PRNG key. The key is generated by splitting `self._next_prng` as such this is
        not pure and so should not be used within a `jit` context.
        :return: a new prng
        """
        rng, self._next_prng = jax.random.split(self._next_prng)
        return rng

    def _make_rngs(self) -> dict[str, jax.random.PRNGKeyArray]:
        """
        Make a dict of rngs to be passed to apply calls
        TODO: Look into using mixin
        :return:
        """
        rng = self._next_rand()
        rngl = jax.random.split(rng, num=len(self.rng_keys))
        rngs = {k: rngl[i] for i, k in enumerate(self.rng_keys)}
        return rngs

    @partial(jax.jit, static_argnums=(0,))
    def _rngs_mix(self, key_dict: Rand_Dict, mixin: int) -> Rand_Dict:
        """
        Faster way to get new rands that can be JIT'ed. Use flax mixin methods wrappers
        :param key_dict: dict of rands to update
        :param mixin: data to mixin
        :return:
        """
        # _fold_in_static LazyRng.create(v, mixin).as_jax_rng()
        return {k: jax.random.fold_in(v, mixin) for k, v in key_dict.items()}

    def _step(self, params, batch: Batch, rngs: Rand_Dict = None, global_batch: int = 1):
        """
            Run a single step and calculate the loss value.
            Here so we only need on trace for train and eval per batch size
            :param params: params of model
            :return: [Loss, predictions]
            """
        x = batch[0]
        y = batch[1]
        mask = s[0] if (s := batch[3:4]) else True

        y_pred = self.state.apply_fn({'params': params}, x, rngs=rngs)
        p_loss = self.loss_fn(y, y_pred)
        p_loss = jnp.where(mask, p_loss, 0.0)  # Apply mask to loss
        loss = p_loss.sum() / global_batch
        return loss, y_pred

    @partial(jax.jit, static_argnums=(0,))
    def _j_step(self, params, batch: Batch, rngs: Rand_Dict = None, global_batch: int = 1):
        """
            Run a single step and calculate the loss value.
            Here so we only need on trace for train and eval per batch size
            :param params: params of model
            :return: [Loss, predictions]
            """
        return self._step(params, batch, rngs, global_batch)

    @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    def p_train_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        """
        Train for a single step. This has a pmap so will use all GPUs
        TODO:
         - support multiple output / multi loss via dicts akin to keras
        Notes:
            In order to keep this a pure function, we don't update the `self.state` just return a new state
        """
        return self._p_train_step(state, batch, rngs)

    def _p_train_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        """
        Unwrapped train step, helpful for inheritance where they want to reorder / not jit or pmap
        :param state:
        :param batch:
        :param rngs:
        :return:
        """
        x = batch[0]
        y = batch[1]

        GLOBAL_BATCH = get_batch_size(batch, 2)
        rngs = self._rngs_mix(rngs, state.step)

        step = jax.value_and_grad(self._j_step, has_aux=True)
        (loss, y_pred), grads = step(state.params, batch, rngs, GLOBAL_BATCH)
        grads = jax.lax.psum(grads, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        metrics = dict(loss=loss, **self.metrics_fn(y, y_pred))
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return state, metrics

    @partial(jax.pmap, static_broadcasted_argnums=(0,), in_axes=(None, 0, 0, 0, None), axis_name="batch")
    def p_apply_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict, c: int = 0) -> Tuple[Features, ...]:
        """
        Apply model to a batch of data returning
        :param state: model state object
        :param batch: Batch tuple to predict with
        :param rngs: dict of rngs for use in the model
        :return: tuple of [X, Y, Y_hat]
        """
        return self._p_apply_step(state, batch, rngs, c)

    def _p_apply_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict, c: int = 0) -> Tuple[Features, ...]:
        rngs = self._rngs_mix(rngs, c)
        _, y_pred = self._j_step(state.params, batch, rngs)
        return (*batch, y_pred)

    @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    def p_test_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        """
        Perform a prediction step and calculate metrics for a given batch
        :param state: model state object
        :param batch: Batch tuple to predict with
        :param rngs: dict of rngs for use in the model
        :return: [State, dict metrics]
        """
        return self._p_test_step(state, batch, rngs)

    def _p_test_step(self, state: TrainState, batch: Batch, rngs: Rand_Dict) -> State_Result:
        x = batch[0]
        y = batch[1]

        gbs = get_batch_size(batch, 2)
        loss, y_pred = self._j_step(state.params, batch, rngs, global_batch=gbs)
        metrics = dict(loss=loss, **self.metrics_fn(y, y_pred))
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return state, metrics

    @partial(jax.jit, static_argnums=(0, 2))
    def _slice(self, r_f_state, s):
        return [x[:s] for x in r_f_state]

    def slice(self, r_state, s):
        leaves, treedef = jax.tree_util.tree_flatten(r_state)
        leaves = self._slice(leaves, s)
        return treedef.unflatten(leaves)

    def _stateful_step_runner(self, data: Prefetch, step_fn: P_Func, hist: list, callback: StepCallback,
                              training: bool = True) -> None:
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
        d_iter = data.iter(batch_dims=self.batch_dims)
        # Replicate state to all devices, use this ref over self.state to reduce / broadcast calls
        r_state = replicate(self.state)
        rngs = replicate(self._make_rngs())
        dev_batch_size = get_batch_size(r_state)

        step = int(np.max(self.state.step))
        while True:
            callback.step_start_cb(self)
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                if not (batch := next(d_iter, None)):
                    break

                # Slice state and RNGs as needed if dev_batch is less than number of devs
                s = get_batch_size(batch)
                _r_state = r_state if s == dev_batch_size else self.slice(r_state, s)
                _rngs = rngs if s == dev_batch_size else self.slice(rngs, s)

                # Run step
                r_state, r_metrics = step_fn(_r_state, batch, _rngs)

                # Un-replicate so callbacks and metrics work
                self.state, metrics = unreplicate((r_state, r_metrics))

                # re-broadcast state if needed
                r_state = r_state if s == dev_batch_size else replicate(self.state)

                # Update metrics
                hist.append(metrics)
                step += 1
                callback.step_end_cb(self)

        callback.end_cb(self)

    @staticmethod
    def get_first_el(data: tfd.Dataset):
        first = next(data.take(1).as_numpy_iterator())
        return first

    """ 
    Standard Public interfaces, here is where the code that a standard user of the API
    They take standard arguments and abstract away all the JAX JIT / PMAP / Replication
    """

    def step(self, batch: Batch) -> Tuple[Features, Num[Array, ""], Result]:
        """
        Run a single step without any jax transformations, helpful for debugging
        :param batch:
        :return: [predictions, loss, metrics]
        """
        x, y = batch[:2]
        rngs = self._make_rngs()
        y_pred = self.state.apply_fn({'params': self.state.params}, batch[0], rngs=rngs)
        p_loss = self.loss_fn(y, y_pred)
        met = self.metrics_fn(y, y_pred)
        return (y_pred, p_loss, met)

    def train_step(self, batch: Batch) -> Result:
        """
        Run a single step
        :param batch: Batch data
        :return:
        """
        r_state = replicate(self.state)
        rngs = replicate(self._make_rngs())
        r_state, metrics = self.p_train_step(r_state, batch, rngs)
        self.state, metrics = unreplicate((r_state, metrics))
        return metrics

    def fit(self, train_data: tfd.Dataset, val_data: tfd.Dataset = None, num_epochs: int = 1):
        """
        Fit model to a given dataset
        :param train_data: data to fit the model to
        :param val_data: validation data to
        :param num_epochs: number of epochs to appy the data
        :return:
        """
        setup_start_time = time.monotonic()
        self.num_epochs = num_epochs

        train_cardinality = int(train_data.cardinality())
        self.train_steps = train_cardinality if train_cardinality > 0 else None

        self.eval_steps = None
        if val_data:
            eval_cardinality = int(val_data.cardinality())
            self.eval_steps = eval_cardinality if eval_cardinality > 0 else None

        # Check to make sure the data isn't all the same value. It's happened, it's a pain
        if FLAGS.sanity_check:
            logging.debug("Sanity Check load data")
            first_el = self.get_first_el(train_data)
            first_el = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), first_el)
            reshaped = self.on_dev_shape(first_el, 0, False)
            sanity_check(reshaped[0])

        con = Console(color_system="windows", force_interactive=FLAGS.interactive, force_terminal=FLAGS.interactive)
        live = Live(self.train_window, console=con, refresh_per_second=FLAGS.refresh_per_second)
        live.start()

        # train_data = add_device_batch(train_data)
        # val_data = add_device_batch(val_data) if val_data else val_data

        callbacks = CallbackRunner(callbacks=self.callbacks)
        callbacks.on_fit_start(self)

        duration = time.monotonic() - setup_start_time
        duration = seconds_pretty(duration)
        logging.info(f"Setup complete took: {duration}")

        train_data_iter = Prefetch(train_data, buffer_size=FLAGS.prefetch_buffer, batch_dims=self.batch_dims,
                                   train=True, on_dev_shape=self.on_dev_shape)
        val_data_iter = Prefetch(val_data, buffer_size=FLAGS.prefetch_buffer, batch_dims=self.batch_dims,
                                 train=False, on_dev_shape=self.on_dev_shape) if val_data else None

        for e in range(self.num_epochs):
            logging.debug("Starting epoch %d", e)
            epoch_start_time = time.monotonic()
            callbacks.on_epoch_start(self)
            self.train_hist["epochs"].append({"train": [], "test": []})

            # Run Train Step
            logging.debug("Starting train step of epoch %d", e)
            self._stateful_step_runner(train_data_iter, self.p_train_step, self.train_hist["epochs"][-1]["train"],
                                       callbacks.train_step_callbacks())

            # Update after first epoch sine they should all be the same size
            if self.train_steps is None:
                self.train_steps = len(self.train_hist["epochs"][-1]["train"])

            # Test model - Only run if there is val_data
            if val_data:
                logging.debug("Starting val step of epoch %d", e)
                self._stateful_step_runner(val_data_iter, self.p_test_step, self.train_hist["epochs"][-1]["test"],
                                           callbacks.test_step_callbacks(), training=False)

                # Update after first epoch sine they should be the same size
                if self.eval_steps is None:
                    self.eval_steps = len(self.train_hist["epochs"][-1]["test"])

            # End of epoc metrics
            logging.debug("Epoch metrics epoch %d", e)

            mets = self.epoc_metrics()
            self.train_hist["train"].append(mets)
            met = make_metric_string(mets)

            duration = time.monotonic() - epoch_start_time
            duration = seconds_pretty(duration)
            logging.info(f"Epoch: {e} - {duration}  {met}")

            # End of epoch callbacks
            callbacks.on_epoch_end(self)

        callbacks.on_fit_end(self)
        live.stop()  # Close the live window since we aren't in a contex
        return

    def epoc_metrics(self):
        # TODO: Maybe move this to a logging callback
        mean_train = mean_reduce_dicts(self.train_hist["epochs"][-1]["train"])
        mean_test = mean_reduce_dicts(self.train_hist["epochs"][-1]["test"])
        mean_test = {f"val_{k}": v for k, v in mean_test.items()}  # Add `val` prefix to test metrics
        mets = dict(**mean_train, **mean_test)
        return mets

    def map_model(self, data: tfd.Dataset) -> Iterator[Tuple[Features, ...]]:
        """
        Map the model over the dataset
        Transforming it to include predictions
        :param data: dataset to map over
        :return: an iterator that yields [X, Y, Y_hat]
        """
        # data = add_device_batch(data)
        d_iter = data
        prefetch = Prefetch(d_iter, buffer_size=FLAGS.prefetch_buffer, train=False, on_dev_shape=self.on_dev_shape)
        d_iter = prefetch.iter()
        r_state = replicate(self.state)
        raw_rngs = self._make_rngs()
        rngs = replicate(raw_rngs)
        dev_batch_size = get_batch_size(r_state)
        c = 0
        while True:
            if not (batch := next(d_iter, None)):
                break

            s = get_batch_size(batch)
            _r_state = r_state if s == dev_batch_size else self.slice(r_state, s)
            _rngs = rngs if s == dev_batch_size else self.slice(rngs, s)

            yield self.p_apply_step(_r_state, batch, _rngs, c)
            c += 1

    def reset(self):
        """
        Clear out the state and train history of the model. This is helpful for when you want to train the same
        architecture multiple times to produce models as we can reuse the compiled functions.
        :return:
        """
        self.state = None
        self.train_hist = empty_train_hist()
        return self
