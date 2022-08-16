from __future__ import annotations

import operator
import queue
import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Callable, Any, Tuple
from functools import partial, reduce
from absl import logging


import jax
import numpy as np
from jaxtyping import n
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from tensorflow import data as tfd  # Only for typing
from rich.console import Console
from rich.layout import Layout
from rich.live import Live

from chemise.callbacks.abc_callback import Callback, CallbackRunner, CallbackFn
from chemise.utils import mean_reduce_dicts, make_metric_string, seconds_pretty


def empty_train_hist():
    return {"epochs": [], "train": []}


def make_default_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="graph", ratio=1),
        Layout(name="progbar", size=5),
    )
    return layout


def sanity_check(data: tuple[dict[str, n], dict[str, n]]):
    """
    Check to see if the input and label data looks correct, i.e not all the same value
    :param data:
    :return: bool - True if all values different, dict - input keys and a bool set to True if value is all the same
    """
    inputs = {f"I_{k}": np.all(v == v[0], axis=None) for k, v in data[0].items()}
    r_inputs = reduce(operator.or_, inputs.values(), False)
    labels = {f"O_{k}": np.all(v == v[0], axis=None) for k, v in data[1].items()}
    r_labels = reduce(operator.or_, labels.values(), False)
    is_good = np.logical_not(np.logical_or(r_inputs, r_labels))
    return is_good, dict(**inputs, **labels)


class Prefetcher(Thread):
    """
    Wrap an iterator with a thead to load and prefetch onto the GPU in a no-blocking way
    """
    def __init__(self, data: iter,  buffer_size: int = 3):
        super(Prefetcher, self).__init__()
        self.data = data
        self.q = queue.Queue(buffer_size)

    def run(self):
        devices = jax.local_devices()

        def _prefetch(xs):
            return jax.device_put_sharded(list(xs), devices)

        for data in self.data:
            self.q.put(jax.tree_util.tree_map(_prefetch, data))
        self.q.put(None)

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        if data := self.q.get():
            return data
        raise StopIteration


@jax.tree_util.Partial
def no_metrics_fn(y, y_hat):
    return {}


State_Result = Tuple[TrainState, dict]


@dataclass(unsafe_hash=True)
class BasicTrainer:
    """
    Implement boilerplate helper methods to fit basic models similar to what we get in keras
    We want to have a few basic methods
     - Fit
     - Predict
     - TODO: Transform - same as predict but just add the predictions the input data
    """

    state: TrainState = field(compare=False)
    loss_fn: Callable[[Any, Any], dict]
    metrics_fn: Callable[[Any, Any], dict] = no_metrics_fn
    callbacks: [Callback] = field(default_factory=list, compare=False)
    train_hist: dict[str, list[Any]] = field(default_factory=empty_train_hist, compare=False)
    train_window: Layout = field(default_factory=make_default_layout, compare=False)

    # Train Config Settings
    pre_fetch: int = 2

    @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    def train_step(self, state: TrainState, batch) -> State_Result:
        """
        Train for a single step.
        TODO:
         - support arbitrary loss functions
         - support multiple output / multi loss via dicts akin to keras
         -
        Notes:
            In order to keep this a pure function, we don't update the `self.state` just return a new state
        """
        x = batch[0]
        y = batch[1]

        @partial(jax.value_and_grad, has_aux=True)
        def step(params):
            y_pred = state.apply_fn({'params': params}, x)
            loss = self.loss_fn(y, y_pred)
            return loss, y_pred

        (loss, y_pred), grads = step(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        metrics = dict(loss=loss, **self.metrics_fn(y, y_pred))
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return state, metrics

    @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    def pred_step(self, state: TrainState, batch):
        y_pred = state.apply_fn({'params': state.params}, batch[0])
        return y_pred

    @partial(jax.pmap, static_broadcasted_argnums=(0,), axis_name="batch")
    def test_step(self, state: TrainState, batch):
        y_pred = state.apply_fn({'params': state.params}, batch[0])
        loss = self.loss_fn(batch[1], y_pred)
        return {"loss": loss}

    def _stateful_step_runner(self, data: tfd.Dataset, step_fn: Callable[[TrainState, Any], State_Result], d_count: int,
                              hist: list,
                              start_cb: CallbackFn, step_start_cb: CallbackFn,
                              end_cb: CallbackFn, step_end_cb: CallbackFn) -> None:
        """
        A standard step call, helpful to reduce code in the main train loops
        :param data: data to iterate over
        :param step_fn: the step function to call, must be
        :param hist:
        :param start_cb:
        :param step_start_cb:
        :param end_cb:
        :param step_end_cb:
        :return:
        """
        start_cb(self)
        d_iter = data.as_numpy_iterator()
        d_iter = iter(Prefetcher(d_iter, buffer_size=self.pre_fetch))
        # Replicate state to all devices, use this ref over self.state to reduce / broadcast calls
        r_state = replicate(self.state)
        step = int(self.state.step)
        while True:
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                if not (batch := next(d_iter, None)):
                    break
                step_start_cb(self)
                r_state, metrics = step_fn(r_state, batch)
                self.state, metrics = unreplicate((r_state, metrics))   # un-replicate so callbacks and metrics work
                hist.append(metrics)
                step += int(self.state.step)  # eval step keep in sync with GPU
                step_end_cb(self)
        end_cb(self)

    def fit(self, data: tfd.Dataset, val_data: tfd.Dataset = None, num_epochs: int = 1, interactive: bool = True):
        self.num_epochs = num_epochs

        train_cardinality = int(data.cardinality())
        self.train_steps = train_cardinality if train_cardinality > 0 else None

        self.eval_steps = None
        if val_data:
            eval_cardinality = int(val_data.cardinality())
            self.eval_steps = eval_cardinality if eval_cardinality > 0 else None

        # Check to make sure the data isn't all the same value. It's happened, it's a pain
        first = data.as_numpy_iterator().next()
        pass_sanity, input_errors = sanity_check(first)
        if pass_sanity:
            logging.info("Sanity check passed: %s", input_errors)
        else:
            logging.warning("Sanity check Failed: %s", input_errors)

        platform = jax.default_backend()
        d_count = jax.device_count(platform)
        logging.info("Running on %s with %d devices", platform, d_count)

        logging.debug("Adding device batch of size (%d) to datasets", d_count)
        data = data.batch(d_count, drop_remainder=True).prefetch(2)
        val_data = val_data if not val_data else val_data.batch(d_count, drop_remainder=True).prefetch(2)

        con = Console(color_system="windows", force_interactive=interactive, force_terminal=interactive)
        live = Live(self.train_window, console=con)
        live.start()

        callbacks = CallbackRunner(callbacks=self.callbacks)
        callbacks.on_fit_start(self)

        for e in range(self.num_epochs):
            epoch_start_time = time.monotonic()
            callbacks.on_epoch_start(self)
            self.train_hist["epochs"].append({"train": [], "test": []})

            # Run Train Step
            self._stateful_step_runner(data, self.train_step, d_count, self.train_hist["epochs"][-1]["train"],
                                       callbacks.on_train_start, callbacks.on_train_batch_start,
                                       callbacks.on_train_end, callbacks.on_train_batch_end)

            # Update after first epoch sine they should all be the same size
            if self.train_steps is None:
                self.train_steps = len(self.train_hist["epochs"][-1]["train"])

            # Test model - Only run if there is val_data
            if val_data:
                # Wrap test step in lambda, so it returns state and result to work with the stateful step pattern
                state_test_step = lambda state, batch: (state, self.test_step(state, batch))
                self._stateful_step_runner(val_data, state_test_step, d_count, self.train_hist["epochs"][-1]["test"],
                                           callbacks.on_test_start, callbacks.on_test_batch_start,
                                           callbacks.on_test_end, callbacks.on_test_batch_end)

                # Update after first epoch sine they should be the same size
                if self.eval_steps is None:
                    self.eval_steps = len(self.train_hist["epochs"][-1]["test"])

            # End of epoc metrics
            mean_train = mean_reduce_dicts(self.train_hist["epochs"][-1]["train"])
            mean_test = mean_reduce_dicts(self.train_hist["epochs"][-1]["test"])
            mean_test = {f"val_{k}": v for k, v in mean_test.items()}  # Add `val` prefix to test metrics
            mets = dict(**mean_train, **mean_test)
            self.train_hist["train"].append(mets)

            # TODO: Maybe move this to a logging callback
            met = make_metric_string(mets)
            duration = time.monotonic() - epoch_start_time
            duration = seconds_pretty(duration)
            logging.info(f"Epoch:{e} - {duration}  {met}")

            # End of epoch callbacks
            callbacks.on_epoch_end(self)

        callbacks.on_fit_end(self)
        live.stop()  # Close the live window since we aren't in a contex
        return
