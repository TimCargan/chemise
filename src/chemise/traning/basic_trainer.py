from __future__ import annotations
import operator
import time
from dataclasses import dataclass, field
from typing import Callable, Any, Tuple
from functools import partial, reduce

import numpy as np
from absl import logging
import jax
from flax.jax_utils import prefetch_to_device, replicate, unreplicate
from flax.training.train_state import TrainState
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


def sanity_check(data):
    """
    Check to see if the input and label data looks correct, i.e not all the same value
    :param data:
    :return: bool - True if all values different, dict - input keys and a bool set to True if value is all the same
    """
    for d in data.take(1).as_numpy_iterator():
        inputs = {f"I_{k}": np.all(v == v[0], axis=None) for k, v in d[0].items()}
        r_inputs = reduce(operator.or_, inputs.values(), False)
        labels = {f"O_{k}": np.all(v == v[0], axis=None) for k, v in d[1].items()}
        r_labels = reduce(operator.or_, labels.values(), False)
    is_good = not (r_inputs or r_labels)
    return is_good, dict(**inputs, **labels)

def prefetch(dataset, n_prefetch=1):
    # Taken from: https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                  ds_iter)
    if n_prefetch:
        ds_iter = prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter

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

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
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
        def step(params):
            y_pred = state.apply_fn({'params': params}, x)
            loss = self.loss_fn(y, y_pred)
            return loss, y_pred

        grad_fn = jax.value_and_grad(step, has_aux=True)
        (loss, y_pred), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = dict(loss=loss, **self.metrics_fn(y, y_pred))
        return state, metrics

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def pred_step(self, state: TrainState, batch):
        y_pred = state.apply_fn({'params': state.params}, batch[0])
        return y_pred

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def test_step(self, state: TrainState, batch):
        y_pred = state.apply_fn({'params': state.params}, batch[0])
        loss = self.loss_fn(batch[1], y_pred)
        return {"loss": loss}

    def _stateful_step_runner(self, data, step_fn: Callable[[TrainState, Any], State_Result], hist: list,
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
        reps = c if (c := jax.device_count("GPU")) > 1 else 1
        for batch in prefetch(data.batch(reps, drop_remainder=True)):
            step_start_cb(self)
            r_state = replicate(self.state)
            r_state, metrics = step_fn(r_state, batch)
            self.state = unreplicate(r_state)
            hist.append(metrics)
            step_end_cb(self)
        end_cb(self)

    def fit(self, data, val_data=None, num_epochs=1, interactive=True):
        self.num_epochs = num_epochs

        train_cardinality = int(data.cardinality())
        self.train_steps = train_cardinality if train_cardinality > 0 else None

        self.eval_steps = None
        if val_data:
            eval_cardinality = int(val_data.cardinality())
            self.eval_steps = eval_cardinality if eval_cardinality > 0 else None

        # Check to make sure the data isn't all the same value. It's happened, it's a pain
        pass_sanity, input_errors = sanity_check(data)
        if pass_sanity:
            logging.info("Sanity check passed: %s", input_errors)
        else:
            logging.warning("Sanity check Failed: %s", input_errors)

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
            self._stateful_step_runner(data, self.train_step, self.train_hist["epochs"][-1]["train"],
                                       callbacks.on_train_start, callbacks.on_train_batch_start,
                                       callbacks.on_train_end, callbacks.on_train_batch_end)

            # Update after first epoch sine they should all be the same size
            if self.train_steps is None:
                self.train_steps = len(self.train_hist["epochs"][-1]["train"])

            # Test model - Only run if there is val_data
            if val_data:
                # Wrap test step in lambda, so it returns state and result to work with the stateful step pattern
                state_test_step = lambda state, batch: (state, self.test_step(state, batch))
                self._stateful_step_runner(data, state_test_step, self.train_hist["epochs"][-1]["test"],
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
