import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional
import numpy as np
from flax.training.train_state import TrainState
import jax
from numpy import ndarray
from rich.console import Console
from rich.progress import Progress, TextColumn, ProgressColumn, BarColumn, Column, TimeElapsedColumn, \
    TimeRemainingColumn, MofNCompleteColumn
from rich.text import Text


def seconds_pretty(seconds:float) -> str:
    """
    Format the number of seconds to a pretty string in most significant  e.g
    0.012 -> 12ms
    12.32 -> 12s
    :param seconds:
    :return: A string of the number of seconds in order of magnitude form
    """

    if seconds > 1:
        return f"{seconds:3.0f}s"

    second_exp = seconds
    for e in ["ms", "µs", "ns"]:
        second_exp = second_exp * 1e3
        if second_exp > 1:
            return f"{second_exp:3.0f}{e}"

    return f"{second_exp:3.3f}ns"

class StepTime(ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")

        speed = 1 / speed  # Convert from steps per second to seconds per step
        speed = seconds_pretty(speed)
        return Text(f"{speed}/step", style="progress.data.speed")


def make_progress(console: Console):
    return Progress(TextColumn("[progress.description]{task.description}"),
                    MofNCompleteColumn(),
                    BarColumn(),
                    TimeRemainingColumn(compact=True, elapsed_when_finished=True),
                    StepTime(),
                    TextColumn("{task.fields[metrics]}"),
                    auto_refresh=False, console=console, refresh_per_second=1)


def make_metric_string(metrics: dict[str, str | ndarray | float]):
    def value_format(v):
        if isinstance(v, str):
            return v
        return f"{v:.3f}"

    met_string = "{}: {}"
    return f"-- {', '.join([met_string.format(k, value_format(v)) for k, v in metrics.items()])}"


@dataclass(unsafe_hash=True)
class BasicTrainer:
    """
    Implement boilerplate helper methods to fit basic models similar to what we get in keras
    We want to have a few basic methods
     - Fit
     - Predict
     - Transform - same as predict but just add the predictions the input data
    """

    state: TrainState = field(compare=False)
    loss_fn: Callable

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch):
        """
        Train for a single step.
        TODO:
         - support arbitrary loss functions
         - support multiple output / multi loss via dicts akin to keras
         -
        Notes:
            In order to keep this a pure function, we don't update the `self.state` just return a new state
        """
        def step(params):
            y_pred = state.apply_fn({'params': params}, batch[0])
            loss = self.loss_fn(batch[1], y_pred)
            return loss

        grad_fn = jax.value_and_grad(step, has_aux=False)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"loss": loss}
        return state, metrics


    @partial(jax.jit, static_argnums=(0,))
    def pred_step(self, state, batch):
        y_pred = state.apply_fn({'params': state.params}, batch[0])
        return y_pred

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state, batch):
        y_pred = state.apply_fn({'params': state.params}, batch[0])
        loss = self.loss_fn(batch[1], y_pred)
        return {"loss": loss}

    def fit(self, data, val_data=None, num_epochs=1):
        con = Console(color_system="windows", force_interactive=True)

        train_cardinality = int(data.cardinality())
        train_steps = train_cardinality if train_cardinality > 0 else None

        if val_data:
            eval_cardinality = int(val_data.cardinality())
            eval_steps = eval_cardinality if eval_cardinality > 0 else None

        with make_progress(con) as progress:
            epoch_task = progress.add_task(f"Epochs", total=num_epochs, metrics="")
            for e in range(num_epochs):
                train_task = progress.add_task(f"Train", total=train_steps, metrics="")
                track_loss = []
                for i in data.as_numpy_iterator():
                    self.state, loss = self.train_step(self.state, i)
                    track_loss.append(loss["loss"])
                    progress.update(train_task, advance=1, metrics=make_metric_string(loss), refresh=True)

                # Eval model
                if val_data:
                    val_prog = progress.add_task(f"Eval", total=eval_steps, metrics="")
                    val_loss = []
                    for i in val_data.as_numpy_iterator():
                        loss = self.eval_step(self.state, i)
                        val_loss.append(loss["loss"])
                        progress.update(val_prog, advance=1, refresh=True)

                    progress.update(val_prog, completed=True, visible=False)
                progress.update(train_task, visible=False)

                # Update after first epoch sine they should all be the same size
                if train_steps is None:
                    train_steps = len(track_loss)

                if eval_steps is None:
                    eval_steps = len(val_loss)

                # End of epoc metrics
                mean_loss = np.mean(track_loss)
                val_loss = "Unknown" if val_data is None else np.mean(val_loss)
                # con.log(f"{e}: {mean_loss} val {val_loss}")
                met = make_metric_string({"loss": mean_loss, "val_loss": val_loss})
                progress.update(epoch_task, advance=1, metrics=met, refresh=True)
