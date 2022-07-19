from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional
import numpy as np
from flax.training.train_state import TrainState
import jax
from rich.console import Console
from rich.progress import Progress, TextColumn, ProgressColumn, BarColumn, Column, TimeElapsedColumn, TimeRemainingColumn
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
        return f"{seconds:.0f}s"

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


def make_progress(console, prog_steps):
    return Progress(TextColumn("[progress.description]{task.description}"),
             BarColumn(),
             TimeRemainingColumn(compact=True, elapsed_when_finished=True) if prog_steps else TimeElapsedColumn(),
             StepTime(),
             TextColumn("-- Loss: {task.fields[loss]}"),
             auto_refresh=False, console=console)


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
        cardinality = int(data.cardinality())
        prog_steps = cardinality if cardinality > 0 else None
        con = Console(color_system="windows")

        for e in range(num_epochs):
            np_d = data.as_numpy_iterator()
            with make_progress(con, prog_steps) as progress:
                track_loss = []
                task = progress.add_task(f"[green]Epoch {e}/{num_epochs}: ", total=prog_steps, loss="inf")
                for i in np_d:
                    self.state, loss = self.train_step(self.state, i)
                    track_loss.append(loss["loss"])
                    progress.update(task, advance=1, loss=f"{loss['loss']:.3}")
                    progress.refresh()
                progress.update(task, completed=True)

                # Eval model
                if val_data:
                    val_prog = progress.add_task(f"Eval: ", total=None, loss=f"{loss['loss']:.3}")
                    val_loss = []
                    for i in val_data.as_numpy_iterator():
                        loss = self.eval_step(self.state, i)
                        val_loss.append(loss["loss"])
                        progress.update(val_prog, advance=1)
                        progress.refresh()
                    progress.update(val_prog, completed=True, visible=False)

            # Update after first epoch sine they should all be the same size
            if prog_steps is None:
                prog_steps = len(track_loss)

            # End of epoc metrics
            mean_loss = np.mean(track_loss)
            val_loss = "Unknown" if val_data is None else np.mean(val_loss)
            print(f"{e}:  {mean_loss} val {val_loss}")

