from dataclasses import dataclass
from typing import Callable
import numpy as np
from flax.training.train_state import TrainState
import jax
from rich.console import Console
from rich.progress import Progress, TextColumn, ProgressColumn
from rich.text import Text

def seconds_pretty(seconds):
    if seconds > 1:
        s = round(seconds, 3)
        return f"{s}s"

    mill = seconds * 1e3
    if mill > 1:
        s = round(mill, 3)
        return f"{s}ms"

    nano = mill * 1e3
    s = round(nano, 3)
    return f"{s}ns"


class StepTime(ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")

        speed = 1 / speed # Convert from steps per second to step time
        speed = seconds_pretty(speed)
        return Text(f"{speed}/step", style="progress.data.speed")

@dataclass
class BasicTrainer:
    """
    Implement boilerplate helper methods to fit basic models similar to what we get in keras
    We want to have a few basic methods
     - Fit
     - Predict
     - Transform - same as predict but just add the predictions the input data
    """

    state: TrainState
    loss_fn: Callable

    @staticmethod
    @jax.jit
    def train_step(state, loss_fn, batch):
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
            loss = loss_fn(batch[1], y_pred)
            return loss

        grad_fn = jax.value_and_grad(step, has_aux=False)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"loss": loss}
        return state, metrics

    def fit(self, data, num_epochs=1):
        cardinality = int(data.cardinality())
        prog_steps = cardinality if cardinality > 0 else None
        con = Console(color_system="windows")
        for e in range(num_epochs):
            np_d = data.as_numpy_iterator()
            with Progress(*Progress.get_default_columns(), StepTime(), TextColumn("-- Loss: {task.fields[loss]}"), auto_refresh=False, console=con) as progress:

                track_loss = []
                task = progress.add_task(f"[green]Epoch {e}: ", total=prog_steps, loss=69)
                for i in np_d:
                    self.state, loss = self.train_step(self.state, self.loss_fn, i)
                    track_loss.append(loss["loss"])
                    progress.update(task, advance=1, loss=loss["loss"])
                    progress.refresh()

            # Update after first epoch sine they should all be the same size
            if prog_steps is None:
                prog_steps = len(track_loss)

            mean_loss = np.mean(track_loss)
            print(f"{e}:  {mean_loss}")