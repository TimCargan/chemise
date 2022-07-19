from dataclasses import dataclass
from typing import Callable

import numpy as np
import optax
from flax.training.train_state import TrainState
import jax


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
        for e in range(num_epochs):
            np_d = data.as_numpy_iterator()
            track_loss = []
            for i in np_d:
                self.state, loss = self.train_step(self.state, self.loss_fn, i)
                track_loss.append(loss["loss"])
                if self.state.step % 10 == 0:
                    mean_loss = np.mean(track_loss)
                    print(f"{e}: {self.state.step} -  {mean_loss}")

            mean_loss = np.mean(track_loss)
            print(f"{e}:  {mean_loss}")