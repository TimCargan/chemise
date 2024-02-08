"""Core interfaces for all callbacks."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable

CallbackFn = Callable[[Any], None]


class EarlyStopping(Exception):  # noqa: N818
    """Exception to raise if early stopping is needed."""
    pass


class Callback(ABC):  # noqa: B024
    """Abstract base class used to build new callbacks."""

    def on_fit_start(self, trainer):  # noqa: B027
        """Called once at the start of training, e.g at the start of `BasicTrainer.fit`.

        Args:
          trainer: The trainer object
        """
        pass

    def on_fit_end(self, trainer):  # noqa: B027
        """Called once at the end of training, e.g at the end of `BasicTrainer.fit`.

        Args:
          trainer: The trainer object
        """
        pass

    def on_epoch_start(self, trainer):  # noqa: B027
        """Called at the beginning of every epoch.

        Args:
          trainer: The trainer object
        """
        pass

    def on_epoch_end(self, trainer):  # noqa: B027
        """Called at the end of every epoch.

        Args:
          trainer: The trainer object

        Raises:
          raises: EarlyStopping if early stopping is needed
        """
        pass

    def on_train_start(self, trainer):  # noqa: B027
        """Called at the start of training each epoch.

        Args:
          trainer: The trainer object
        """
        pass

    def on_train_end(self, trainer):  # noqa: B027
        """Called at the end of training each epoch.

        Args:
          trainer: The trainer object
        """
        pass

    def on_train_batch_start(self, trainer):  # noqa: B027
        """Called at the beginning each training batch.

        These should be lightweight since they get called every train step

        Args:
          trainer: The trainer object
        """
        pass

    def on_train_batch_end(self, trainer):  # noqa: B027
        """Called at the end each training batch.

        These should be lightweight since they get called every train step

        Args:
          trainer: The trainer object
        """
        pass

    def on_test_start(self, trainer):  # noqa: B027
        """Called at the start of testing each epoch.

        Args:
          trainer: The trainer object
        """
        pass

    def on_test_end(self, trainer):  # noqa: B027
        """Called at the end of testing each epoch.

        Args:
          trainer: The trainer object
        """
        pass

    def on_test_batch_start(self, trainer):  # noqa: B027
        """Called at the beginning each test batch.

         These should be lightweight since they get called every train step

        Args:
          trainer: The trainer object
        """
        pass

    def on_test_batch_end(self, trainer):  # noqa: B027
        """Called at the end each test batch.

         These should be lightweight since they get called every train step

        Args:
          trainer: The trainer object
        """
        pass


class StepCallback:
    """Wrapper for callbacks used in step runners."""
    def start_cb(self, trainer):
        """Called at the start step (train, test).

        Args:
          trainer:  The trainer object
        """
        pass

    def step_start_cb(self, trainer):
        """Called at the start of a batch.

        Args:
          trainer:  The trainer object
        """
        pass

    def step_end_cb(self, trainer):
        """Called at the end of a batch.

        Args:
          trainer: The trainer object
        """
        pass

    def end_cb(self, trainer):
        """Called at the end of a step (train, test).

        Args:
          trainer:  The trainer object
        """
        pass


@dataclass
class CallbackRunner:
    """The callback runner, taking a list of callbacks and calls each one.

    Attributes:
        callbacks: The list of callbakcs
    """
    callbacks: list[Callback]

    def on_fit_start(self, trainer):
        """Call all `on_fit_start` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_fit_start(trainer)

    def on_fit_end(self, trainer):
        """Call all `on_fit_end` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_fit_end(trainer)

    def on_epoch_start(self, trainer):
        """Call all `on_epoch_start` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_epoch_start(trainer)

    def on_epoch_end(self, trainer):
        """Call all `on_epoch_end` callbacks.

         On epoch end supports the early stopping using the python exception framework and EarlyStopping exception.

        Args:
          trainer: The trainer object
        """
        early_stopping = None
        for cb in self.callbacks:
            try:
                cb.on_epoch_end(trainer)
            except EarlyStopping:
                early_stopping = True
                # TODO: Add some logic to pass messages up the stack for better logging / debugging
        if early_stopping:
            raise EarlyStopping

    def on_train_start(self, trainer):
        """Call all `on_train_start` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer):
        """Call all `on_train_end` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_train_batch_start(self, trainer):
        """Call all `on_train_batch_start` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_train_batch_start(trainer)

    def on_train_batch_end(self, trainer):
        """Call all `on_train_batch_end` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_train_batch_end(trainer)

    def on_test_start(self, trainer):
        """Call all `on_test_start` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_test_start(trainer)

    def on_test_end(self, trainer):
        """Call all `on_test_end` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_test_end(trainer)

    def on_test_batch_start(self, trainer):
        """Call all `on_test_batch_start` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_test_batch_start(trainer)

    def on_test_batch_end(self, trainer):
        """Call all `on_test_batch_end` callbacks.

        Args:
          trainer: The trainer object
        """
        for cb in self.callbacks:
            cb.on_test_batch_end(trainer)

    def train_step_callbacks(self) -> StepCallback:
        """Util functions to package all the train_step callbacks into one object.

        Returns:
            Step callbacks for training step
        """
        cbs = StepCallback()
        cbs.start_cb = self.on_train_start
        cbs.end_cb = self.on_train_end
        cbs.step_start_cb = self.on_train_batch_start
        cbs.step_end_cb = self.on_train_batch_end
        return cbs

    def test_step_callbacks(self) -> StepCallback:
        """Util functions to package all the test_step callbacks into one object.

        Returns:
            Step callbacks for test step
        """
        cbs = StepCallback()
        cbs.start_cb = self.on_test_start
        cbs.end_cb = self.on_test_end
        cbs.step_start_cb = self.on_test_batch_start
        cbs.step_end_cb = self.on_test_batch_end
        return cbs
