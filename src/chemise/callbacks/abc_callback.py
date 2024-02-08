from abc import ABC
from dataclasses import dataclass
from typing import Callable, Any

CallbackFn = Callable[[Any], None]


class EarlyStopping(Exception):
    """Exception to raise if early stopping is needed"""
    pass


class Callback(ABC):
    """
    Abstract base class used to build new callbacks.
    """

    def set_step_number(self, step: int):
        """
        Set the current step number, called once at the beginning of a fit
        :param step:
        :return:
        """
        pass

    def on_fit_start(self, trainer):
        """
        Called once at the start of training, e.g at the start of `BasicTrainer.fit`
        :param trainer:
        :return:
        """
        pass

    def on_fit_end(self, trainer):
        """
        Called once at the end of training, e.g at the start of `BasicTrainer.fit`
        :param trainer:
        :return:
        """
        pass

    def on_epoch_start(self, trainer):
        """
        Called at the beginning of every epoch
        :param trainer:
        :return:
        """
        pass

    def on_epoch_end(self, trainer):
        """
        Called at the end of every epoch
        :param trainer:
        :return:
        :raises: EarlyStopping if early stopping is needed
        """
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_train_batch_start(self, trainer):
        """
        Called at the beginning each training batch
        These should be lightweight since they get called every train step
        :param trainer:
        :return:
        """
        pass

    def on_train_batch_end(self, trainer):
        """
        Called at the end each training batch
        These should be lightweight since they get called every train step
        :param trainer:
        :return:
        """
        pass

    def on_test_start(self, trainer):
        pass

    def on_test_end(self, trainer):
        pass

    def on_test_batch_start(self, trainer):
        """
         Called at the beginning each test batch
         These should be lightweight since they get called every train step
         :param trainer:
         :return:
         """
        pass

    def on_test_batch_end(self, trainer):
        pass


class StepCallback:
    def start_cb(self, trainer):
        pass

    def step_start_cb(self, trainer):
        pass

    def step_end_cb(self, trainer):
        pass

    def end_cb(self, trainer):
        pass


@dataclass
class CallbackRunner:
    callbacks: list[Callback]

    def set_step_number(self, step: int):
        for cb in self.callbacks:
            cb.set_step_number(step)

    def on_fit_start(self, trainer):
        for cb in self.callbacks:
            cb.on_fit_start(trainer)

    def on_fit_end(self, trainer):
        for cb in self.callbacks:
            cb.on_fit_end(trainer)

    def on_epoch_start(self, trainer):
        for cb in self.callbacks:
            cb.on_epoch_start(trainer)

    def on_epoch_end(self, trainer):
        """Run the end of epoch callbacks.
         On epoch end supports the early stopping using the python exception framework and EarlyStopping exception.
         """
        early_stopping = None
        for cb in self.callbacks:
            try:
                cb.on_epoch_end(trainer)
            except EarlyStopping as es:
                early_stopping = True
                # TODO: Add some logic to pass messages up the stack for better logging / debugging
        if early_stopping:
            raise EarlyStopping

    def on_train_start(self, trainer):
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_train_batch_start(self, trainer):
        for cb in self.callbacks:
            cb.on_train_batch_start(trainer)

    def on_train_batch_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_batch_end(trainer)

    def on_test_start(self, trainer):
        for cb in self.callbacks:
            cb.on_test_start(trainer)

    def on_test_end(self, trainer):
        for cb in self.callbacks:
            cb.on_test_end(trainer)

    def on_test_batch_start(self, trainer):
        for cb in self.callbacks:
            cb.on_test_batch_start(trainer)

    def on_test_batch_end(self, trainer):
        for cb in self.callbacks:
            cb.on_test_batch_end(trainer)

    def train_step_callbacks(self) -> StepCallback:
        """
        Util functions to package all the train_step callbacks into one object
        :return:
        """
        cbs = StepCallback()
        cbs.start_cb = self.on_train_start
        cbs.end_cb = self.on_train_end
        cbs.step_start_cb = self.on_train_batch_start
        cbs.step_end_cb = self.on_train_batch_end
        return cbs

    def test_step_callbacks(self) -> StepCallback:
        """
        Util functions to package all the test_step callbacks into one object
        :return:
        """
        cbs = StepCallback()
        cbs.start_cb = self.on_test_start
        cbs.end_cb = self.on_test_end
        cbs.step_start_cb = self.on_test_batch_start
        cbs.step_end_cb = self.on_test_batch_end
        return cbs
