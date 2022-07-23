from abc import ABC
from dataclasses import dataclass
from typing import Callable, Any

CallbackFn = Callable[[Any], None]

class Callback(ABC):
    """
    Abstract base class used to build new callbacks.
    """
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


@dataclass
class CallbackRunner:
    callbacks: list[Callback]

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
        for cb in self.callbacks:
            cb.on_epoch_end(trainer)

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
