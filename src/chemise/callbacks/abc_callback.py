from abc import ABC
from dataclasses import dataclass

class Callback(ABC):
    """
    Abstract base class used to build new callbacks.
    """
    def on_batch_start(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

@dataclass
class CallbackRunner:
    callbacks: list[Callback]

    def on_batch_start(self, trainer):
        for cb in self.callbacks:
            cb.on_batch_start(trainer)

    def on_batch_end(self, trainer):
        for cb in self.callbacks:
            cb.on_batch_end(trainer)

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
