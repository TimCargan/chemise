from dataclasses import dataclass
from flax.training import checkpoints as cp

from chemise.callbacks.abc_callback import Callback
# from chemise.traning import BasicTrainer


@dataclass
class Checkpointer(Callback):
    ckpt_dir: str
    keep: int = 1
    overwrite: bool = False
    keep_every_n_steps: int = None

    auto_restore: bool = False

    def on_fit_begin(self, trainer):
        if self.auto_restore:
            trainer.state = cp.restore_checkpoint(self.ckpt_dir, trainer.state)

    def on_epoch_end(self, trainer):
        cp.save_checkpoint(target=trainer.state, step=trainer.state.step,
                           ckpt_dir=self.ckpt_dir, overwrite=self.overwrite,
                           keep=self.keep, keep_every_n_steps=self.keep_every_n_steps)
