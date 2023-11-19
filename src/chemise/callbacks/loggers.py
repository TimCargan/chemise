from dataclasses import dataclass

import jax.numpy as jnp
import mlflow
from absl import flags, logging

from chemise.callbacks.abc_callback import Callback
# from chemise.traning import BasicTrainer
from chemise.utils import list_dict_to_dict_list, mean_reduce_dicts

FLAGS = flags.FLAGS

@dataclass
class Mlflow(Callback):
    # Performance
    update_metric_freq: int = 10
    _step_count: int = 0

    log_opt_hyperparams: bool = False
    # def on_fit_start(self, trainer):
    #     #TOOD: add some checks to make sure there is exper running its here
    #     pass

    def set_step_number(self, step: int):
        self._step_count = step

    def on_train_batch_end(self, trainer):
        if self._step_count % self.update_metric_freq == 0:
            met = trainer.train_hist["epochs"][-1]["train"][-self.update_metric_freq:]
            # Swap dict of list and then reduce mean
            met = list_dict_to_dict_list(met)
            met = {k: list(jnp.nanmean(jnp.stack(v, axis=0), axis=0)) for k, v in met.items()}
            # add a counter to the metric for vector runs
            met = {f"{k}_{i}" if i > 0 else k: float(v) for k, lv in met.items() for i, v in enumerate(lv)}

            if self.log_opt_hyperparams:
                opt_hyperparams = trainer.state.opt_state.hyperparams
                opt_hyperparams = {k: float(jnp.nanmean(v)) for k, v in opt_hyperparams.items()}
                met = met | opt_hyperparams
            try:
                mlflow.log_metrics(met, step=self._step_count)
            except Exception as e:
                logging.warning(e)
        self._step_count += 1

    def on_test_end(self, trainer):
        met = trainer.train_hist["epochs"][-1]["test"]
        met = mean_reduce_dicts(met)
        met = {f"val_{k}": float(v) for k, v in met.items()}
        mlflow.log_metrics(met, step=self._step_count)
