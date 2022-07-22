from dataclasses import dataclass
from rich.panel import Panel
from rich.progress import Progress, TextColumn, ProgressColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.text import Text
from chemise.callbacks.abc_callback import Callback
# from chemise.traning import BasicTrainer
from chemise.utils import make_metric_string, seconds_pretty

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


def make_progress() -> Progress:
    return Progress(TextColumn("[progress.description]{task.description}"),
                    MofNCompleteColumn(),
                    BarColumn(),
                    TimeRemainingColumn(compact=True, elapsed_when_finished=True),
                    StepTime(),
                    TextColumn("{task.fields[metrics]}"),
                    auto_refresh=True, refresh_per_second=2, speed_estimate_period=3600 * 24)

@dataclass
class ProgressBar(Callback):
    window_pane: str = "progbar"

    def on_fit_start(self, trainer):
        self.progress = make_progress()
        self.epoch_task = self.progress.add_task(f"Epochs", total=trainer.num_epochs, metrics="")
        self.train_task = self.progress.add_task(f"Train", completed=0, total=None, metrics="", visible=True, start=False)
        self.val_prog = self.progress.add_task(f"Eval", completed=0, total=None, metrics="", visible=False, start=False)
        if pane := trainer.train_window.get(self.window_pane):
            pane.update(Panel(self.progress))

    def on_epoch_start(self, trainer):
        self.progress.reset(self.train_task, total=trainer.train_steps, visible=True, start=False)
        self.progress.reset(self.val_prog, total=trainer.eval_steps, visible=True, start=False)

    def on_epoch_end(self, trainer):
        met = trainer.train_hist["train"][-1]
        self.progress.update(self.epoch_task, advance=1, metrics=make_metric_string(met), refresh=True)

    def on_train_start(self, trainer):
        self.progress.start_task(self.train_task)

    def on_test_start(self, trainer):
        self.progress.start_task(self.val_prog)

    def on_train_batch_end(self, trainer):
        met = trainer.train_hist["epochs"][-1]["train"][-1]
        self.progress.update(self.train_task, advance=1, metrics=make_metric_string(met))

    def on_test_batch_end(self, trainer):
        self.progress.update(self.val_prog, advance=1)
