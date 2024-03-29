from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING

import numpy as np
import plotext as plt
from einops import reduce
from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.panel import Panel

from chemise.callbacks.abc_callback import Callback
from chemise.utils import list_dict_to_dict_list

if TYPE_CHECKING:
    from chemise.traning.basic_trainer import BasicTrainer


decoder = AnsiDecoder()

class PlotexMixin(JupyterMixin):

    def __init__(self, draw_fn, title="", **kwargs):
        self.title = title
        self.kwargs = kwargs
        self.draw_fn = draw_fn
        self.re_draw = True

    def update(self, **kwargs):
        self.kwargs = kwargs
        self.re_draw = True

    def __rich_console__(self, console, options):
        if self.re_draw:
            self.width = options.max_width or console.width
            self.height = options.height or console.height
            canvas = self.draw_fn(self.width, self.height, title=self.title, **self.kwargs)
            self.rich_canvas = Group(*decoder.decode(canvas))
            self.re_draw = False
        yield self.rich_canvas

def make_line_plot(width, height, title="", xs=None, ys=None):
    plt.clf()
    plt.title(title)
    min_v = np.Inf
    max_v = 0
    y = []
    for n, y in ys.items():
        y = reduce(np.array(y), "s ... -> s", reduction="mean")
        limit = max(floor(len(y) * 0.9), 10) # 10 or the last 90% of the elements
        min_v = v if (v := np.min(y[-limit:])) < min_v else min_v
        max_v = v if (v := np.max(y[-limit:])) > max_v else max_v
        plt.plot(y, label=n)
    plt.plotsize(width, height)
    # plt.theme('dark')
    if len(y) > 10:
        plt.ylim(min_v, max_v)
    return plt.build()

@dataclass
class Line(Callback):
    window_pane: str = "graph"

    def __init__(self, title: str):
        self.plotter = PlotexMixin(title=title, ys={"t": [0]}, draw_fn=make_line_plot)

    def on_fit_start(self, basic_trainer: BasicTrainer):
        if pane := basic_trainer.train_window.get(self.window_pane):
            pane.update(Panel(self.plotter))

    def on_epoch_end(self, trainer: BasicTrainer):
        ys = list_dict_to_dict_list(trainer.train_hist["train"])
        self.plotter.update(ys=ys)





