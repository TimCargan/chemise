from __future__ import annotations

import abc
import dataclasses
from typing import Callable

import jax
import numpy as np


@dataclasses.dataclass
class Spec:
    shape: tuple[..., int]
    dtype: np.dtype


class Data(abc.ABC):
    """Mirror the interface for a tf.data.Dataset object but in pure python. As we move away from the TF dependency."""
    @abc.abstractmethod
    def cardinality(self) -> int:
        """Return the cardinality of the data set, if unknown return -1."""

    @property
    @abc.abstractmethod
    def element_spec(self) -> dict:
        """Returns the shape of the data."""

    @abc.abstractmethod
    def as_numpy_iterator(self) -> iter:
        """Return an iterator that yields numpy data"""

    @abc.abstractmethod
    def take(self, n: int) -> Data:
        """Take the first n elements"""

    @abc.abstractmethod
    def map(self, f: Callable, *args, **kwargs) -> Data:
        """Map the function f over elements of the dataset"""


class ListData(Data):
    def __init__(self, data: list, add_batch_dim: bool = False):
        self.data = data
        self.add_batch_dim = add_batch_dim

    def cardinality(self) -> int:
        return len(self.data)

    @property
    def element_spec(self):
        def make_spec(el):
            """"""
            shape = (None, *el.shape[:1]) if self.add_batch_dim else(*el.shape[:1], )
            shape = (*shape, *el.shape[1:])
            return Spec(shape=shape, dtype=el.dtype)
        return jax.tree_util.tree_map(make_spec, self.data[0])

    def take(self, n: int) -> ListData:
        return ListData(self.data[:n], self.add_batch_dim)

    def as_numpy_iterator(self):
        for x in self.data:
            yield x

    def map(self, f, *args, **kwargs):
        print("Warning: you called map")
        return ListData(list(map(f, self.data)), self.add_batch_dim)
