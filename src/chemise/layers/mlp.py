from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import n


class MLP(nn.Module):
    """
    A multilayer perceptron (stacked `Dense` layers)
    """
    depth: int
    width: int
    activation: Callable[[n], n] = nn.relu
    key: str = None

    @nn.compact
    def __call__(self, x: n):
        if self.key:
            x = x[self.key]

        for d in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation(x)
        return x


class MLC(nn.Module):
    """
    MLC - Multi Layer Conv

    A stack of [conv -> pool -> activation]
    """
    depth: int
    features: int
    kernel_size: Sequence[int] = (3, 3)
    pool_size: Sequence[int] = (2, 2)
    pool_fn: Callable[[n], n] = nn.max_pool
    activation_fn: Callable[[n], n] = nn.relu
    key: str = None
    @nn.compact
    def __call__(self, x: n["... H W C"]) -> n:
        if self.key:
            x = x[self.key]

        axes = jnp.arange(-1, len(self.kernel_size) * -1)

        for d in range(self.depth):
            x = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding="SAME")(x)
            x = self.pool_fn(x, self.pool_size, strides=self.pool_size)
            x = self.activation_fn(x)
            x = nn.LayerNorm(feature_axes=axes, reduction_axes=axes)(x)
        return x