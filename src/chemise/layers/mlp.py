from typing import Callable, Sequence, Any

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Num, Array

Dtype = Any

class MLP(nn.Module):
    """
    A multilayer perceptron (stacked `Dense` layers)
    """
    depth: int
    width: int
    activation: Callable[[Num[Array, "..."]], Num[Array, "..."]] = nn.relu
    key: str = None
    dropout_rate: float = 0.
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Num[Array, "..."], deterministic=True) -> Num[Array, "..."]:
        if self.key:
            x = x[self.key]

        for d in range(self.depth):
            x = nn.Dense(self.width, dtype=self.dtype)(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
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
    pool_fn: Callable[[Num[Array, "..."]], Num[Array, "..."]] = nn.max_pool
    activation_fn: Callable[[Num[Array, "..."]], Num[Array, "..."]] = nn.relu
    norm_fn: Callable[[Num[Array, "..."]], Num[Array, "..."]] = None  #nn.LayerNorm()
    padding: str = "SAME"
    key: str = None
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Num[Array, "... H W C"]) -> Num[Array, "... nH nW nC"]:
        if self.key:
            x = x[self.key]

        axes = -np.arange(0, len(self.kernel_size)+1) - 1
        assert len(axes) > len(self.kernel_size)

        for d in range(self.depth):
            x = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding=self.padding, dtype=self.dtype)(x)
            x = self.activation_fn(x)
            x = self.pool_fn(x, self.pool_size, strides=self.pool_size)
            x = self.norm_fn(x) if self.norm_fn else x

        return x