from typing import Callable
import flax.linen as nn
from jaxtyping import n


class MLP(nn.Module):
    """
    A multilayer perceptron (stacked `Dense` layers)
    """
    depth: int
    width: int
    activation: Callable[[n], n] = nn.relu

    @nn.compact
    def __call__(self, x: n):
        for d in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation(x)
        return x

