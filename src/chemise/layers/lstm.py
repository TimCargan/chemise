import functools
import jax
import jax.numpy as jnp
from flax import linen as nn


class SimpleLSTM(nn.Module):
    """A simple unidirectional LSTM.
    Example Usage:

    @nn.compact
    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        _, forward_outputs = SimpleLSTM()(initial_state, inputs)
    """
    @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        return nn.OptimizedLSTMCell()(carry, x)

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        # Use fixed random key since default state init fn is just zeros.
        return nn.OptimizedLSTMCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, hidden_size)


class AutoregLSTM(nn.Module):
    output_layer: nn.Dense = None

    def setup(self) -> None:
        self.cell = nn.OptimizedLSTMCell()

    def __call__(self, carry_pred, x):
        def _step(self, carry_pred, x):
            carry, past_pred = carry_pred
            in_x = jnp.concatenate([x, past_pred], axis=-1)
            carry, y = self.cell(carry, in_x)
            y = self.output_layer(y)
            return (carry, y), y

        carry_pred, x = nn.transforms.scan(_step,
                            variable_broadcast='params',
                            in_axes=1, out_axes=1,
                            split_rngs={'params': False}
                            )(self, carry_pred, x)
        return carry_pred, x