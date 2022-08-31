import jax
import jax.test_util
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from chemise.layers.mlp import MLP, MLC

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class MLPTest(parameterized.TestCase):
    @parameterized.parameters(
        (1, 128, 16, 128),
        (3, 128, 16, 128),
        (0, 128, 16, 16)
    )
    def test_mlp_returns_correct_output_shape(self, depth, width, in_size, exp_width):
        batch_size = 2
        model = MLP(depth=depth, width=width)
        rng = jax.random.PRNGKey(0)
        inputs = np.random.RandomState(0).normal(size=[batch_size, in_size])
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, exp_width), output.shape)

class MLCTest(parameterized.TestCase):
    @parameterized.parameters(
        (1, 128, 64),
        (3, 128, 16),
        (1, 16, 8)
    )
    def test_mlc_2d_returns_correct_output_shape(self, depth, in_size, exp_width):
        batch_size = 2
        model = MLC(depth=depth, features=16, kernel_size=(3,3), pool_size=(2,2))
        rng = jax.random.PRNGKey(0)
        inputs = np.random.RandomState(0).normal(size=[batch_size, in_size, in_size, 1])
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, exp_width, exp_width, 16), output.shape)

    @parameterized.parameters(
        (1, 128, 64),
        (3, 128, 16),
        (1, 16, 8)
    )
    def test_mlc_3d_returns_correct_output_shape(self, depth, in_size, exp_width):
        batch_size = 2
        model = MLC(depth=depth, features=16, kernel_size=(3,3,3), pool_size=(2,2,2))
        rng = jax.random.PRNGKey(0)
        inputs = np.random.RandomState(0).normal(size=[batch_size, in_size, in_size, in_size, 1])
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, exp_width, exp_width, exp_width, 16), output.shape)


if __name__ == '__main__':
    absltest.main()
