import jax
import jax.test_util
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from chemise.layers.mlp import MLP, MLC

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

"""
Tests for the liner Multi Layer (ML) Models
for the most part we just check that the shapes are correct
"""
class MLPTest(parameterized.TestCase):
    @parameterized.parameters(
        (1, 128, 16, 128),
        (3, 128, 16, 128),
        (0, 128, 16, 16)
    )
    def test_mlp_2d_returns_correct_output_shape(self, depth, width, in_size, exp_width):
        batch_size = 2
        model = MLP(depth=depth, width=width)
        rng = jax.random.PRNGKey(0)
        inputs = np.random.RandomState(0).normal(size=[batch_size, in_size])
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, exp_width), output.shape)

    @parameterized.parameters(
        (1, 128, 16, 128),
        (3, 128, 16, 128),
        (0, 128, 16, 16)
    )
    def test_mlp_nd_returns_correct_output_shape(self, depth, width, in_size, exp_width):
        batch_size = 2
        model = MLP(depth=depth, width=width)
        rng = jax.random.PRNGKey(0)
        for n in [[], [1], [1,1]]:
            inputs = jax.random.normal(rng, shape=[batch_size, *n, in_size])
            output, _ = model.init_with_output(rng, inputs)
            self.assertEqual((batch_size, *n, exp_width), output.shape)

    def test_with_key_input(self):
        batch_size = 2
        model = MLP(depth=1, width=128, key="x")
        rng = jax.random.PRNGKey(0)
        inputs = {"x": jax.random.normal(rng, shape=[batch_size, 16]), "y": jax.random.normal(rng, shape=[batch_size, 6])}
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, 128), output.shape)


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

    def test_with_key_input(self):
        batch_size = 2
        model = MLC(depth=3, features=16, kernel_size=(3, 3), pool_size=(2, 2), key="x")
        rng = jax.random.PRNGKey(0)
        inputs = {"x": jax.random.normal(rng, shape=[batch_size, 128, 128, 1])}
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, 16, 16, 16), output.shape)


if __name__ == '__main__':
    absltest.main()
