import chex
import jax
import jax.test_util
import numpy as np
from absl.testing import absltest, parameterized

from chemise.layers.mlp import MLC, MLP

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

"""
Tests for the liner Multi Layer (ML) Models
for the most part we just check that the shapes are correct
"""
class MLPTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        (dict(depth=1, width=128, in_size=16, exp_width=128),
         dict(depth=3, width=64, in_size=16, exp_width=64),
         dict(depth=0, width=128, in_size=16, exp_width=16)
         ),
        dims=(1,2,3)
    )
    def test_returns_correct_output_shape(self, depth, width, in_size, exp_width, dims):
        batch_size = 2
        extra_dim = [1]*dims
        model = MLP(depth=depth, width=width)
        rng = jax.random.PRNGKey(0)

        inputs = jax.random.normal(rng, shape=[batch_size, *extra_dim, in_size])
        output, pram = model.init_with_output(rng, inputs)
        @self.variant
        def run(pram, inputs):
            return model.apply(pram, inputs)

        res = run(pram, inputs)
        self.assertEqual((batch_size, *extra_dim, exp_width), res.shape)
        self.assertFalse(np.all(res == res[0], axis=None)) # Quick sanity check that they aren't all the same

    def test_with_key_input(self):
        batch_size = 2
        model = MLP(depth=1, width=128, key="x")
        rng = jax.random.PRNGKey(0)
        inputs = {"x": jax.random.normal(rng, shape=[batch_size, 16]), "y": jax.random.normal(rng, shape=[batch_size, 6])}
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, 128), output.shape)


class MLCTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        (dict(depth=1, in_size=128, exp_width=64),
         dict(depth=3, in_size=128, exp_width=16),
         dict(depth=1, in_size=16, exp_width=8)
         ),
        dims=(1,2,3)
    )
    def test_correct_output_shape(self, depth, in_size, exp_width, dims):
        batch_size = 2
        model = MLC(depth=depth, features=16, kernel_size=(3,)*dims, pool_size=(2,)*dims)
        rng = jax.random.PRNGKey(0)
        inputs = np.random.RandomState(0).normal(size=[batch_size, *[in_size]*dims, 1])
        output, pram = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, *[exp_width]*dims, 16), output.shape)
        @self.variant
        def run(pram, inputs):
            return model.apply(pram, inputs)

        res = run(pram, inputs)
        self.assertEqual((batch_size, *[exp_width]*dims, 16), res.shape)
        self.assertFalse(np.all(res == res[0], axis=None)) # Quick sanity check that they aren't all the same

    def test_with_key_input(self):
        batch_size = 2
        model = MLC(depth=3, features=16, kernel_size=(3, 3), pool_size=(2, 2), key="x")
        rng = jax.random.PRNGKey(0)
        inputs = {"x": jax.random.normal(rng, shape=[batch_size, 128, 128, 1])}
        output, _ = model.init_with_output(rng, inputs)
        self.assertEqual((batch_size, 16, 16, 16), output.shape)




if __name__ == '__main__':
    absltest.main()
