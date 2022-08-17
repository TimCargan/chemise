import jax
import jax.test_util
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from chemise.layers.mlp import MLP

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




if __name__ == '__main__':
    absltest.main()
