import jax.test_util

import chex
import jax.numpy as jnp
import jax.test_util
import tensorflow as tf
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from chemise.traning.prefetch import Packer

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
FLAGS = flags.FLAGS

class Packer_Tests(parameterized.TestCase):
    keys = ["a", "b", "c"]

    @classmethod
    def setUpClass(cls):
        """
        Set Flags
        """
        FLAGS(["Test"])
        data = tf.data.Dataset.from_tensors(
            ({"x": tf.ones(20), "b": tf.ones(20), "z": tf.ones(20)}, {"y": [1]})
        ).repeat(100).batch(10, drop_remainder=True)
        cls.data = data

    def setUp(self):
        self.packer = Packer(self.data.element_spec)

    def test_all_good(self):
        # All Good
        data = next(self.data.take(1).as_numpy_iterator())
        tdata = self.data.map(self.packer.pack)
        packed = next(tdata.take(1).as_numpy_iterator())
        p_packed = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), packed)
        p_unpack = self.packer.unpack(p_packed)
        unpack = jax.tree_util.tree_map(lambda x: x[0], p_unpack)
        chex.assert_trees_all_close(data, unpack)





if __name__ == '__main__':
    absltest.main()
