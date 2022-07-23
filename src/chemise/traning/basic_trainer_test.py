import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax.test_util
from tensorflow import data as tfd
import basic_trainer

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class BasicTrainerTests(parameterized.TestCase):
    """
    Test the train functions
    """
    def test_sanity_check(self):
        keys = ["a", "b", "c"]

        # All bad
        zero_in = {k: np.zeros((2,20)) for k in keys}
        zero_l = {"l": np.zeros((2,20))}
        ds = tfd.Dataset.from_tensors((zero_in, zero_l))
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": True, "I_b": True, "I_c": True, "O_l": True}
        self.assertDictEqual(expect, in_dict)

        # All Good
        rnd_in = {k: np.random.random((2,20)) for k in keys}
        rnd_l = {"l": np.random.random((2,20))}
        ds = tfd.Dataset.from_tensors((rnd_in, rnd_l))
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertTrue(good)
        expect = {"I_a": False, "I_b": False, "I_c": False, "O_l": False}
        self.assertDictEqual(expect, in_dict)

        # Input All Good - Label bad
        rnd_in = {k: np.random.random((2,20)) for k in keys}
        ds = tfd.Dataset.from_tensors((rnd_in, zero_l))
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": False, "I_b": False, "I_c": False, "O_l": True}
        self.assertDictEqual(expect, in_dict)

        # One Bad Good
        rnd_in["a"] = np.zeros((2,20,30))
        ds = tfd.Dataset.from_tensors((rnd_in, rnd_l))
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": True, "I_b": False, "I_c": False, "O_l": False}
        self.assertDictEqual(expect, in_dict)

if __name__ == '__main__':
    absltest.main()
