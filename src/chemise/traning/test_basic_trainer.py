import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax.test_util
import chemise.traning.basic_trainer as basic_trainer

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class BasicTrainer_SanityCheck_Tests(parameterized.TestCase):
    keys = ["a", "b", "c"]

    def setUp(self):
        self.zero_in = {k: np.zeros((2,20)) for k in self.keys}
        self.zero_l = {"l": np.zeros((2,20))}

        self.rnd_in = {k: np.random.random((2,20)) for k in self.keys}
        self.rnd_l = {"l": np.random.random((2,20))}

    def test_all_bad(self):
        # All bad
        ds = (self.zero_in, self.zero_l)
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": True, "I_b": True, "I_c": True, "O_l": True}
        self.assertDictEqual(expect, in_dict)

    def test_all_good(self):
        # All Good
        ds = (self.rnd_in, self.rnd_l)
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertTrue(good)
        expect = {"I_a": False, "I_b": False, "I_c": False, "O_l": False}
        self.assertDictEqual(expect, in_dict)

    def test_input_good_label_bad(self):
        # Input All Good - Label bad
        ds = (self.rnd_in, self.zero_l)
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": False, "I_b": False, "I_c": False, "O_l": True}
        self.assertDictEqual(expect, in_dict)

    def test_one_bad(self):
        # One Bad Good
        self.rnd_in["a"] = np.zeros((2,20,30))
        ds = (self.rnd_in, self.rnd_l)
        good, in_dict = basic_trainer.sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": True, "I_b": False, "I_c": False, "O_l": False}
        self.assertDictEqual(expect, in_dict)


if __name__ == '__main__':
    absltest.main()
