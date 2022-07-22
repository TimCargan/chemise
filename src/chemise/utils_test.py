import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax.test_util

import utils

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class BasicTrainerHelpersTests(parameterized.TestCase):

    def test_seconds_pretty(self):
        self.assertEqual(" 12s", utils.seconds_pretty(12.0001))
        self.assertEqual("1200s", utils.seconds_pretty(1200))
        self.assertEqual(" 12ms",   utils.seconds_pretty(0.012001))
        self.assertEqual("120µs",   utils.seconds_pretty(0.000120))
        self.assertEqual(" 12ns",   utils.seconds_pretty(12e-9))
        self.assertEqual("0.012ns", utils.seconds_pretty(12e-12))

    def test_make_metric_string(self):
        import jax.numpy as jnp
        self.assertEqual("-- loss: inf", utils.make_metric_string({"loss": "inf"}))
        self.assertEqual("-- loss: inf, other: 1.0", utils.make_metric_string({"loss": "inf", "other": 1}))

        # Standard Python cases
        float_cases = [(1234567.89, "1.235e+06"), (0.1, "0.1"), (0.1234567, "0.1235"), (0.000_0001, "1e-07"), (0.000_000_000_000_1, "1e-13")]
        int_cases = [(123456789, "1.235e+08"), (72, "72.0"), (1, "1.0")]
        array_cases = [(jnp.array(123456789), "1.235e+08")]
        for i, o in float_cases + int_cases + array_cases:
            self.assertEqual(f"-- loss: {o}", utils.make_metric_string({"loss": i}))

        with self.assertRaises(TypeError):
            utils.make_metric_string({"loss": jnp.array([72, 72])})
            utils.make_metric_string("String")

    def test_mean_reduce_dicts(self):
        l = [{"loss": v} for v in range(10)]
        red = utils.mean_reduce_dicts(l)
        self.assertIn("loss", red)
        self.assertEqual(np.mean(range(10)), red["loss"])
        self.assertEmpty(utils.mean_reduce_dicts([]))


    def list_dict_to_dict_list(self):
        l = [{"loss": v} for v in range(10)]
        red = utils.list_dict_to_dict_list(l)
        self.assertIn("loss", red)
        self.assertListEqual(list(range(10)), red["loss"])
        self.assertEmpty(utils.list_dict_to_dict_list([]))
