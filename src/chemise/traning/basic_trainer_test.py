from absl.testing import absltest
from absl.testing import parameterized
import jax.test_util

import basic_trainer

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class BasicTrainerHelpersTests(parameterized.TestCase):

    def test_seconds_pretty(self):
        self.assertEqual(" 12s", basic_trainer.seconds_pretty(12.0001))
        self.assertEqual("1200s", basic_trainer.seconds_pretty(1200))
        self.assertEqual(" 12ms",   basic_trainer.seconds_pretty(0.012001))
        self.assertEqual("120µs",   basic_trainer.seconds_pretty(0.000120))
        self.assertEqual(" 12ns",   basic_trainer.seconds_pretty(12e-9))
        self.assertEqual("0.012ns", basic_trainer.seconds_pretty(12e-12))

    def test_make_metric_string(self):
        import jax.numpy as jnp
        self.assertEqual("-- loss: inf", basic_trainer.make_metric_string({"loss": "inf"}))
        self.assertEqual("-- loss: inf, other: 1.0", basic_trainer.make_metric_string({"loss": "inf", "other": 1}))

        # Standard Python cases
        float_cases = [(1234567.89, "1.235e+06"), (0.1, "0.1"), (0.1234567, "0.1235"), (0.000_0001, "1e-07"), (0.000_000_000_000_1, "1e-13")]
        int_cases = [(123456789, "1.235e+08"), (72, "72.0"), (1, "1.0")]
        array_cases = [(jnp.array(123456789), "1.235e+08")]
        for i, o in float_cases + int_cases + array_cases:
            self.assertEqual(f"-- loss: {o}", basic_trainer.make_metric_string({"loss": i}))

        with self.assertRaises(TypeError):
            basic_trainer.make_metric_string({"loss": jnp.array([72, 72])})
            basic_trainer.make_metric_string("String")

class BasicTrainerTests(parameterized.TestCase):
    """
    Test the train functions
    """
    pass

if __name__ == '__main__':
    absltest.main()
