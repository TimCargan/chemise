from absl.testing import absltest
from absl.testing import parameterized
import jax.test_util

import basic_trainer

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LstmTest(parameterized.TestCase):

    def test_seconds_pretty(self):
        self.assertEqual("12s", basic_trainer.seconds_pretty(12.0001))
        self.assertEqual("1200s", basic_trainer.seconds_pretty(1200))
        self.assertEqual(" 12ms",   basic_trainer.seconds_pretty(0.012001))
        self.assertEqual("120µs",   basic_trainer.seconds_pretty(0.000120))
        self.assertEqual(" 12ns",   basic_trainer.seconds_pretty(12e-9))
        self.assertEqual("0.012ns", basic_trainer.seconds_pretty(12e-12))

if __name__ == '__main__':
    absltest.main()
