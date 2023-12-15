import dataclasses
import datetime
import time
from unittest.mock import MagicMock, patch

import jax.test_util
from absl.testing import absltest, parameterized

import chemise.callbacks.checkpointer as ckpts

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


@dataclasses.dataclass
class MockState:
    step = 0


@dataclasses.dataclass
class MockTrainer:
    state: MockState = dataclasses.field(default_factory=MockState)


class CheckpointerTest(parameterized.TestCase):

    @patch("flax.training.orbax_utils.save_args_from_target")
    def test_intra_train_time_freq(self, _):
        """Tests if the simple LSTM returns the correct shape."""
        obj = MockTrainer()
        c = ckpts.Checkpointer(ckpt_dir=".", intra_train_freq_time=datetime.timedelta(seconds=2), auto_restore=False)
        c.ckpt_mgr.save = MagicMock(name="Save")

        # Setup state, assume this takes less than 2 seconds
        c.on_fit_start(obj)
        c.on_train_batch_end(obj)
        c.ckpt_mgr.save.assert_not_called()

        # Wait the time delta so it should run a save
        time.sleep(3)
        old_time = c._last_ckpt_time
        c.on_train_batch_end(obj)
        c.ckpt_mgr.save.assert_called()
        self.assertNotEquals(old_time, c._last_ckpt_time)

        # Test that it will now wait again
        c.on_train_batch_end(obj)
        self.assertTrue(c.ckpt_mgr.save.call_count == 1)


if __name__ == '__main__':
    absltest.main()
