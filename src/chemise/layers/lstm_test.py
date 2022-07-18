from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.test_util
import numpy as np

import lstm

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LstmTest(parameterized.TestCase):

    def test_lstm_returns_correct_output_shape(self):
        """Tests if the simple LSTM returns the correct shape."""
        batch_size = 2
        seq_len = 3
        embedding_size = 4
        hidden_size = 5
        model = lstm.SimpleLSTM()
        rng = jax.random.PRNGKey(0)
        inputs = np.random.RandomState(0).normal(size=[batch_size, seq_len, embedding_size])
        initial_state = lstm.SimpleLSTM.initialize_carry((batch_size,), hidden_size)
        (_, output), _ = model.init_with_output(rng, initial_state, inputs)
        self.assertEqual((batch_size, seq_len, hidden_size), output.shape)


if __name__ == '__main__':
    absltest.main()
