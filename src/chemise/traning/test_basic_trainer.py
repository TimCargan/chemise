from unittest.mock import MagicMock

import flax.linen as nn
import jax.test_util
import numpy as np
import optax
import tensorflow as tf
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax.training.train_state import TrainState

import chemise.traning.basic_trainer as basic_trainer
from chemise.layers import MLP

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
FLAGS = flags.FLAGS

@jax.tree_util.Partial
def mse(y_true, y_pred):
    return optax.l2_loss(y_pred, y_true["y"]).mean()
@jax.tree_util.Partial
def metrics(y_true, y_pred):
    center = optax.l2_loss(y_pred, y_true["y"]).mean()
    return {"met": center}
def make_runner():
    m = nn.Sequential([MLP(depth=1, width=16, key="x"), nn.Dense(1)])
    rng = jax.random.PRNGKey(0)
    rng, _ = jax.random.split(rng)
    data = {"x": np.zeros((1, 20))}
    params = m.init(rng, data)['params']
    tx = optax.adam(3e-4)
    state = TrainState.create(apply_fn=m.apply, params=params, tx=tx)
    runner = basic_trainer.BasicTrainer(state=state, loss_fn=mse, metrics_fn=metrics, rng_keys=["lstm_cell"])
    return runner

def make_step_mock(name:str=None):
    def side_effect(state, *args, **kwargs):
        return state, {}

    mock = MagicMock(name=name)
    mock.side_effect = side_effect
    return mock
class BasicTrainer_Tests(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set Flags
        """
        FLAGS(["Test"])
        data = tf.data.Dataset.from_tensors(
            ({"x": tf.ones(20)}, {"y": [1]})
        ).repeat(100).batch(10, drop_remainder=True)
        cls.data = data


    def setUp(self):
        runner = make_runner()
        self.runner = runner

    def test_metrics_are_calculated(self):
        self.runner.fit(self.data, self.data, num_epochs=1)
        hist = self.runner.train_hist["train"][-1]
        self.assertContainsSubset(["val_loss", "loss", "met", "val_met"], hist.keys())
    def test_fit_calls_steps(self):
        # Mock out train and test since we don't want to do the work
        self.runner.p_train_step = make_step_mock("train")
        self.runner.p_test_step = make_step_mock("test")

        self.runner.fit(self.data, num_epochs=1)

        self.assertEqual(100 // 10, self.runner.p_train_step.call_count)
        self.assertEqual(0, self.runner.p_test_step.call_count)

        self.runner.fit(self.data, self.data, num_epochs=1)
        self.assertEqual(10, self.runner.p_test_step.call_count)


class BasicTrainer_SanityCheck_Tests(parameterized.TestCase):
    keys = ["a", "b", "c"]

    def setUp(self):
        self.zero_in = {k: np.zeros((2,20)) for k in self.keys}
        self.zero_l = {"l": np.zeros((2,20))}

        self.rnd_in = {k: np.random.random((2,20)) for k in self.keys}
        self.rnd_l = {"l": np.random.random((2,20))}

    def test_non_dict(self):
        # All bad
        ds = (self.zero_in, self.zero_l["l"])
        good, in_dict = basic_trainer._sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": np.array(0), "I_b": np.array(0), "I_c": np.array(0), "O_l": np.array(0)}
        self.assertDictEqual(expect, in_dict)


    def test_all_bad(self):
        # All bad
        ds = (self.zero_in, self.zero_l)
        good, in_dict = basic_trainer._sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": np.array(0), "I_b": np.array(0), "I_c": np.array(0), "O_l": np.array(0)}
        self.assertDictEqual(expect, in_dict)

    def test_all_good(self):
        # All Good
        ds = (self.rnd_in, self.rnd_l)
        good, in_dict = basic_trainer._sanity_check(ds)
        self.assertTrue(good)
        expect = {}
        self.assertDictEqual(expect, in_dict)

    def test_input_good_label_bad(self):
        # Input All Good - Label bad
        ds = (self.rnd_in, self.zero_l)
        good, in_dict = basic_trainer._sanity_check(ds)
        self.assertFalse(good)
        expect = {"O_l": np.array(0)}
        self.assertDictEqual(expect, in_dict)

    def test_one_bad(self):
        # One Bad Good
        self.rnd_in["a"] = np.zeros((2,20,30))
        ds = (self.rnd_in, self.rnd_l)
        good, in_dict = basic_trainer._sanity_check(ds)
        self.assertFalse(good)
        expect = {"I_a": np.array(0)}
        self.assertDictEqual(expect, in_dict)


if __name__ == '__main__':
    absltest.main()
