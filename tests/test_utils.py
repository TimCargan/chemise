import jax.numpy as jnp
import jax.test_util
import numpy as np
import utils
from absl.testing import parameterized

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class BasicTrainerHelpersTests(parameterized.TestCase):
    @parameterized.parameters(
        ("1200s", 1200),
        (" 12s", 12.0001),
        (" 12ms", 0.012001),
        ("120µs", 0.000120),
        (" 12ns", 12e-9),
        ("0.012ns", 12e-12)
    )
    def test_seconds_pretty(self, string, seconds):
        self.assertEqual(string, utils.seconds_pretty(seconds))

    @parameterized.parameters(
        (1234567.89, "1.235e+06"), (0.1, "0.1"), (0.1234567, "0.1235"), (0.000_0001, "1e-07"),
        (0.000_000_000_000_1, "1e-13"),
        (123456789, "1.235e+08"), (72, "72.0"), (1, "1.0"),
        (jnp.array(123456789), "1.235e+08")
    )
    def test_metric_string_python_scalars(self, number, string):
        self.assertEqual(f"-- loss: {string}", utils.make_metric_string({"loss": number}))

    def test_make_metric_string(self):
        self.assertEqual("-- loss: inf", utils.make_metric_string({"loss": "inf"}))
        self.assertEqual("-- loss: inf, other: 1.0", utils.make_metric_string({"loss": "inf", "other": 1}))

        with self.assertRaises(TypeError):
            utils.make_metric_string({"loss": jnp.array([72, 72])})
            # noinspection PyTypeChecker
            utils.make_metric_string("String")

    def test_mean_reduce_dicts(self):
        l = [{"loss": v} for v in range(10)]
        red = utils.mean_reduce_dicts(l)
        self.assertIn("loss", red)
        self.assertEqual(np.mean(range(10)), red["loss"])
        self.assertEmpty(utils.mean_reduce_dicts([]))

    def test_list_dict_to_dict_list(self):
        l = [{"loss": v} for v in range(10)]
        red = utils.list_dict_to_dict_list(l)
        self.assertIn("loss", red)
        self.assertListEqual(list(range(10)), red["loss"])
        self.assertEmpty(utils.list_dict_to_dict_list([]))

    @parameterized.parameters(
        (1, (32, 10,), 32),
        (2, (32, 2, 10,), 64),
        (2, (2, 32, 10,), 64),
        (3, (32, 2, 10,), 640),
    )
    def test_get_batch_size(self, nb_dims, shape, size):
        self.assertEqual(size, utils.get_batch_size(jnp.zeros(shape), batch_dims=nb_dims))

    @parameterized.parameters(
        (1, (32, 10,), (32,)),
        (2, (32, 2, 10,), (32, 2,)),
        (2, (2, 32, 10,), (2, 32,)),
        (3, (32, 2, 10,), (32, 2, 10,)),
    )
    def test_get_batch_dims(self, nb_dims, shape, ex_dims):
        dims = utils.get_batch_dims(jnp.zeros(shape), batch_dims=nb_dims)
        self.assertEqual(nb_dims, len(dims))
        self.assertEqual(ex_dims, dims)