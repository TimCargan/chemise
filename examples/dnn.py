import jax
import optax  # Optimizers
import tensorflow as tf
tf.config.set_visible_devices([], "GPU") # Hide the GPU as we only need tf for data pipeline
from flax import linen as nn  # The Linen API
from flax.training.train_state import TrainState  # Useful dataclass to keep train state

from chemise.utils import datasetspec_to_zero



from chemise.callbacks import Checkpointer, Line, Profile, ProgressBar
from chemise.layers import MLP
from chemise.traning import BasicTrainer
from absl import app
from absl import flags

flags.DEFINE_string("output_dir", default="./", help="Location to write checkpoints and output files")
flags.DEFINE_integer("mlp_width", default=128, help="With of MLP")
flags.DEFINE_integer("mlp_depth", default=4, help="With of MLP")

FLAGS = flags.FLAGS


class FlaxDnet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Reshape the inputs out of the dict
        x = x["x"]
        x = MLP(width=FLAGS.mlp_width, depth=FLAGS.mlp_depth)(x)
        pred = nn.Dense(1)(x)
        return pred


@jax.tree_util.Partial
def loss(y_true, y_pred):
    return optax.l2_loss(y_pred, y_true["pred"])




def make_model(zeros) -> BasicTrainer:
    m = FlaxDnet()
    rng = jax.random.PRNGKey(0)
    rng, lstm_rng, init_rng = jax.random.split(rng, num=3)

    rngs = {"params": rng, 'lstm_cell': lstm_rng}
    params = jax.jit(m.init)(rngs, zeros[0])['params']
    tx = optax.adam(3e-4)
    state = TrainState.create(apply_fn=m.apply, params=params, tx=tx)

    return BasicTrainer(state=state, loss_fn=loss, rng_keys=["lstm_cell"])

def main(argv):
        # Make some random data, this is bad as the test and train is all the same
        data = tf.data.Dataset.from_tensors(({"x": tf.ones(20)}, {"pred": [1]}))
        d = data.repeat(100).batch(10)
        t = data.repeat(5).batch(10)

        # Make the model and init the weights with 0s
        zeros = datasetspec_to_zero(d.element_spec)
        m = make_model(zeros)

        # Set up the callbacks
        ckpter = Checkpointer(FLAGS.output_dir, overwrite=True, auto_restore=True)
        graph = Line(title="Loss")
        prog_bar = ProgressBar(update_metric_freq=3)
        prof = Profile(profile_dir=FLAGS.output_dir, steps=(200, 205))
        m.callbacks = [ckpter, graph, prog_bar, prof]
        m.fit(d, num_epochs=1, val_data=t)
        return m


if __name__ == "__main__":
    app.run(main)