# jnjit = chex.fake_pmap_and_jit()
# jnjit.start()
import jax
import numpy as np
import optax  # Optimizers
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # Hide the GPU as we only need tf for data pipeline
from flax import linen as nn  # The Linen API
from flax.training.train_state import TrainState  # Useful dataclass to keep train state
from jaxtyping import Num, Array
from chemise.utils import datasetspec_to_zero
from chemise.callbacks import Checkpointer, Line, Profile, ProgressBar
from chemise.traning import BasicTrainer
from chemise.layers.transformers import TransformerEncoder, PositionalEncoding
from absl import app
from absl import flags

## See https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html

flags.DEFINE_string("output_dir", default="./", help="Location to write checkpoints and output files")
flags.DEFINE_integer("mlp_width", default=128, help="With of MLP")
flags.DEFINE_integer("mlp_depth", default=4, help="With of MLP")

FLAGS = flags.FLAGS


class TransfomerModel(nn.Module):
    model_dim: int  # Hidden dimensionality to use inside the Transformer
    num_classes: int  # Number of classes to predict per sequence element
    num_heads: int  # Number of heads to use in the Multi-Head Attention blocks
    num_layers: int  # Number of encoder blocks to use
    dropout_prob: float = 0.0  # Dropout to apply inside the model
    input_dropout_prob: float = 0.0  # Dropout to apply on the input features

    @nn.compact
    def __call__(self, x, mask=None, add_positional_encoding=True, train=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
            train - If True, dropout is stochastic
        """
        x = x["x"]
        x = nn.Dropout(self.input_dropout_prob)(x, deterministic=not train)
        x = nn.Dense(self.model_dim)(x)

        if add_positional_encoding:
            x = PositionalEncoding(self.model_dim)(x)

        x = TransformerEncoder(
            input_dim=self.model_dim,
            dim_feedforward=self.model_dim * 2,
            dropout_prob=self.dropout_prob,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )(x, mask=mask, train=train)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.model_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_prob)(x, deterministic=not train)
        x = nn.Dense(self.num_classes)(x)
        return x


@jax.tree_util.Partial
def loss(y_true, y_pred):
    loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true["pred"]).mean()
    return loss

@jax.tree_util.Partial
def metrics(y_true, y_pred: Num[Array, "..."]):
    acc = (y_pred.argmax(axis=-1) == y_true["pred"]).mean()
    return {"acc": acc}

def make_data(size: int, cats: int = 10, seq_len: int = 16) -> tf.data.Dataset:
    """
    Make a dataset with (x, rev(x))
    :param size:
    :param cats:
    :param seq_len:
    :return:
    """

    def d(x):
        f = tf.random.categorical(tf.math.log([[1. / cats for _ in range(cats)]]), seq_len) + 1
        f = tf.squeeze(f)

        sx = tf.maximum(tf.constant(16, dtype=tf.int64), x % seq_len)
        f = f[ :sx]

        r = tf.reverse(f, axis=[0])
        zero_pad = tf.zeros((seq_len-sx), tf.int64)

        f = tf.concat([f, zero_pad], axis=-1)
        r = tf.concat([r, zero_pad], axis=-1)

        f = tf.one_hot(f, cats+1)
        return f, r

    def s(x, y):
        """ Make fixed size and add dict """
        nx = tf.zeros((seq_len, cats+1), dtype=tf.float32) + x
        ny = tf.zeros((seq_len,), dtype=tf.int64) + y
        ny = tf.squeeze(ny)
        return {"x": nx}, {"pred": ny}

    data = tf.data.Dataset.range(size).map(d).map(s).shuffle(512)

    return data


def make_model(zeros) -> BasicTrainer:
    m = TransfomerModel(model_dim=64, num_classes=129, num_heads=4, num_layers=2, dropout_prob=0.0)
    rng = jax.random.PRNGKey(0)

    rng, lstm_rng, dropout, init_rng = jax.random.split(rng, num=4)
    rngs = {"params": rng, 'lstm_cell': lstm_rng, "dropout": dropout}
    print(m.tabulate(rngs, zeros[0]))
    params = jax.jit(m.init)(rngs, zeros[0])['params']

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-2,
        warmup_steps=100,
        decay_steps=30 * 20_000 // 32,
        end_value=1e-6
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
        optax.adamw(lr_schedule)
    )

    state = TrainState.create(apply_fn=m.apply, params=params, tx=tx)

    return BasicTrainer(state=state, loss_fn=loss, metrics_fn=metrics, rng_keys=["dropout"])


def main(argv):
    # Make some data
    train = make_data(20_000, cats=128, seq_len=32).batch(32, drop_remainder=True)
    test = make_data(500, cats=128, seq_len=32).batch(32, drop_remainder=True)

    # Make the model and init the weights with 0s
    zeros = datasetspec_to_zero(train.element_spec, batch_size=16)
    m = make_model(zeros)

    # Set up the callbacks
    ckpter = Checkpointer(FLAGS.output_dir, overwrite=True, auto_restore=False)
    graph = Line(title="Loss")
    prog_bar = ProgressBar(update_metric_freq=3)
    prof = Profile(profile_dir=FLAGS.output_dir, steps=(200, 205))
    m.callbacks = [ckpter, graph, prog_bar]



    m.fit(train_data=train, val_data=test, num_epochs=30)

    dd = list(test.take(1).as_numpy_iterator())[0]
    res = m(dd[0])
    res_am = np.argmax(res, axis=-1)
    print(f"in: {np.argmax(dd[0]['x'][0], axis=-1)} , target: {dd[1]['pred'][0]}, res: {np.argmax(res, axis=-1)[0]}")
    return m


if __name__ == "__main__":
    # no_jit.start()
    app.run(main)
