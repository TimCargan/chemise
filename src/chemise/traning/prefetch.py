import collections
import itertools
import queue
from functools import partial
from threading import Thread

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import flags
from absl import logging
from einops import rearrange
from jax.tree_util import tree_map

flags.DEFINE_boolean("inc_local", default=True, help="Include Local Vector")
flags.DEFINE_boolean("inc_globcv", default=True, help="Include Local Vector")

FLAGS = flags.FLAGS

FOLDS = [[1190, 1395, 1005, 17314],
         [458, 471, 918, 212],
         [1467, 56424, 862, 1161],
         [384, 643, 1007, 56963],
         [534, 55827, 235, 440]]

# @partial(jax.pmap, in_axes=(0, None), axis_name="batch")

@jax.pmap
@jax.jit
def make_vec_list(data):
    batches = []
    vecs = []
    zeros = []
    for el in data:
        batch_count = el[-1].shape[0]
        vec = el[-1].shape[2]
        batches.append(batch_count)
        vecs.append(vec)
        zeros.append(tree_map(jax.jit(lambda n: jnp.zeros_like(n[0])), el))

    def slice(n, i):
        return n[i]
    def tree_slice(tree, batch_index):
        i = tree_map(lambda x: batch_index, tree)
        return tree_map(slice, tree, i)

    def stack_els(ls):
        tree = jax.tree_util.tree_structure(ls[0])
        flat = [jax.tree_util.tree_flatten(d)[0] for d in ls]
        flat_n = [[n[l] for n in flat] for l in range(len(flat[0]))]
        stacked = [jnp.concatenate(x, axis=2) for x in flat_n]
        return jax.tree_util.tree_unflatten(tree, stacked)

    max_batches = max(batches)

    padded = [tree_map(lambda x: jnp.pad(x, [(0, (max_batches - x.shape[0] % max_batches) % max_batches), *[(0, 0)] * len(x.shape[1:])]),
                el) for el in data]
    step = stack_els(padded)
    lefs, tree = jax.tree_util.tree_flatten(step)
    sliced = [[l[i] for l in lefs] for i in range(max_batches)]
    output = [tree.unflatten(leaves) for leaves in sliced]
    return output

@jax.pmap
def extract(x, rmask):
    r = jax.random.permutation(jax.random.PRNGKey(0), rmask.shape[0])
    global_shuffled = tree_map(jax.jit(lambda n: n[r]), x)
    global_batched = tree_map(jax.jit(lambda n: rearrange(n, "(b s) ... -> b s ...", s=64)), global_shuffled)
    return global_batched


MODE_CODE = {"pass": 0, "local": 1, "global": 2, "cv": 3, "kn": 4, "global++": 5}
def add_mode(xys, mode):
    xs = xys[0]
    xs["mode"] = jnp.where(xys[0]["plant"] == 0, 0, mode)
    return (xs, *xys[1:])

@jax.pmap
def add_kn(xys):
    def stack_els(ls):
        tree = jax.tree_util.tree_structure(ls[0])
        flat = [jax.tree_util.tree_flatten(d)[0] for d in ls]
        flat_n = [[n[l] for n in flat] for l in range(len(flat[0]))]
        stacked = [jnp.concatenate(x, axis=0) for x in flat_n]
        return jax.tree_util.tree_unflatten(tree, stacked)
    def _add_kn(xs):
        xs = {k: v for k, v in xs.items()}
        xs["irradiance_in"] = xs["irradiance_in_kn"]
        return xs

    kn_extract = (_add_kn(xys[0]), *xys[1:])
    kn_extract = add_mode(kn_extract, MODE_CODE["kn"])
    cv_kn = stack_els([xys, kn_extract])
    return cv_kn

@partial(jax.pmap, in_axes=(0, 0, None, None))
def fold_extract(x, rmask, fold_idx, train):
    # Extract and shuffle data
    r = jax.random.permutation(jax.random.PRNGKey(0), rmask.shape[0])
    global_shuffled = tree_map(jax.jit(lambda n: n[r]), x)
    global_shuffled = add_mode(global_shuffled, MODE_CODE["cv"])

    # Find mask of plants for fold
    plants = x[0]["plant"][:, 0, 0]
    fold_mask = (plants == fold_idx)
    fold_mask = jnp.any(fold_mask, axis=0)
    fold_mask = jax.lax.cond(train, lambda v: jnp.logical_not(v), lambda v: v, fold_mask)
    fold_mask = jnp.reshape(fold_mask, (-1, 1))

    def x_help(fold_mask, shape):
        extra_dims = (np.arange(len(shape) - 2) + 1) * -1
        reshape = jnp.expand_dims(fold_mask, axis=extra_dims)
        return reshape

    global_shuffled = tree_map(lambda n: n * x_help(fold_mask, n.shape), global_shuffled)
    global_batched = tree_map(jax.jit(lambda n: rearrange(n, "(b s) ... -> b s ...", s=64)), global_shuffled)
    return global_batched

@jax.pmap
def extract_tree(xys):
    local = tree_map(lambda l: rearrange(l, "(b s) ... -> b s ...", s=64), xys)
    local = add_mode(local, MODE_CODE["local"])
    glob = tree_map(lambda x: rearrange(x, "b p ... -> (p b) 1 ..."), xys)
    glob = add_mode(glob, MODE_CODE["global"])
    return local, glob


class Prefetch_dev:
    """
    Wrap an iterator with a thead to load and prefetch onto the GPU(s) in a no-blocking way
    """

    def __init__(self, data: tf.data.Dataset, buffer_size: int = 3, batch_dims:int = 1, train: bool=True):
        super(Prefetch_dev, self).__init__()
        self.train = train
        self.data_ds = data
        self.q = queue.Queue(buffer_size)
        self.buffer_size = buffer_size
        platform = jax.default_backend()
        d_count = jax.device_count(platform)
        logging.log_first_n(logging.INFO, "Running on %s with %d devices", 1, platform, d_count)

        devices = jax.local_devices()

        first = self.data_ds.element_spec
        batch_dim_size = get_batch_dims(first, batch_dims=batch_dims)
        logging.debug("Sharded prefetch to %d devices, assumed new batch shape [%d, %s, ...]", len(devices),
                      len(devices), ", ".join(str(i) for i in batch_dim_size))

        # Cacluate node sizes
        shapes = jax.tree_util.tree_map(lambda x: np.array((*x.shape, sum(bytes(str(x.dtype), 'utf-8')))), first)
        leave, tree_struct = jax.tree_util.tree_flatten(shapes)
        sizes = collections.defaultdict(list)
        for i, s in enumerate(leave):
            sizes[(*s,)].append(i)

        un_pack_idx = [i for dim, idx in sizes.items() for i in idx]
        un_pack_lookups = {v: i for i, v in enumerate(un_pack_idx)}

        # pack numpy arrays to minimise number of H2D ops
        def pack_tree(*t):
            flat, _ = jax.tree_util.tree_flatten(t)
            # flat = [tf.cast(f, tf.float32) if f.dtype == tf.int32 or f.dtype == tf.int64 else f for f in flat]
            packed = [tf.stack([flat[i] for i in idx]) for dim, idx in sizes.items()]
            return packed

        self.data = self.data_ds.map(pack_tree, num_parallel_calls=tf.data.AUTOTUNE)

        @jax.pmap
        @jax.jit
        def unpack(stacked):
            unorder = [stacked[i][si] for i, idxs in enumerate(sizes.values()) for si, ti in enumerate(idxs)]
            order = [unorder[un_pack_lookups[i]] for i in range(len(unorder))]
            t = jax.tree_util.tree_unflatten(tree_struct, order)
            return t
        self.unpack = unpack

    def unbatch(self, xys):
        local_vec, global_explode = extract_tree(xys)
        ret = ()
        if FLAGS.inc_local:
            ret = (local_vec,)
        if FLAGS.inc_globcv:
            # Global shape
            zero_mask = global_explode[-1][:, :, 0, 0]
            g_v = extract(global_explode, zero_mask)
            # CV and KN extract
            cvs = []
            for f in FOLDS:
                pf = jnp.reshape(jnp.array(f), (4, 1))
                fold_mask = fold_extract(global_explode, zero_mask, pf, self.train)
                if not self.train:
                    # If not training add KN
                    fold_mask = add_kn(fold_mask)
                cvs.append(fold_mask)

                ret = (g_v, *cvs, *ret)
        return ret

    def iter(self, batch_dims:int = 1):
        queue = collections.deque()
        devices = jax.local_devices()
        dev_count = len(devices)
        data_iter = self.data.as_numpy_iterator()

        def _prefetch(data, devs):
            # Send H2D
            flat_n = [[t[l] for t in data] for l in range(len(data[0]))]
            sent_data = [jax.device_put_sharded(x, devs) for x in flat_n]
            return sent_data

        def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
            for _ in range(n):
                batch = list(itertools.islice(data_iter, dev_count))
                num_shards = len(batch)
                if num_shards < 1:
                    return

                # 1 shard or n shard of all the same size
                mask_match = True
                if num_shards == 1 or (mask_match and (bs := [get_batch_dims(el, batch_dims=batch_dims) for el in batch]).count(bs[0]) == num_shards):
                    queue.append(_prefetch(batch, devices[: num_shards]))
                    return
                else:
                    # End of batch just use one GPU for now, add un-even to queue
                    for el in batch:
                        queue.append(_prefetch([el], devices[: 1]))

        enqueue(self.buffer_size)  # Fill up the buffer, less the first already in.
        while queue:
            el = queue.popleft()
            tree = self.unpack(el)
            ub = self.unbatch(tree)
            batches = make_vec_list(ub)
            logging.log_every_n(logging.DEBUG, "un-batched data", 5)
            for b in batches:
                yield b
            # yield tree
            enqueue(1)


class Prefetch(Thread):
    """
    Wrap an iterator with a thead to load and prefetch onto the GPU(s) in a no-blocking way
    """

    def __init__(self, data: iter, buffer_size: int = 3):
        super(Prefetch, self).__init__()
        self.data = data
        self.q = queue.Queue(buffer_size)

    def run(self):
        for data in self.data:
            self.q.put(data)
        self.q.put(None)

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        if data := self.q.get():
            self.q.task_done()
            return data
        raise StopIteration


def get_batch_dims(ds, batch_dims=1) -> tuple[int]:
    """
    Get the likely batch size of a pytree of data
    :param ds:
    :param batch_dims: Number of leading dims to consider part of the batch, default 1,
    if grater than 1 returns the product of the dims
    :return:
    """
    flat, _ = jax.tree_util.tree_flatten(ds)
    shape = np.shape(flat[0])
    return shape[:batch_dims]


def get_batch_size(ds, batch_dims=1) -> int:
    """
    Get the likely batch size of a pytree of data
    :param ds:
    :param batch_dims: Number of leading dims to consider part of the batch, default 1,
    if grater than 1 returns the product of the dims
    :return:
    """
    bds = get_batch_dims(ds, batch_dims)
    batch_size = np.prod(bds)
    return int(batch_size)
