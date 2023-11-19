import collections
import itertools
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import numpy as np
import tensorflow as tf
from absl import flags, logging

from chemise.utils import get_batch_dims

flags.DEFINE_boolean("prefetch_pack", default=False, help="Pack items when prefetch to reduce number of H2D mem-copy calls")
FLAGS = flags.FLAGS

@dataclass(unsafe_hash=True)
class Packer:
    """
    Pack and unpack data to minimise the number of H2D calls, can be helpfully if there are a lot of small input feats
    """
    def __init__(self, first):
        # Calculate node sizes
        shapes = jax.tree_util.tree_map(lambda x: np.array((*x.shape, sum(bytes(str(x.dtype), 'utf-8')))), first)
        leave, self.tree_struct = jax.tree_util.tree_flatten(shapes)

        sizes = collections.defaultdict(list)
        self.sizes = sizes

        for i, s in enumerate(leave):
            sizes[(*s,)].append(i)

        un_pack_idx = [i for dim, idx in sizes.items() for i in idx]
        self.un_pack_lookups = {v: i for i, v in enumerate(un_pack_idx)}

    def pack(self, *t):
        """
        pack numpy arrays to minimise number of H2D ops
        :param t: tree to pack
        :return: 
        """""
        flat, _ = jax.tree_util.tree_flatten(t)
        packed = [tf.stack([flat[i] for i in idx]) for dim, idx in self.sizes.items()]
        return packed

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def unpack(self, stacked):
        """
        Take a tree packed using the pack function and unpack it
        :param stacked:
        :return:
        """
        unorder = [stacked[i][si] for i, idxs in enumerate(self.sizes.values()) for si, ti in enumerate(idxs)]
        order = [unorder[self.un_pack_lookups[i]] for i in range(len(unorder))]
        t = jax.tree_util.tree_unflatten(self.tree_struct, order)
        return t




class Prefetch:
    """
    Prefetch onto the GPU(s), data is compacted to reduce the number of H2D ops
    """
    def __init__(self, data: tf.data.Dataset, buffer_size: int = 3, batch_dims: int = 1, train: bool = True,
                 on_dev_shape: Callable[[Any, int, bool], list[Any]] = None):
        super(Prefetch, self).__init__()
        self.train = train
        self.data_raw = data
        self.buffer_size = buffer_size
        self.on_dev_shape = on_dev_shape
        assert on_dev_shape is not None, "Must have an on dev shape function, can just be an ident"

        first = self.data_raw.element_spec
        self.packer = Packer(first)
        self.data_packed = self.data_raw.map(self.packer.pack, num_parallel_calls=tf.data.AUTOTUNE)

        batch_dim_size = get_batch_dims(first, batch_dims=batch_dims)
        platform = jax.default_backend()
        d_count = jax.device_count(platform)

        logging.log_first_n(logging.INFO, "Running on %s with %d devices", 1, platform, d_count)
        logging.debug("Sharded prefetch to %d devices, assumed new batch shape [%d, %s, ...]", d_count,
                      d_count, ", ".join(str(i) for i in batch_dim_size))

    def iter(self, batch_dims: int = 1):
        queue = collections.deque()
        devices = jax.local_devices()
        dev_count = len(devices)
        data_iter = self.data_packed if FLAGS.prefetch_pack else self.data_raw
        data_iter = data_iter.as_numpy_iterator()

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
        c = random.randint(-100, 100)
        while queue:
            el = queue.popleft()
            el = self.packer.unpack(el) if FLAGS.prefetch_pack else el
            els = self.on_dev_shape(el, c, self.train)
            for el in els:
                yield el
            enqueue(1)


