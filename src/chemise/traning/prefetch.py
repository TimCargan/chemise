import collections
import itertools
import queue
from threading import Thread

import jax
import numpy as np
import tensorflow as tf
from absl import logging


class Prefetch_dev:
    """
    Wrap an iterator with a thead to load and prefetch onto the GPU(s) in a no-blocking way
    """

    def __init__(self, data: tf.data.Dataset, buffer_size: int = 3, batch_dims:int = 1):
        super(Prefetch_dev, self).__init__()
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

        self.data = self.data_ds.map(pack_tree, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()

        @jax.pmap
        @jax.jit
        def unpack(stacked):
            unorder = [stacked[i][si] for i, idxs in enumerate(sizes.values()) for si, ti in enumerate(idxs)]
            order = [unorder[un_pack_lookups[i]] for i in range(len(unorder))]
            t = jax.tree_util.tree_unflatten(tree_struct, order)
            return t
        self.unpack = unpack


    def iter(self, batch_dims:int = 1):
        queue = collections.deque()
        devices = jax.local_devices()
        dev_count = len(devices)

        def stack_els(ls, devs):
            # flat = [jax.tree_util.tree_flatten(d)[0] for d in ls]
            # # pack numpy arrays to minimise number of H2D ops
            # packed = []
            # for ft in flat:
            #     p = [np.stack([ft[i] for i in idx]) for dim, idx in sizes.items()]
            #     packed.append(p)

            # Send H2D
            flat_n = [[t[l] for t in ls] for l in range(len(ls[0]))]
            stacked = [jax.device_put_sharded(x, devs) for x in flat_n]

            # # Slice in and fill empy list for leaves of tree struct
            # un_packed = unpack(stacked)
            # t = jax.tree_util.tree_unflatten(tree_struct, un_packed)
            return stacked

        def _prefetch(data, devs):
            stack_tree = stack_els(data, devs)
            return stack_tree

        shard_data =list(itertools.islice(self.data, dev_count))
        prf = _prefetch(shard_data, devices[:len(shard_data)])
        queue.append(prf)

        def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
            for _ in range(n):
                batch = list(itertools.islice(self.data, dev_count))
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
                    # # End of batch, add un-even to queue
                    # batch_sizes = collections.defaultdict(list)
                    # for i, s in enumerate(bs):
                    #     batch_sizes[s].append(i)
                    #
                    # logging.debug(f"End of batch things, {batch_sizes}")
                    #
                    # sizes = sorted(batch_sizes.keys(), reverse=True)
                    # for bs in sizes:
                    #     batch_part = [el for i, el in enumerate(batch) if i in batch_sizes[bs]]
                    #     queue.append(_prefetch(batch_part, devices[: len(batch_part)]))

        enqueue(self.buffer_size)  # Fill up the buffer, less the first already in.
        while queue:
            el = queue.popleft()
            tree = self.unpack(el)
            yield tree
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
