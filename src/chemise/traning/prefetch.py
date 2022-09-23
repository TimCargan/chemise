import collections
import itertools
import queue
from threading import Thread

import flax.training.prefetch_iterator
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging


class Prefetch_dev(Thread):
    """
    Wrap an iterator with a thead to load and prefetch onto the GPU(s) in a no-blocking way
    """

    def __init__(self, data: iter, buffer_size: int = 3):
        super(Prefetch_dev, self).__init__()
        self.data = data
        self.q = queue.Queue(buffer_size)
        self.buffer_size = buffer_size
        platform = jax.default_backend()
        d_count = jax.device_count(platform)
        logging.log_first_n(logging.INFO, "Running on %s with %d devices", 1, platform, d_count)

    def iter(self):
        queue = collections.deque()
        devices = jax.local_devices()
        dev_count = len(devices)
        def _prefetch(data, devs):
            return jax.device_put_sharded(data, devs)

        first = next(self.data)
        batch_size = get_batch_size(first)
        logging.debug("Sharded prefetch to %d devices, assumed new batch shape [%d, %d, ...]", len(devices),
                      len(devices), batch_size)

        shard_data = [first] + list(itertools.islice(self.data, dev_count - 1))
        queue.append(_prefetch(shard_data, devices[:len(shard_data)]))

        def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
            for _ in range(n):
                batch = list(itertools.islice(self.data, dev_count))
                num_shards = len(batch)
                if num_shards < 1:
                    return
                # 1 shard or n shard of all the same size
                if num_shards == 1 or (num_shards > 1 and get_batch_size(batch[-1]) == batch_size):
                    queue.append(_prefetch(batch, devices[: num_shards]))
                    return
                # n shards where we assume all but the last are the same
                # (if this doesn't hold something funky happened batching)
                n_shards, last_shard = batch[:-1], batch[-1:]
                assert get_batch_size(n_shards[-1]) == batch_size, "Multiple batches of unequal size"
                queue.append(_prefetch(n_shards, devices[: num_shards - 1]))
                queue.append(_prefetch(last_shard, devices[: 1]))

        enqueue(self.buffer_size - 1)  # Fill up the buffer, less the first already in.
        while queue:
            yield queue.popleft()
            enqueue(1)
    def iter_old(self):
        @jax.jit
        def _stack(flat):
            flat_n = [[n[l] for n in flat] for l in range(len(flat[0]))]
            stacked = [jnp.stack(x) for x in flat_n]
            return stacked
        def _prefetch_s(data, devs):
            tree = jax.tree_util.tree_flatten(data[0])[1]
            flat = [jax.tree_util.tree_flatten(d)[0] for d in data]
            stacked = _stack(flat)
            uf = jax.tree_util.tree_unflatten(tree, stacked)
            return uf


        devices = jax.local_devices()
        first = next(self.data)
        batch_size = get_batch_size(first)

        logging.debug("Sharded prefetch to %d devices, assumed new batch shape [%d, %d, ...]", len(devices),
                      len(devices), batch_size)

        shard_data = [first]
        tail = []
        for data in self.data:
            if len(shard_data) == len(devices):
                res = _prefetch(shard_data, devices)
                yield res
                shard_data = []
            if get_batch_size(data) == batch_size:
                shard_data.append(data)
            else:
                tail.append(data)

        if shard_data:
            logging.info("Number of batches % devices != 0, added a step less that total devices")
            yield _prefetch(shard_data, devices[:len(shard_data)])

        if tail:
            logging.info("Small final batch added")
            tail_len = len(tail)
            assert tail_len <= len(devices), "More than device number of tails"
            yield _prefetch(tail, devices[:tail_len])

        # raise StopIteration

    def __iter__(self):
        return self

    def __next(self):
        if data := self.q.get():
            self.q.task_done()
            return data
        raise StopIteration


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


def get_batch_size(ds, batch_dims=1) -> int:
    """
    Get the likely batch size of a pytree of data
    :param ds:
    :param batch_dims: Number of leading dims to consider part of the batch, default 1,
    if grater than 1 returns the product of the dims
    :return:
    """
    flat, _ = jax.tree_util.tree_flatten(ds)
    shape = np.shape(flat[0])
    batch_size = np.prod(shape[:batch_dims])
    return int(batch_size)
