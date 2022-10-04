import collections
import itertools
import queue
from threading import Thread

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

    def iter(self, with_meta=False, batch_dims:int = 1):
        queue = collections.deque()
        devices = jax.local_devices()
        dev_count = len(devices)
        def _prefetch(data, devs):
            if with_meta:
                batch = [d[0] for d in data]
                meta = data[0][1:]
                prefetched_data = jax.device_put_sharded(batch, devs)
                return (prefetched_data, *meta)

            return jax.device_put_sharded(data, devs)

        first = next(self.data)
        batch_dim_size = get_batch_dims(first, batch_dims=batch_dims)
        logging.debug("Sharded prefetch to %d devices, assumed new batch shape [%d, %s, ...]", len(devices),
                      len(devices), ", ".join(str(i) for i in batch_dim_size))

        batch_size = int(np.prod(batch_dim_size))
        shard_data = [first] + list(itertools.islice(self.data, dev_count - 1))
        queue.append(_prefetch(shard_data, devices[:len(shard_data)]))

        def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
            for _ in range(n):
                batch = list(itertools.islice(self.data, dev_count))
                num_shards = len(batch)
                if num_shards < 1:
                    return
                # 1 shard or n shard of all the same size
                if num_shards == 1 or (num_shards > 1 and get_batch_size(batch[-1], batch_dims=batch_dims) == batch_size):
                    queue.append(_prefetch(batch, devices[: num_shards]))
                    return

                # End of batch, add un-even to queue
                batch_sizes = {}
                for i, s in enumerate([get_batch_size(el, batch_dims=batch_dims) for el in batch]):
                    cur = batch_sizes.get(s, [])
                    cur.append(i)
                    batch_sizes[s] = cur

                sizes = sorted(batch_sizes.keys(), reverse=True)
                for bs in sizes:
                    batch_part = [el for i, el in enumerate(batch) if i in batch_sizes[bs]]
                    queue.append(_prefetch(batch_part, devices[: len(batch_part)]))

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


def get_batch_dims(ds, batch_dims=1) -> list[int]:
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
