import collections
import itertools
import queue
from threading import Thread

import jax
import numpy as np
from absl import logging


class Prefetch_dev:
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

        enqueue(self.buffer_size - 1)  # Fill up the buffer, less the first already in.
        while queue:
            yield queue.popleft()
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
