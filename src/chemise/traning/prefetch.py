from absl import logging
import queue
from threading import Thread
import jax
import numpy as np


class Prefetch(Thread):
    """
    Wrap an iterator with a thead to load and prefetch onto the GPU(s) in a no-blocking way
    """

    def __init__(self, data: iter, buffer_size: int = 3):
        super(Prefetch, self).__init__()
        self.data = data
        self.q = queue.Queue(buffer_size)
        platform = jax.default_backend()
        d_count = jax.device_count(platform)
        logging.log_first_n(logging.INFO, "Running on %s with %d devices", 1, platform, d_count)


    def run(self):
        devices = jax.local_devices()
        first = next(self.data)
        batch_size = get_batch_size(first)

        logging.debug("Sharded prefetch to %d devices, assumed new batch shape [%d, %d, ...]", len(devices), len(devices), batch_size)

        shard_data = [first]
        tail = []
        for data in self.data:
            if len(shard_data) == len(devices):
                self.q.put(jax.device_put_sharded(shard_data, devices))
                shard_data = []
            if get_batch_size(data) == batch_size:
                shard_data.append(data)
            else:
                tail.append(data)

        if shard_data:
            logging.info("Number of batches % devices != 0, added a step less that total devices")
            self.q.put(jax.device_put_sharded(shard_data, devices[:len(shard_data)]))

        if tail:
            logging.info("Small final batch added")
            tail_len = len(tail)
            assert tail_len <= len(devices), "More than device number of tails"
            self.q.put(jax.device_put_sharded(shard_data, devices[:tail_len]))

        self.q.put(None)

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        if data := self.q.get():
            self.q.task_done()
            return data
        raise StopIteration


def get_batch_size(ds) -> int:
    """
    Get the likely batch size of a pytree of data
    :param ds:
    :return:
    """
    flat, _ = jax.tree_util.tree_flatten(ds)
    shape = np.shape(flat[0])
    return shape[0]
