import queue
from threading import Thread

import jax


class Prefetch(Thread):
    """
    Wrap an iterator with a thead to load and prefetch onto the GPU(s) in a no-blocking way
    """
    def __init__(self, data: iter,  buffer_size: int = 3):
        super(Prefetch, self).__init__()
        self.data = data
        self.q = queue.Queue(buffer_size)

    def run(self):
        devices = jax.local_devices()

        def _prefetch(xs):
            return jax.device_put_sharded(list(xs), devices)

        for data in self.data:
            self.q.put(jax.tree_util.tree_map(_prefetch, data))
        self.q.put(None)

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        if data := self.q.get():
            self.q.task_done()
            return data
        raise StopIteration
