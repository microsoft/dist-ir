class PipeDreamScheduler:
    def __init__(self, num_microbatches, async_sgd=False):
        self._num_microbatches = num_microbatches
        self._async_sgd = async_sgd
        if self._async_sgd:
            raise NotImplementedError("Asynchronous SGD is not supported")

    def schedule(self, module, partition_map):
        pass
