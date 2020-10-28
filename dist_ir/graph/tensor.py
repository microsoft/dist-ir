from .value import Value


class Tensor(Value):
    def __init__(self, name, data):
        super().__init__(name)
        self.data = data

    def shape(self):
        if self.data is None:
            return None
        return self.data.shape
