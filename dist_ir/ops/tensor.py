class Tensor:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def shape(self):
        if self.data is None:
            return None
        return self.data.shape
