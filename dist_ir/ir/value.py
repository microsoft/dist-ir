class Value:
    def __init__(self, name, type, device=0):
        self._name = name
        self._type = type
        self._device = device

    @property
    def name(self):
        return self._name
