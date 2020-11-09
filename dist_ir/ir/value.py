class Value:
    def __init__(self, name, type, device=0):
        self._name = name
        self._type = type
        self._device = device

    def __str__(self):
        return f"{self._name}: type={str(self._type)}, device={self._device}"

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def device(self):
        return self._device
