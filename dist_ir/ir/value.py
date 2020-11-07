class Value:
    def __init__(self, name, type, device=0, data=None):
        self._name = name
        self._type = type
        self._device = device
        self._data = data

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def device(self):
        return self._device

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
