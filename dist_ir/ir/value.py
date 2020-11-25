class Value:
    def __init__(self, name, value_type, device=None):
        self._name = name
        self._type = value_type
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

    @type.setter
    def type(self, typ):
        self._type = typ

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device