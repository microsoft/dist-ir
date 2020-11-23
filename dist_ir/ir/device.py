class Device:
    def __init__(self, device_id, device_type):
        self._device_id = device_id
        self._device_type = device_type

    def __str__(self):
        return f"{self._device_id} ({self._device_type})"

    def __eq__(self, other):
        return self._device_id == other._device_id

    def __hash__(self):
        return hash(str(self))

    @property
    def device_id(self):
        return self._device_id

    @property
    def device_type(self):
        return self._device_type
