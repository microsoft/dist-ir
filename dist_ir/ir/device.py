class Device:

    device_variable_id = 0

    def __init__(self, device_id, device_type, bound_devices=None):
        self._device_id = device_id
        self._device_type = device_type
        self._bound_devices = bound_devices

    def __str__(self):
        return f"{self._device_id} ({self._device_type})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return other is not None and self._device_id == other._device_id

    def __hash__(self):
        return hash(str(self))

    @property
    def device_id(self):
        return self._device_id

    @property
    def device_type(self):
        return self._device_type

    @property
    def bound_devices(self):
        if self._bound_devices is None:
            return [self]
        else:
            return self._bound_devices

    def is_variable(self):
        return self._bound_devices is not None

    @classmethod
    def get_new_device_variable(cls, device_type, bound_devices):
        device_id = f"d{cls.device_variable_id}"
        cls.device_variable_id += 1
        return Device(device_id, device_type, bound_devices=bound_devices)
