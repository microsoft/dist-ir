class Device:

    device_variable_id = 0

    def __init__(self, device_id, device_type, is_variable=False):
        self._device_id = device_id
        self._device_type = device_type
        self._is_variable = is_variable

    def __str__(self):
        return f"{self._device_id} ({self._device_type})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return other is not None and self._device_id == other._device_id

    def __lt__(self, other):
        return self._device_id < other._device_id

    def __hash__(self):
        return hash(str(self))

    @property
    def device_id(self):
        return self._device_id

    @property
    def device_type(self):
        return self._device_type

    @property
    def is_variable(self):
        return self._is_variable

    @classmethod
    def get_new_device_variable(cls, device_type):
        device_id = f"d{cls.device_variable_id}"
        cls.device_variable_id += 1
        return Device(device_id, device_type, is_variable=True)
