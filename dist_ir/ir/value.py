class Value:
    def __init__(self, name, value_type):
        self._name = name
        self._type = value_type

    def __str__(self):
        return f"{self._name}: type={str(self._type)}"

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
