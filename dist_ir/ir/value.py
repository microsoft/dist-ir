class Value:
    def __init__(self, name, value_type):
        self._name = name
        self._type = value_type

    def __str__(self):
        return f"{self._name}: type={str(self._type)}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self._name == other._name and self._type == other._type

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, typ):
        self._type = typ
