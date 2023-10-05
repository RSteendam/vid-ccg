import string


class Category:
    def __init__(
        self, value, arity: int, modifier: bool, direction: string, base, arg=None
    ):
        self._value = value
        self._modifier = modifier
        self._arity = arity
        self._atomic = arity == 0
        self._key = self._value
        self._hash = hash(self._key)
        self._N = self._value == "N"
        self._S = self._value == "S"
        self.direction = direction
        self.base = base
        self.arg = arg

    def __str__(self):
        return self._value

    def __repr__(self):
        return self._value

    def __key(self):
        return self._key

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Category):
            return self.__key() == other.__key()
        return NotImplemented

    @property
    def value(self):
        return self._value

    @property
    def atomic(self):
        return self._atomic

    @property
    def modifier(self):
        return self._modifier

    @property
    def N(self):
        return self._N

    @property
    def S(self):
        return self._S

    @property
    def arity(self):
        return self._arity
