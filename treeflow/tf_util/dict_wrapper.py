from __future__ import annotations

import typing as tp
from collections.abc import MutableMapping

_TKey = tp.TypeVar("_TKey")
_TValue = tp.TypeVar("_TValue")


class DictWrapper(MutableMapping[_TKey, _TValue]):
    """
    Provides a base class that implements `collections.abc.MutableMapping` by wrapping
    a dictionary and allows custom types to work with `tf.nest`

    `MutableMapping` is preferred to `Mapping` according to `tf.nest.sequence_like`.
    `namedtuple` cannot be generic so we prefer `Mapping` types.
    """

    class_keys: tp.Set[_TKey] = set()

    def __init_subclass__(cls: tp.Type[DictWrapper]) -> None:
        assert cls.class_keys, "Keys must be overridden"
        return super().__init_subclass__()

    def __init__(
        self,
        mapping: tp.Optional[tp.Mapping[_TKey, _TValue]] = None,
        values: tp.Optional[tp.Mapping[_TKey, tp.Optional[_TValue]]] = None,
    ):
        if mapping:  # TODO: Is key checking logic called in function mode?
            for key in mapping:
                assert key in type(self).class_keys
            self._dict = dict(mapping)
        else:
            self._dict = dict()
        if values:
            for key, value in values.items():
                assert key in type(self).class_keys
                if value:
                    self[key] = value

    def __getitem__(self, k: _TKey) -> _TValue:
        return self._dict.__getitem__(k)

    def __setitem__(self, k: _TKey, v: _TValue) -> None:
        assert k in type(self).class_keys
        return self._dict.__setitem__(k, v)

    def __iter__(self) -> tp.Iterator[_TKey]:
        return self._dict.__iter__()

    def __repr__(self) -> str:
        attr_string = ", ".join(
            [f"{key}={repr(self[key])}" for key in type(self).class_keys]
        )
        return f"{type(self).__name__}({attr_string})"

    def __delitem__(self, k: _TKey) -> None:
        return self._dict.__delitem__(k)

    def __len__(self) -> int:
        return self._dict.__len__()


__all__ = [DictWrapper.__name__]
