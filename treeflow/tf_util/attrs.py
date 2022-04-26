import tensorflow.python.util.nest as nest
import warnings


class AttrsLengthMixin:
    def __len__(self) -> int:
        warnings.warn("Temporary hotfix")
        assert nest._is_attrs(self)
        return len(nest._get_attrs_items(self))


__all__ = ["AttrsLengthMixin"]
