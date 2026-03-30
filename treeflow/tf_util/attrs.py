import attr


class AttrsLengthMixin:
    def __len__(self) -> int:
        return len(attr.fields(type(self)))


__all__ = ["AttrsLengthMixin"]
