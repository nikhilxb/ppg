import torch.utils.data.dataset as dataset
from textwrap import indent
from typing import Sequence, Iterable, Any


class Struct:
    """Data storage class with pretty-printing."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs) -> 'Struct':
        self.__dict__.update(kwargs)
        return self

    def keys(self) -> Iterable:
        return self.__dict__.keys()

    def values(self) -> Iterable:
        return self.__dict__.values()

    def items(self) -> Iterable:
        return self.__dict__.items()

    def __repr__(self) -> str:
        return "{{\n{}\n}}".format(
            indent(
                "\n".join("{} =\n{}".format(k, indent(str(v), " " * 4)) for k, v in self.items()),
                " " * 4
            )
        )

    def __str__(self) -> str:
        return self.__repr__()


class SequenceDataset(dataset.Dataset):
    """Defines a dataset from a sequence of examples."""

    def __init__(self, examples: Sequence):
        self.examples: Sequence = examples

    def __getitem__(self, i: int) -> Any:
        return self.examples[i]

    def __len__(self) -> int:
        return len(self.examples)
