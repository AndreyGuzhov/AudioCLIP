import abc
import torch

from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional


TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorOrTwo = Union[torch.Tensor, TensorPair]


class AbstractNet(abc.ABC, torch.nn.Module):

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> TensorOrTwo:
        pass

    @abc.abstractmethod
    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def loss_fn_name(self) -> str:
        pass


class AbstractTransform(abc.ABC, Callable[[torch.Tensor], torch.Tensor]):

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'
