from abc import ABC, abstractmethod
from numpy.typing import NDArray

class Module(ABC):
    """
    Base class for all neural network modules
    """
    def __init__(self) -> None:
        self.training = True

    def __call__(self, inp):
        return self.forward(inp)

    @abstractmethod
    def forward(self, inp: NDArray) -> NDArray:
        pass

    @abstractmethod
    def backward(self, grad_output: NDArray) -> NDArray:
        pass

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False
