import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Activation(ABC):
    """Base class for all activation functions"""

    @abstractmethod
    def forward(self, inp: NDArray) -> NDArray:
        """Compute the forward pass of the activation function."""
        pass

    @abstractmethod
    def backward(self, inp: NDArray) -> NDArray:
        """Compute the backward pass (derivative) of the activation function."""
        pass

    def __call__(self, inp: NDArray) -> NDArray:
        return self.forward(inp)

class ReLU(Activation):
    """ReLU activation function"""

    def forward(self, inp: NDArray) -> NDArray:
        """
        Forward pass of ReLU: max(0, inp)
        """
        return np.maximum(0, inp)

    def backward(self, inp: NDArray) -> NDArray:
        """
        Derivative of ReLU is 1 for inp > 0, else 0
        """
        return (inp > 0).astype(float)

    def __repr__(self) -> str:
        return "ReLU"

class Sigmoid(Activation):
    """Sigmoid activation function with numerical stability"""

    def forward(self, inp: NDArray) -> NDArray:
        """
        Forward pass of the sigmoid function with clipping for numerical stability:
        `sigmoid(x) = 1 / (1 + exp(-x))`, and clip x to avoid overflow
        """
        return 1 / (1 + np.exp(-np.clip(inp, -500, 500)))
    
    def backward(self, inp: NDArray) -> NDArray:
        """
        Derivative of the sigmoid function is s * (1 - s) where s is the sigmoid output
        """
        s = self.forward(inp)
        return s * (1 - s)
    
    def __repr__(self) -> str:
        return "Sigmoid"

class Identity(Activation):
    """Identity activation function (no operation)"""
    
    def forward(self, inp: NDArray) -> NDArray:
        return inp
    
    def backward(self, inp: NDArray) -> NDArray:
        return np.ones_like(inp)