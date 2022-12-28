import numpy as np
from numpy.typing import NDArray
from core import Module
from activations import Activation, Identity


class Linear(Module):
    """A linear (fully connected) layer with an optional activation function."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: Activation = Identity,
    ) -> None:
        """
        - in_dim: input feature dimension
        - out_dim: output feature dimension
        - activation: activation function to apply after the linear transformation. 
        Default is Identity (no activation).
        """    
        super().__init__()
        # Xavier/Glorot initialization for weights
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.weights = np.random.normal(0, scale, (out_dim, in_dim))
        self.biases = np.zeros(out_dim)

        self.activation = activation

        # For storing gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

        # For storing intermediate values during forward pass
        self.last_input = None
        self.last_linear_output = None

    def forward(self, inp_batch: NDArray) -> NDArray:
        # Store input for backward pass
        self.last_input = inp_batch.copy()

        # Batch: (batch_size, in_dim) -> (batch_size, out_dim)
        linear_output = (self.weights @ inp_batch.T).T + self.biases

        # Store linear output for backward pass
        self.last_linear_output = linear_output.copy()

        # Apply activation function
        output = self.activation(linear_output)

        return output

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        grad_output: gradient of loss w.r.t. this layer's output
        Returns: gradient of loss w.r.t. this layer's input
        """
        # Apply activation derivative
        grad_activation = self.activation.backward(self.last_linear_output)
        grad_linear = grad_output * grad_activation

        # Gradient w.r.t. weights: (out_dim, batch_size) @ (batch_size, in_dim) -> (out_dim, in_dim)
        self.grad_weights = grad_linear.T @ self.last_input

        # Gradient w.r.t. biases: sum over batch dimension
        self.grad_biases = np.sum(grad_linear, axis=0)

        # Gradient w.r.t. input: (batch_size, out_dim) @ (out_dim, in_dim) -> (batch_size, in_dim)
        grad_input = grad_linear @ self.weights

        return grad_input

    def update_weights(self, learning_rate: float) -> None:
        """Update weights using computed gradients using simple gradient descent."""
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases