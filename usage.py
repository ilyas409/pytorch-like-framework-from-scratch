import numpy as np
from activations import ReLU, Sigmoid
from core import Module
from linear import Linear
from losses import BinaryCrossEntropy
from numpy.typing import NDArray
from utils import accuracy


class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_0 = Linear(3, 4, ReLU())
        self.fc_1 = Linear(4, 4, ReLU())
        self.fc_output = Linear(4, 1, Sigmoid())

        self.layers = [self.fc_0, self.fc_1, self.fc_output]

    def forward(self, x: NDArray) -> NDArray:
        x = self.fc_0(x)
        x = self.fc_1(x)
        x = self.fc_output(x)
        return x

    def backward(self, grad_output: NDArray):
        """Backpropagate gradients through all layers"""
        grad = grad_output
        # Go through layers in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self, learning_rate: float):
        """Update weights for all layers"""
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def zero_gradients(self):
        """Reset gradients to zero"""
        for layer in self.layers:
            layer.grad_weights.fill(0)
            layer.grad_biases.fill(0)

model = Model()

# Example training data (XOR-like problem)
X_train = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 1, 1]], dtype=np.float32)

y_train = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Training parameters
learning_rate = 0.1
epochs = 1000

print("Training the model...")

criterion = BinaryCrossEntropy()

# Training loop
for epoch in range(epochs):
    # Reset gradients
    model.zero_gradients()

    # Forward pass
    predictions = model(X_train)

    # Compute loss
    loss = criterion(predictions, y_train)

    # Gradient of loss w.r.t. predictions
    loss_gradient = criterion.backward()

    # Backward pass
    model.backward(loss_gradient)

    # Update weights
    model.update_weights(learning_rate)

    # Accuracy
    acc = accuracy(predictions, y_train)
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Test the trained model
print("\nFinal predictions:")
final_predictions = model(X_train)
for i, (input_val, pred, true) in enumerate(zip(X_train, final_predictions, y_train)):
    print(f"Input: {input_val}, Predicted: {pred[0]:.4f}, True: {true[0]}")
