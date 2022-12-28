# Intro

This is a simple neural network implementation for binary classification tasks. This model is part of a PyTorch-inspired library built from scratch with vanilla NumPy for educational and learning purposes.

## Sample Model Definition

```python
class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_0 = Linear(3, 4, ReLU())
        self.fc_1 = Linear(4, 4, ReLU())
        self.fc_output = Linear(4, 1, Sigmoid())
        self.layers = [self.fc_0, self.fc_1, self.fc_output]
```

## Key Methods

### Forward Pass
```python
def forward(self, x: NDArray) -> NDArray:
    x = self.fc_0(x)
    x = self.fc_1(x)
    x = self.fc_output(x)
    return x
```

### Backward Pass
```python
def backward(self, grad_output: NDArray):
    """Backpropagate gradients through all layers"""
    grad = grad_output
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
    return grad
```

### Weight Updates
```python
def update_weights(self, learning_rate: float):
    """Update weights for all layers"""
    for layer in self.layers:
        layer.update_weights(learning_rate)
```

### Gradient Reset
```python
def zero_gradients(self):
    """Reset gradients to zero"""
    for layer in self.layers:
        layer.grad_weights.fill(0)
        layer.grad_biases.fill(0)
```

## Usage Example

### 1. Create Model Instance
```python
model = Model()
```

### 2. Prepare Training Data
```python
# Example XOR-like problem
X_train = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 1, 1]], dtype=np.float32)
y_train = np.array([[0], [1], [1], [0]], dtype=np.float32)
```

### 3. Set Training Parameters
```python
learning_rate = 0.1
epochs = 1000
criterion = BinaryCrossEntropy()
```

### 4. Training Loop
```python
for epoch in range(epochs):
    # Reset gradients
    model.zero_gradients()
    
    # Forward pass
    predictions = model(X_train)
    
    # Compute loss
    loss = criterion(predictions, y_train)
    
    # Compute loss gradient
    loss_gradient = criterion.backward()
    
    # Backward pass
    model.backward(loss_gradient)
    
    # Update weights
    model.update_weights(learning_rate)
    
    # Calculate accuracy
    acc = accuracy(predictions, y_train)
    
    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

### 5. Make Predictions
```python
# Test the trained model
final_predictions = model(X_train)
for i, (input_val, pred, true) in enumerate(zip(X_train, final_predictions, y_train)):
    print(f"Input: {input_val}, Predicted: {pred[0]:.4f}, True: {true[0]}")
```

## Training Process

The training follows these steps in each epoch:

1. **Reset Gradients**: Clear accumulated gradients from previous iteration
2. **Forward Pass**: Compute predictions by passing input through all layers
3. **Loss Calculation**: Measure error using Binary Cross Entropy
4. **Gradient Computation**: Calculate gradients of loss with respect to predictions
5. **Backward Pass**: Propagate gradients back through all layers
6. **Weight Update**: Adjust weights using computed gradients and learning rate

## Expected Output

During training, you should see decreasing loss and increasing accuracy:

```
Epoch 0, Loss: 0.7234, Accuracy: 0.5000
Epoch 100, Loss: 0.6543, Accuracy: 0.7500
Epoch 200, Loss: 0.5821, Accuracy: 0.7500
...
```

Final predictions will show the model's learned behavior on the training data.

## Use Cases

This model architecture is suitable for:
- Binary classification problems
- Small datasets that can fit in memory
- Learning XOR-like logical patterns
- Educational purposes and experimentation

## Customization

To adapt this model for your use case:
- Change input/output dimensions in layer definitions
- Modify activation functions (ReLU, Sigmoid, etc.)
- Adjust learning rate and number of epochs
- Replace loss function for different problem types