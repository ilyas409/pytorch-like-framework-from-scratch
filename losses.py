from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class Loss(ABC):
    """Base class for all loss functions"""

    @abstractmethod
    def forward(self, y_pred: NDArray, y_true: NDArray) -> float:
        pass
    
    @abstractmethod
    def backward(self) -> NDArray:
        pass
    
    def __call__(self, y_pred: NDArray, y_true: NDArray) -> float:
        return self.forward(y_pred, y_true)


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss function"""
    
    def __init__(self, epsilon: float = 1e-15) -> None:
        self.epsilon = epsilon
        self.y_pred = None
        self.y_true = None
        self.y_pred_clipped = None
    
    def forward(self, y_pred: NDArray, y_true: NDArray) -> float:
        """Compute binary cross-entropy loss"""
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Clip predictions to avoid log(0) and log(1)
        self.y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        return -np.mean(y_true * np.log(self.y_pred_clipped) + 
                       (1 - y_true) * np.log(1 - self.y_pred_clipped))
    
    def backward(self) -> NDArray:
        """Compute the actual BCE gradient w.r.t. predictions"""
        if self.y_pred_clipped is None or self.y_true is None:
            raise ValueError("Must call forward() before backward()")
        
        # Actual derivative of BCE w.r.t. predictions:
        # d/dp [-y*log(p) - (1-y)*log(1-p)] = -y/p + (1-y)/(1-p)
        # = (p - y) / (p * (1 - p))
        
        # But this can be numerically unstable, so we use the direct form:
        gradient = -(self.y_true / self.y_pred_clipped - 
                    (1 - self.y_true) / (1 - self.y_pred_clipped))
        
        # Average over batch
        return gradient / len(self.y_pred_clipped)