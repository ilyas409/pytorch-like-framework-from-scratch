import numpy as np
from numpy.typing import NDArray

def accuracy(y_pred: NDArray, y_true: NDArray) -> float:
    """Compute accuracy"""
    y_pred_class = np.round(y_pred)
    return np.mean(y_pred_class == y_true)