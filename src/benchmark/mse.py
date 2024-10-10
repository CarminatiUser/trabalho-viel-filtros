import numpy as np

def mse(original: np.ndarray, compressed: np.ndarray) -> float:
    err = np.sum((original.astype(float) - compressed.astype(float)) ** 2)
    err /= float(original.shape[0] * original.shape[1])
    return err