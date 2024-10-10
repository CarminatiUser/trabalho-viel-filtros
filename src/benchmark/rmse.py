import numpy as np
from benchmark.mse import mse

def rmse(original: np.ndarray, compressed: np.ndarray) -> float:
    err = mse(original, compressed)
    return np.sqrt(err)