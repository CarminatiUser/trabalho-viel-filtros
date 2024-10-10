import numpy as np
from benchmark.mse import mse as calculate_mse

def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)