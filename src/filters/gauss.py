import numpy as np
from filters.convolution import conv2d

def generate_gauss(sigma=1, size_x=3, size_y=3) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-1,1,size_x), np.linspace(-1,1,size_y))
    calc = 1/((2*np.pi*(sigma**2)))
    exp = np.exp(-(((x**2) + (y**2))/(2*(sigma**2))))
    return exp*calc

def gauss_filter(img : np.ndarray, padding=True) -> np.ndarray:
    gaus_3x3 = generate_gauss(sigma=1, size_x=3, size_y=3)
    return conv2d(img = img, kernel = gaus_3x3, padding=padding)



