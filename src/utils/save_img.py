import cv2
import numpy as np

def save_img(img: np.ndarray, directory: str, filename: str) -> None:
    cv2.imwrite(directory + '/' + filename + '.png', img)