import numpy as np
import cv2

HIGH_PASS_IDEAL = 'ideal'
HIGH_PASS_GAUSS = 'gauss'

def generate_high_pass_filter(shape, center, radius, type=HIGH_PASS_GAUSS, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)

    lpFilter = np.zeros(shape, np.float32)
    if type == HIGH_PASS_IDEAL:  # Ideal high pass filter
        lpFilter[d >= radius] = 1
    elif type == HIGH_PASS_GAUSS: # Gaussian Highpass Filter 
        lpFilter = 1.0 - np.exp(-d**2 / (2 * (radius**2)))

    lpFilter_matrix = np.zeros((rows, cols, 2), np.float32)
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    
    return lpFilter_matrix



def high_pass_filter(img: np.ndarray, radius=60, type=HIGH_PASS_GAUSS) -> np.ndarray:
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    image_f32 = np.float32(img)
    dft = cv2.dft(image_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = generate_high_pass_filter(dft_shift.shape[:2], center=(ccol, crow), radius=radius, type=type)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    filtered_img = np.abs(img_back)
    filtered_img -= filtered_img.min()
    filtered_img = (filtered_img * 255) / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img



