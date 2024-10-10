import numpy as np
import cv2

LOW_PASS_IDEAL = 'ideal'
LOW_PASS_GAUSS = 'gauss'

def generate_low_pass_filter(shape, center, radius, type=LOW_PASS_GAUSS, n=2) -> np.ndarray:
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))
    lpFilter = np.zeros((rows, cols), np.float32)

    if type == LOW_PASS_IDEAL:
        lpFilter[d <= radius] = 1
    elif type == LOW_PASS_GAUSS:
        lpFilter = np.exp(-d**2 / (2 * (radius**2)))

    lpFilter_matrix = np.zeros((rows, cols, 2), np.float32)
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter

    return lpFilter_matrix

def low_pass_filter(img: np.ndarray, radius=60, type=LOW_PASS_GAUSS) -> np.ndarray:
    rows, cols = img.shape
    center_row, center_col = rows // 2, cols // 2

    image_f32 = np.float32(img)
    dft = cv2.dft(image_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = generate_low_pass_filter(dft_shift.shape[:2], center=(center_col, center_row), radius=radius, type=type)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    filtered_img = np.abs(img_back)
    filtered_img -= filtered_img.min()
    filtered_img = (filtered_img * 255) / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img







