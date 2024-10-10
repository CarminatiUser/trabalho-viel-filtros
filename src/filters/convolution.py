import numpy as np

def add_padding_to_img(img: np.ndarray, padding_height: int, padding_width: int):
    n, m = img.shape
    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height: n + padding_height, padding_width: m + padding_width] = img
    return padded_img

def conv2d(img: np.ndarray, kernel: np.ndarray, padding=True) -> np.ndarray:
    k_height, k_width = kernel.shape
    img_height, img_width = img.shape

    if padding:
        pad_height = k_height // 2
        pad_width = k_width // 2
        padded_img = add_padding_to_img(img, pad_height, pad_width)
    else:
        padded_img = img

    output = np.zeros((img_height, img_width), dtype=float)

    for i_img in range(img_height):
        for j_img in range(img_width):
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    output[i_img, j_img] += padded_img[i_img + i_kernel, j_img + j_kernel] * kernel[i_kernel, j_kernel]

    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def conv2d_sharp(img: np.ndarray, kernel: np.ndarray, padding=True) -> np.ndarray:
    k_height, k_width = kernel.shape
    img_height, img_width = img.shape

    if padding:
        pad_height = k_height // 2
        pad_width = k_width // 2
        padded_img = add_padding_to_img(img, pad_height, pad_width)
    else:
        padded_img = img

    output = np.zeros((img_height, img_width), dtype=float)

    for i_img in range(img_height):
        for j_img in range(img_width):
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    output[i_img, j_img] += padded_img[i_img + i_kernel, j_img + j_kernel] * kernel[i_kernel, j_kernel]

    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)
