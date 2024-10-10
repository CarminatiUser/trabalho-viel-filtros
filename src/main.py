import cv2
import numpy as np
from time import time
from filters.gauss import gauss_filter
from filters.low_pass_filter import low_pass_filter
from filters.sobel import sobel_sharpening
from filters.high_pass_filter import high_pass_filter
from filters.canny import canny_edge_detection
from benchmark.mse import mse
from benchmark.psnr import psnr
from benchmark.rmse import rmse
from utils.save_img import save_img
from utils.create_output_dir import create_output_dir
from utils.measure_duration import measure_duration

OUTPUT_DIR = "./output"

def main():
    create_output_dir(OUTPUT_DIR)
    img = cv2.imread('assets/woman.png', cv2.IMREAD_GRAYSCALE)
    save_img(img, OUTPUT_DIR, filename='original')

    """
    Gaussian Filter
    """

    img_gauss, gauss_time_in_seconds = measure_duration(gauss_filter, img=img)
    save_img(img_gauss, OUTPUT_DIR, filename='gauss')

    """
    Sobel Filter
    """

    img_sobel, sobel_time_in_seconds = measure_duration(sobel_sharpening, img=img)
    save_img(img_sobel, OUTPUT_DIR, filename='sobel')

    """
    Canny Filter
    """

    img_canny, canny_time_in_seconds = measure_duration(canny_edge_detection, img=img, low_threshold=15, high_threshold=50)
    save_img(img_canny, OUTPUT_DIR, filename='canny')

    """
    Low-Pass Filter
    """
    
    img_lp_ideal, lp_ideal_time_in_seconds = measure_duration(low_pass_filter, img=img, radius=60, type='ideal')
    img_lp_gauss, lp_gauss_time_in_seconds = measure_duration(low_pass_filter, img=img, radius=60, type='gauss')
    save_img(img_lp_ideal, OUTPUT_DIR, filename='low_pass_ideal')
    save_img(img_lp_gauss, OUTPUT_DIR, filename='low_pass_gauss')

    """
    High-Pass Filter
    """

    img_hp_ideal, hp_ideal_time_in_seconds = measure_duration(high_pass_filter, img=img, radius=120, type='ideal')
    img_hp_gauss, hp_gauss_time_in_seconds = measure_duration(high_pass_filter, img=img, radius=120, type='gauss')
    save_img(img_hp_ideal, OUTPUT_DIR, filename='high_pass_ideal')
    save_img(img_hp_gauss, OUTPUT_DIR, filename='high_pass_gauss')

    """
    Filters X Canny
    """

    # Sobel X Canny
    sobel_canny_psnr = psnr(img_sobel, img_canny)
    sobel_canny_rmse = rmse(img_sobel, img_canny)
    sobel_canny_mse = mse(img_sobel, img_canny)

    # High-Pass Ideal X Canny
    hp_ideal_canny_psnr = psnr(img_hp_ideal, img_canny)
    hp_ideal_canny_rmse = rmse(img_hp_ideal, img_canny)
    hp_ideal_canny_mse = mse(img_hp_ideal, img_canny)

    # High-Pass Gaussian X Canny
    hp_gauss_canny_psnr = psnr(img_hp_gauss, img_canny)
    hp_gauss_canny_rmse = rmse(img_hp_gauss, img_canny)
    hp_gauss_canny_mse = mse(img_hp_gauss, img_canny)

    """
    Benchmarks
    """

    gauss_low_pass_ideal_psnr = psnr(img_gauss, img_lp_ideal)
    gauss_low_pass_ideal_rmse = rmse(img_gauss, img_lp_ideal)
    gauss_low_pass_ideal_mse = mse(img_gauss, img_lp_ideal)
    gauss_low_pass_gauss_psnr = psnr(img_gauss, img_lp_gauss)
    gauss_low_pass_gauss_rmse = rmse(img_gauss, img_lp_gauss)
    gauss_low_pass_gauss_mse = mse(img_gauss, img_lp_gauss)

    print("Filtro espacial de esmaecimento gaussiano x Filtro passa-baixa ideal:")
    print(f"PSNR: {gauss_low_pass_ideal_psnr}, RMSE: {gauss_low_pass_ideal_rmse}, MSE: {gauss_low_pass_ideal_mse}\n")

    print("Filtro espacial de esmaecimento gaussiano x Filtro passa-baixa gaussiano:")
    print(f"PSNR: {gauss_low_pass_gauss_psnr}, RMSE: {gauss_low_pass_gauss_rmse}, MSE: {gauss_low_pass_gauss_mse}\n")

    print(f"Sobel vs Canny - PSNR: {sobel_canny_psnr}, RMSE: {sobel_canny_rmse}, MSE: {sobel_canny_mse}\n")

    print(f"Passa-alta ideal vs Canny - PSNR: {hp_ideal_canny_psnr}, RMSE: {hp_ideal_canny_rmse}, MSE: {hp_ideal_canny_mse}\n")

    print(f"Passa-alta gaussiano vs Canny - PSNR: {hp_gauss_canny_psnr}, RMSE: {hp_gauss_canny_rmse}, MSE: {hp_gauss_canny_mse}\n")

    print(f"Gaussiano: {gauss_time_in_seconds} s")
    print(f"Sobel: {sobel_time_in_seconds} s")
    print(f"Canny: {canny_time_in_seconds} s")
    print(f"Passa-baixa ideal: {lp_ideal_time_in_seconds} s")
    print(f"Passa-baixa gaussiano: {lp_gauss_time_in_seconds} s")
    print(f"Passa-alta ideal: {hp_ideal_time_in_seconds} s")
    print(f"Passa-alta gaussiano: {hp_gauss_time_in_seconds} s")

if __name__ == "__main__":
    main()


    