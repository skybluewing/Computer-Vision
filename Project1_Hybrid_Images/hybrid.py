import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import numpy as np
import cv2


def cross_correlation_2d(img, kernel):
    # gray scale image
    if len(img.shape) == 2:
        result = corr_matrix(img, kernel)
    else:
        # RGB image
        m, n, channel = img.shape
        result = np.zeros((m, n, channel))
        for i in range(channel):
            result[:, :, i] = corr_matrix(img[:, :, i], kernel)
    return result


def corr_matrix(img, kernel):
    l, w = kernel.shape  # length and width of the kernel
    length, width = img.shape  # length and width of the image
    # create a new pad for adding kernel size around image
    len_half, wid_half = (l - 1) / 2, (w - 1) / 2
    pad = np.zeros((length + 2 * len_half, width + 2 * wid_half))
    pad[len_half: len_half + length, wid_half: wid_half + width] = img
    flat_matrix = np.zeros((length * width, l * w))
    k = 0
    for i in range(length):
        for j in range(width):
            flat_matrix[k] = pad[i: i + l, j: j + w].reshape((l * w,))
            k += 1
    kernel_reshape = kernel.reshape((l * w,))
    return np.dot(flat_matrix, kernel_reshape).reshape((length, width))


# Flip inverse a kernel
def convolve_2d(img, kernel):
    return cross_correlation_2d(img, kernel[::-1, ::-1])


def gaussian_blur_kernel_2d(sigma, width, height):
    x_half = (width - 1) / 2
    y_half = (height - 1) / 2
    x_sq = np.arange(-x_half, x_half + 1, 1.0) ** 2
    y_sq = np.arange(-y_half, y_half + 1, 1.0) ** 2
    u = np.sqrt(1 / (2 * np.pi)) * sigma * np.exp(-x_sq / (2 * sigma ** 2))
    v = np.sqrt(1 / (2 * np.pi)) * sigma * np.exp(-y_sq / (2 * sigma ** 2))
    kernel_blur = np.outer(u, v) / (np.sum(u) * np.sum(v))
    return kernel_blur


def low_pass(img, sigma, size):
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor):
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)