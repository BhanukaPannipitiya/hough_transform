import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import feature
import cv2

def hough_line(img, angle_step=1):
    edges = feature.canny(img, sigma=2)
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = edges.shape
    diag_len = int(round(np.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos, edges

def draw_lines(img, accumulator, thetas, rhos):
    threshold = 100
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(len(rhos)):
        for j in range(len(thetas)):
            if accumulator[i, j] > threshold:
                rho = rhos[i]
                theta = thetas[j]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                img_rgb = cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img_rgb

def segment_edges(img):
    edges = feature.canny(img, sigma=2)
    img_rgb = np.stack((img,) * 3, axis=-1)
    
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 2)

    return img_rgb
