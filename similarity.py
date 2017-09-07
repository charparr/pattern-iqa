import argparse
import os
import phasepack
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mean_square_error import compute_mse
from complex_wavelet_ssim import compute_cw_ssim
from structural_similarity import compute_ssim
from gradient_magnitude import compute_gms
from feature_similarity import compute_fsim

im1 = cv2.imread('/home/cparr/workspace/pattern_similarity/test_images/tired_run_jpeg10.jpg')
im2 = cv2.imread('/home/cparr/workspace/pattern_similarity/test_images/tired_run_jpeg10.jpg')

# If the image is RGB, convert to single band grayscale

if len(im1.shape) > 2:
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

if len(im2.shape) > 2:
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)


mse_value, mse_map = compute_mse(im1, im2)
ssim_value, ssim_map = compute_ssim(im1, im2, 5)
cw_ssim_value, cw_ssim_map = compute_cw_ssim(im1, im2, 30)
gms_value, gms_map = compute_gms(im1, im2)
fsim_value = compute_fsim(im1, im2)

print("MSE: ", mse_value)
print("SSIM: ", ssim_value)
print("CW-SSIM: ", cw_ssim_value)
print("GMSD: ", gms_value)
print("FSIM: ", fsim_value)

plt.imshow(mse_map)
plt.title('MSE')
plt.colorbar()
plt.show()

plt.imshow(ssim_map)
plt.title('SSIM')
plt.colorbar()
plt.show()

plt.imshow(cw_ssim_map)
plt.title('CW-SSIM')
plt.colorbar()
plt.show()

plt.imshow(gms_map)
plt.title('GMS')
plt.colorbar()
plt.show()