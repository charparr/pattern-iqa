import numpy as np
from scipy import signal
from timeit import default_timer as timer


def compute_cw_ssim(im1, im2, width):
    """
    Compute the Complex Wavelet SSIM (CW-SSIM) similarity.  A major drawback of the spatial domain SSIM algorithm is
    that it is highly sensitive to translation, scaling and rotation of images. Extending the method to the complex
    wavelet transform domain makes it insensitive to these “non-structured” image distortions that are typically caused
    by the movement of the image acquisition devices, rather than the changes of the structures of the objects in the
    visual scene.
    Args:
      reference image: must be same dims as target
      target image: must be same dims as reference
      width: width for the wavelet convolution (default: 30)
    Returns:
      Computed CW-SSIM float value and map.
    """
    start = timer()
    print("Computing Complex Wavelet SSIM...")
    # Define a width for the wavelet convolution
    widths = np.arange(1, width + 1)

    # Unwrap image arrays to 1 dimensional arrays
    sig1 = np.ravel(im1)
    sig2 = np.ravel(im2)

    # Perform a continuous wavelet transform (cwt) using the Ricker (a.k.a. Mexican Hat a.k.a. Marr) wavelet
    # widths are the widths of the wavelet
    # The Ricker wavelet is the the negative normalized second derivative of a Gaussian function
    cwt1 = signal.cwt(sig1, signal.ricker, widths)
    cwt2 = signal.cwt(sig2, signal.ricker, widths)

    # Compute the first component
    # First compute the product of the absolute value of the cwt of im1 and the absolute value of the cwt of im2
    abscwt1_abscwt2 = np.multiply(abs(cwt1), abs(cwt2))
    # Next compute the square of of the absolute values of the cwt for each image
    cwt1_abs_square = np.square(abs(cwt1))
    cwt2_abs_square = np.square(abs(cwt2))
    component1_top = 2 * np.sum(abscwt1_abscwt2, axis=0) + 0.01
    component1_bottom = np.sum(cwt1_abs_square, axis=0) + np.sum(cwt2_abs_square, axis=0) + 0.01
    component_1 = component1_top / component1_bottom

    # Compute the second component
    # This is determined by the consistency of phase changes between the two images.
    # The structural information of local image features is mainly contained in the relative phase patterns of the
    # wavelet coefficients.
    # Consistent phase shit of all coefficients does not change the structure of the local image feature.
    # First compute the product of the cwt of the first image and the complex conjugate of the cwt of the second image
    cwt1_conj_cwt2 = np.multiply(cwt1, np.conjugate(cwt2))
    component2_top = 2 * np.abs(np.sum(cwt1_conj_cwt2, axis=0)) + 0.01
    component2_bottom = 2 * np.sum(np.abs(cwt1_conj_cwt2), axis=0) + 0.01
    component_2 = component2_top / component2_bottom

    # Compute the CW-SSIM index
    cw_ssim_map = (component_1 * component_2).reshape(im1.shape[0], im1.shape[1])
    # Average the per pixel results
    cw_ssim_index = round(np.average(cw_ssim_map), 3)
    end = timer()
    print("Computing Complex Wavelet SSIM...Complete. Elapsed Time [s]: " + str(end - start))

    return cw_ssim_index, cw_ssim_map
