import numpy as np
import cv2


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced

    pad = int((kW - 1) / 2)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions

            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise product between the ROI and
            # the kernel, then summing the matrix

            k = (roi * kernel).sum()

            # store the convolved value in the output (x,y)
            # coordinate of the output image
            output[y - pad, x - pad] = k
    return output


def compute_gms(im1, im2):
    """
    Return a map of Gradient Magnitude Similarity (GMS) and the global
    GMS Deviation Index (GMSD).

    Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014).
    Gradient magnitude similarity deviation: A highly efficient perceptual
    image quality index.
    IEEE Transactions on Image Processing, 23(2), 668–695.
    http://doi.org/10.1109/TIP.2013.2293423
    """

    print("Computing Gradient Magnitude Similarity...")
    # Construct Prewitt kernels with values from literature
    h_x = [0.33, 0, -0.33, 0.33, 0, -0.33, 0.33, 0, -0.33]
    h_x = np.array(h_x).reshape(3, 3)
    h_y = np.flipud(np.rot90(h_x))

    # Create gradient magnitude images for each image with Prewitt kernels
    # Reference (im1)
    ref_conv_hx = convolve(im1, h_x)
    ref_conv_hy = convolve(im1, h_y)
    ref_grad_mag = np.sqrt((ref_conv_hx ** 2) + (ref_conv_hy ** 2))

    # 'Distorted' (im2)
    dst_conv_hx = convolve(im2, h_x)
    dst_conv_hy = convolve(im2, h_y)
    dst_grad_mag = np.sqrt((dst_conv_hx ** 2) + (dst_conv_hy ** 2))

    c = 0.0026  # Constant provided by literature

    gms_map = (2 * ref_grad_mag * dst_grad_mag + c) / (ref_grad_mag ** 2 + dst_grad_mag ** 2 + c)
    gms_index = np.sqrt(np.sum((gms_map - gms_map.mean()) ** 2) / gms_map.size)

    print("Computing Gradient Magnitude Similarity...Complete.")

    return gms_index, gms_map
