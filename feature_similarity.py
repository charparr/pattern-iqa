import cv2
import phasepack
import numpy as np


def compute_fsim(im1, im2):
    """
    Return the Feature Similarity Index (FSIM).
    Can also return FSIMc for color images

    Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011).
    FSIM: A feature similarity index for image quality assessment.
    IEEE Transactions on Image Processing, 20(8), 2378–2386.
    http://doi.org/10.1109/TIP.2011.2109730
    """

    print("Computing Feature Similarity...")

    t1 = 0.85  # Constant from literature
    t2 = 160  # Constant from literature

    # Phase congruency (PC) images are a dimensionless measure for the
    # significance of local structure.
    pc1 = phasepack.phasecong(im1, nscale=4, norient=4,
                              minWaveLength=6, mult=2, sigmaOnf=0.55)
    pc2 = phasepack.phasecong(im2, nscale=4, norient=4,
                              minWaveLength=6, mult=2, sigmaOnf=0.55)
    pc1 = pc1[0]  # Reference PC map
    pc2 = pc2[0]  # Distorted PC map

    # Similarity of PC components
    s_PC = (2 * pc1 * pc2 + t1) / (pc1 ** 2 + pc2 ** 2 + t1)

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    refgradX = cv2.Sobel(im1, cv2.CV_64F, dx=1, dy=0, ksize=-1)
    refgradY = cv2.Sobel(im1, cv2.CV_64F, dx=0, dy=1, ksize=-1)
    targradX = cv2.Sobel(im2, cv2.CV_64F, dx=1, dy=0, ksize=-1)
    targradY = cv2.Sobel(im2, cv2.CV_64F, dx=0, dy=1, ksize=-1)
    refgradient = np.maximum(refgradX, refgradY)
    targradient = np.maximum(targradX, targradY)

    # refgradient = np.sqrt(( refgradX**2 ) + ( refgradY**2 ))
    # targradient = np.sqrt(( targradX**2 ) + ( targradY**2 ))

    # The gradient magnitude similarity
    s_G = (2 * refgradient * targradient + t2) / (refgradient ** 2 + targradient ** 2 + t2)
    s_L = s_PC * s_G  # luma similarity
    pcM = np.maximum(pc1, pc2)
    fsim = round(np.nansum(s_L * pcM) / np.nansum(pcM), 3)

    print("Computing Feature Similarity...Complete.")

    return fsim
