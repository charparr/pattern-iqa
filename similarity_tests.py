import argparse
import similarity_plots
import os
import phasepack
import numpy as np
import cv2
import pandas as pd
from skimage import io
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from scipy import signal
from scipy.ndimage import zoom
import itertools


parser = argparse.ArgumentParser()
parser.add_argument('names', metavar='N', type=str, nargs='+')
args = parser.parse_args()
print(args)

class SnowDepthPattern(object):

    def __init__(self, fpath):

        self.fpath = fpath
        #self.arr = zoom(io.imread(fpath), 2, order=1)
        self.arr = io.imread(fpath)
        self.shape = self.arr.shape
        self.size = self.arr.size
        self.mu = np.mean(self.arr)
        self.sigma = np.std(self.arr)


class PatternComparison(object):

    def __init__(self, p1, p2):
        """Creates a pair of snow patterns to compare them"""
        self.p1 = p1
        self.p2 = p2

    def convolve(self, image, kernel):

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

    def calc_mse(self):
        """Calculate Mean Square Error"""
        self.mse = round(mse(self.p1, self.p2), 3)
        self.mse_map = (self.p1 - self.p2)**2

    def calc_ssim(self):
        """Calculating Structural Similarity Index"""
        self.ssim_results = ssim(self.p1, self.p2, full=True)
        self.ssim = round(self.ssim_results[0], 3)
        self.ssim_map = self.ssim_results[1]

    def calc_cw_ssim(self, width):
        """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
        image to the target image.
        Args:
          reference image: must be same dims as target
          target image: must be same dims as reference
          width: width for the wavelet convolution (default: 30)
        Returns:
          Computed CW-SSIM float value and map.
        """

        # Define a width for the wavelet convolution
        widths = np.arange(1, width+1)

        # Use the image data as arrays
        sig1 = np.ravel(self.p1)
        sig2 = np.ravel(self.p2)

        # Convolution
        cwtmatr1 = signal.cwt(sig1, signal.ricker, widths)
        cwtmatr2 = signal.cwt(sig2, signal.ricker, widths)

        # Compute the first term
        c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
        c1_2 = np.square(abs(cwtmatr1))
        c2_2 = np.square(abs(cwtmatr2))
        num_ssim_1 = 2 * np.sum(c1c2, axis=0) + 0.01
        den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + 0.01

        # Compute the second term
        c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
        num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + 0.01
        den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + 0.01

        # Construct the result
        cw_ssim_map = (num_ssim_1 / den_ssim_1)*(num_ssim_2 / den_ssim_2)
        self.cw_ssim_map = cw_ssim_map.reshape(self.p1.shape[0],
                                               self.p1.shape[1])

        # Average the per pixel results
        self.cw_ssim_index = round(np.average(self.cw_ssim_map), 3)

    def calc_gms(self):
        """
        Return a map of Gradient Magnitude Similarity (GMS) and the global
        GMS Deviation Index (GMSD).

        Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014).
        Gradient magnitude similarity deviation: A highly efficient perceptual
        image quality index.
        IEEE Transactions on Image Processing, 23(2), 668–695.
        http://doi.org/10.1109/TIP.2013.2293423
        """

        # Construct Prewitt kernels with values from literature
        h_x = [0.33, 0, -0.33, 0.33, 0, -0.33, 0.33, 0, -0.33]
        h_x = np.array(h_x).reshape(3, 3)
        h_y = np.flipud(np.rot90(h_x))

        # Create gradient magnitude images with Prewitt kernels
        ref_conv_hx = self.convolve(self.p1, h_x)
        ref_conv_hy = self.convolve(self.p1, h_y)
        ref_grad_mag = np.sqrt((ref_conv_hx**2) + (ref_conv_hy**2))

        dst_conv_hx = self.convolve(self.p2, h_x)
        dst_conv_hy = self.convolve(self.p2, h_y)
        dst_grad_mag = np.sqrt((dst_conv_hx**2) + (dst_conv_hy**2))

        c = 0.0026  # Constant provided by literature

        self.gms_map = (2 * ref_grad_mag * dst_grad_mag + c) / (ref_grad_mag ** 2 + dst_grad_mag ** 2 + c)
        self.gms_index = np.sum((self.gms_map - self.gms_map.mean())**2) / (self.gms_map.size ** 0.5)

    def calc_fsim(self):
        """
        Return the Feature Similarity Index (FSIM).
        Can also return FSIMc for color images

        Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011).
        FSIM: A feature similarity index for image quality assessment.
        IEEE Transactions on Image Processing, 20(8), 2378–2386.
        http://doi.org/10.1109/TIP.2011.2109730
        """

        t1 = 0.85 # Constant from literature
        t2 = 160 # Constant from literature

        # Phase congruency (PC) images are a dimensionless measure for the
        # significance of local structure.
        pc1 = phasepack.phasecong(self.p1, nscale=4, norient=4,
                                  minWaveLength=6, mult=2, sigmaOnf=0.55)
        pc2 = phasepack.phasecong(self.p2, nscale=4, norient=4,
                                  minWaveLength=6, mult=2, sigmaOnf=0.55)
        pc1 = pc1[0]  # Reference PC map
        pc2 = pc2[0]  # Distorted PC map

        # Similarity of PC components
        s_PC = (2*pc1*pc2 + t1) / (pc1**2 + pc2**2 + t1)

        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        refgradX = cv2.Sobel(self.p1, cv2.CV_64F, dx=1, dy=0, ksize=-1)
        refgradY = cv2.Sobel(self.p1, cv2.CV_64F, dx=0, dy=1, ksize=-1)
        targradX = cv2.Sobel(self.p2, cv2.CV_64F, dx=1, dy=0, ksize=-1)
        targradY = cv2.Sobel(self.p2, cv2.CV_64F, dx=0, dy=1, ksize=-1)
        refgradient = np.maximum(refgradX, refgradY)
        targradient = np.maximum(targradX, targradY)

        #refgradient = np.sqrt(( refgradX**2 ) + ( refgradY**2 ))
        #targradient = np.sqrt(( targradX**2 ) + ( targradY**2 ))

        # The gradient magnitude similarity
        self.s_G = (2*refgradient*targradient + t2) / (refgradient**2 + targradient**2 + t2)
        self.s_L = s_PC * self.s_G  # luma similarity
        self.pcM = np.maximum(pc1, pc2)
        self.fsim = round( np.nansum(self.s_L * self.pcM) / np.nansum(self.pcM), 3)

snow_depth_15 = SnowDepthPattern('/home/cparr/workspace/pattern_similarity/soldier/data/soldier_20150325.tif')
snow_depth_14 = SnowDepthPattern('/home/cparr/workspace/pattern_similarity/soldier/data/soldier_20140407.tif')
snow_depth_13 = SnowDepthPattern('/home/cparr/workspace/pattern_similarity/soldier/data/soldier_20130403.tif')

surfs = [snow_depth_15, snow_depth_13, snow_depth_14]

comp_pairs = [c for c in itertools.combinations(surfs, 2)]

comp_results = dict()
df = pd.DataFrame(columns=['MSE', 'SSIM', 'CW-SSIM', 'GMS', 'FSIM'])

for p in comp_pairs:

    cname = p[0].fpath[-12:-4] + '__vs__' + p[1].fpath[-12:-4]
    print("Performing " + cname + " Comparison...")

    comp = PatternComparison(p[0].arr, p[1].arr)
    comp.calc_mse()
    print('MSE: %.3f' % comp.mse)
    comp.calc_ssim()
    print('SSIM: %.3f' % comp.ssim)
    comp.calc_cw_ssim(30)
    print('CW-SSIM: %.3f' % comp.cw_ssim_index)
    comp.calc_gms()
    print('GMS Index: %.3f' % comp.gms_index)
    comp.calc_fsim()
    print('Feature Similarity: %3f' % comp.fsim)
    comp.cname = cname

    df.loc[cname] = [comp.mse, comp.ssim, comp.cw_ssim_index, comp.gms_index, comp.fsim]

    results_dir = '/home/cparr/workspace/pattern_similarity/soldier/results/' + cname
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    comp.results_dir = results_dir

    # comp_results[cname + " MSE"] = comp.mse
    # comp_results[cname + " SSIM"] = comp.ssim
    # comp_results[cname + " CW-SSIM"] = comp.cw_ssim_index
    # comp_results[cname + " GMS"] = comp.gms_index
    # comp_results[cname + " FSIM"] = comp.fsim


    similarity_plots.generate_plots(comp)

df.to_csv('/home/cparr/workspace/pattern_similarity/soldier/results/metrics.csv')

