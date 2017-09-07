from skimage.measure import compare_ssim


def compute_ssim(im1, im2, win_size):
    """
    Calculate the Structural Similarity of two single band images.
    The two input images must be the same size and shape.
    Parameters:
        im1: Single band image (array)
        im2: Single band image(array)
        win_size: length of the sliding comparison window. must be odd. (int)
    A valid comparison will return two objects:
    1.) A Mean Global Structural Similarity Value
    2.) The map (array) of element-wise structural similarity.
    The Structural Similarity Index Method (SSIM) is a technique grounded in the idea that the human visual system is
    well adapted to extracting structural information from a scene. From Wang et al., 2004:
    "We define structural information in an image as those attributes that represent the structure of objects in the
    scene, independent of the average luminance and contrast. Since luminance and contrast can vary across a scene, we
    use the local luminance and contrast for our definition. Luminance, contrast, and structure comparisons make up the
    SSIM method.
    Brightness in the SSIM map indicates the magnitude of the local SSIM index (squared for visibility).

    """

    if win_size % 2 != 1:
        print('Please provide an odd integer for the window size.')
    else:
        print("Computing Structural Similarity Index...")
        #mean_ssim = compare_ssim(im1, im2, win_size=win_size, full=False)
        ssim = compare_ssim(im1, im2, win_size=win_size, full=True)
        print("Computing Structural Similarity Index...Complete.")
        return ssim

