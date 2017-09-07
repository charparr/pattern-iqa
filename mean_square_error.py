from skimage.measure import compare_mse


def compute_mse(im1, im2):
    """
    Calculate the Mean Square Error of Two Single Band Images.
    The two input images must be the same size and shape.
    A valid comparison will return two objects:
    1.) A Mean Square Error Value
    2.) The map (array) of element-wise square errors.
    """
    print("Computing Mean Square Error...")
    mse_value = round(compare_mse(im1, im2), 3)
    square_error_map = (im1 - im2) ** 2

    print("Computing Mean Square Error...Complete.")
    return mse_value, square_error_map
