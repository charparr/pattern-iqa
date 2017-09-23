from similarity_tests.complex_wavelet_ssim import compute_cw_ssim
from similarity_tests.gradient_magnitude import compute_gms
from similarity_tests.feature_similarity import compute_fsim
from similarity_tests.mean_square_error import compute_mse
from similarity_tests.structural_similarity import compute_ssim


def compute_similarity(im1, im2, test_id):
    """ Compute all similarity metrics and maps.

    Parameters
    ----------
    im1, im2 : ndarray
        Image.  Any dimensionality.
    test_id : str
        name of the test (im2) (e.g. Hillshade 90)
    Returns
    -------
    results : dict
        Python dictionary storing all metrics and maps
    """

    results = dict()
    results['id'] = test_id
    results['test_im'] = im2
    results['mse_value'], results['mse_map'] = compute_mse(im1, im2)
    results['ssim_value'], results['ssim_map'] = compute_ssim(im1, im2, 5)
    results['cw_ssim_value'], results['cw_ssim_map'] = compute_cw_ssim(im1, im2, 30)
    results['gms_value'], results['gms_map'] = compute_gms(im1, im2)
    results['fsim_value'], results['pc_max_map'] = compute_fsim(im1, im2)

    return results
