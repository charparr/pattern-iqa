import numpy as np
import cv2


def make_slope(dem):
    """ Create slope surface.

    Parameters
    ----------
    dem : ndarray
        Array of surface heights i.e. a DEM.

    Returns
    -------
    slope : ndarray
        slope map.
    """
    x, y = np.gradient(dem)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    return slope


def make_std_slope(slope, win_size):
    """ Create Standard Deviation slope surface.

    Parameters
    ----------
    slope : ndarray
        Array of Slope Values

    Returns
    -------
    sd_slope : ndarray
        Standard Deviation slope map.
    """
    win_mean, win_sqr_mean = (cv2.boxFilter(x, -1, (win_size, win_size),
                                            borderType=cv2.BORDER_REFLECT) for x in (slope, slope * slope))
    sd_slope = np.sqrt(win_sqr_mean - win_mean * win_mean)
    return sd_slope
