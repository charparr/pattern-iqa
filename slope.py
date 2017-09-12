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


def make_std_dem(dem, win_size):
    """ Create Standard Deviation slope surface.

    Parameters
    ----------
    dem : ndarray
        Array of surface heights i.e. a DEM.
    win_size : int
        size of moving window to compute std. dev.

    Returns
    -------
    sd_dem : ndarray
        Standard Deviation elevation map.
    """
    win_mean, win_sqr_mean = (cv2.boxFilter(x, -1, (win_size, win_size),
                                            borderType=cv2.BORDER_REFLECT) for x in (dem, dem * dem))
    sd_dem = np.sqrt(win_sqr_mean - win_mean * win_mean)
    return sd_dem


def make_std_slope(slope, win_size):
    """ Create Standard Deviation slope surface.

    Parameters
    ----------
    slope : ndarray
        Array of Slope Values
    win_size : int
        size of moving window to compute std. dev.

    Returns
    -------
    sd_slope : ndarray
        Standard Deviation slope map.
    """
    win_mean, win_sqr_mean = (cv2.boxFilter(x, -1, (win_size, win_size),
                                            borderType=cv2.BORDER_REFLECT) for x in (slope, slope * slope))
    sd_slope = np.sqrt(win_sqr_mean - win_mean * win_mean)
    return sd_slope


def make_area_ratio(slope):
    """ Create Area Ratio Surface.

    Parameters
    ----------
    slope : ndarray
        Array of Slope Values

    Returns
    -------
    area_ratio : ndarray
        Surface roughness given by ratio of surface area to planar area

    References
    -------
    Carlos Henrique Grohmann, Morphometric analysis in geographic information systems: applications of free software
    GRASS and R, Computers & Geosciences, Volume 30, Issue 9, 2004, Pages 1055-1067, ISSN 0098-3004,
    http://dx.doi.org/10.1016/j.cageo.2004.08.002.
    """
    # 1 is the planar area of each cell...
    area_ratio = (1.0 / np.cos(slope)) / 1.0
    return area_ratio
