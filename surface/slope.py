import numpy as np
import cv2
from scipy.ndimage import filters


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
    slope = np.arctan(np.sqrt(x * x + y * y))
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
    sd_dem = (filters.uniform_filter(dem, win_size) - dem)/(
        filters.maximum_filter(dem, win_size) - filters.minimum_filter(dem, win_size)
    )

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
    sd_slope = (filters.uniform_filter(slope, win_size) - slope)/(
        filters.maximum_filter(slope, win_size) - filters.minimum_filter(slope, win_size))

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
    area_ratio = (1.0 / np.cos(np.rad2deg(slope)))
    return area_ratio


def make_slope_variance(slope, win_size):
    """ Create Slope Variance Surface

    Parameters
    ----------
    slope : ndarray
        Array of Slope Values
    win_size : int
        size of moving window to compute std. dev.

    Returns
    -------
    slope_var : ndarray
        Variance slope map.
    """
    slope_var = filters.maximum_filter(slope, win_size) - filters.minimum_filter(slope, win_size)
    return slope_var


def make_arc_tpi(dem, win_size):
    """ Create an ArcGIS-like Terrain Prominence Index.

    Parameters
    ----------

    dem : ndarray
        Array of surface heights i.e. a DEM.
    win_size : int
        size of moving window to compute std. dev.

    Returns
    -------
    arc_tpi : ndarray
        Terrain Prominence map.
    """

    arc_tpi = (filters.uniform_filter(dem, win_size) - filters.minimum_filter(dem, win_size))/(
        filters.maximum_filter(dem, win_size) - filters.minimum_filter(dem, win_size)
    )
    return arc_tpi


def make_profile_curvature(dem):

    dx, dy = np.gradient(dem)
    d2x, d2y = np.gradient(dem, 2)
    p = dx ** 2 + dy ** 2
    q = 1 + p
    denom = p*q**1.5

    numer = d2x*(dx**2) + 2 * d2x*d2y + d2y*(dy**2)
    kpr = numer / denom

    sd_kpr = (filters.uniform_filter(kpr, 51) - kpr)/(
        filters.maximum_filter(kpr, 51) - filters.minimum_filter(kpr, 51))
    return kpr, sd_kpr

