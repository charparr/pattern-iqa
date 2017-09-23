import numpy as np


def make_hillshade(array, azimuth, angle_altitude):
    """ Create hillshade (illumination) surface.

    Parameters
    ----------
    array : ndarray
        Array of surface heights i.e. a DEM.
    azimuth : float
        Degrees from 360 North. A value of 90 creates an azimuth of 270 where illumination comes from the West.
    angle_altitude : float
        Altitude of the sun (degrees) above the horizon.

    Returns
    -------
    shaded : ndarray
        normalized (0 to 255) hillshade map.
    """

    print("Generating Hillshade...")
    azimuth = 360.0 - azimuth

    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuth_rad = azimuth * np.pi / 180.
    altitude_rad = angle_altitude * np.pi / 180.

    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(
        (azimuth_rad - np.pi / 2.) - aspect)

    shaded = 255 * (shaded + 1) / 2
    print("Generating Hillshade...Complete.")
    return shaded
