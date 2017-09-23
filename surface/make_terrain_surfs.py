from surface.slope import make_slope, make_std_dem, make_std_slope
from surface.slope import make_area_ratio, make_slope_variance, make_arc_tpi, make_profile_curvature
from surface.hillshade import make_hillshade


def make_terrain_tests(dem):

    slope = make_slope(dem)
    slope_var = make_slope_variance(slope, 51)
    sd_slope = make_std_slope(slope, 51)
    sd_dem = make_std_dem(dem, 51)
    area_ratio = make_area_ratio(slope)
    arc_tpi = make_arc_tpi(dem, 51)
    kpr, sd_kpr = make_profile_curvature(dem)
    shade0 = make_hillshade(dem, 0, 45)
    shade90 = make_hillshade(dem, 90, 45)
    shade180 = make_hillshade(dem, 180, 45)
    shade270 = make_hillshade(dem, 270, 45)
    shade45 = make_hillshade(dem, 45, 45)
    shade135 = make_hillshade(dem, 135, 45)
    shade225 = make_hillshade(dem, 225, 45)
    shade315 = make_hillshade(dem, 315, 45)

    test_surfaces = [slope, sd_slope, sd_dem, area_ratio, slope_var, arc_tpi, kpr, sd_kpr,
                     shade0, shade90, shade180, shade270, shade45, shade135, shade225, shade315]

    ids = ['Slope', 'SD of Slope', 'SD of Dem', 'Area Ratio', 'Slope Variance', 'TPI', 'Curvature',
           'SD of Curvature', 'Hillshade 0 Az.', 'Hillshade 90 Az.', 'Hillshade 180 Az.', 'Hillshade 270 Az.',
           'Hillshade 45 Az.', 'Hillshade 135 Az.', 'Hillshade 225 Az.', 'Hillshade 315 Az.']

    return ids, test_surfaces
