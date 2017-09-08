import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hillshade import make_hillshade
from slope import make_slope, make_std_slope
from similarity_metrics import compute_similarity

dem = im1 = cv2.imread('/home/cparr/workspace/pattern_similarity/test_images/clpx_dem_1024.tif', cv2.IMREAD_UNCHANGED)
im1 = cv2.imread('/home/cparr/workspace/pattern_similarity/test_images/clpx_depth_1024.tif', cv2.IMREAD_UNCHANGED)
normalized_im1 = (im1 - np.min(im1)) / (np.max(im1) - np.min(im1))

# If the image is RGB, convert to single band

if len(im1.shape) > 2:
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

slope = make_slope(dem)
sd_slope = make_std_slope(slope, 5)
im2 = make_hillshade(dem, 0, 30)
im3 = make_hillshade(dem, 90, 30)
im4 = make_hillshade(dem, 180, 30)
im5 = make_hillshade(dem, 360, 30)

test_surfaces = [dem, slope, sd_slope, im2, im3, im4, im5]
ids = ['dem', 'slope', 'sd_slope', 'shade0', 'shade90', 'shade180', 'shade360']
rows_list = []

for surfs in zip(test_surfaces, ids):
    normalized_surf = (surfs[0] - np.min(surfs[0])) / (np.max(surfs[0]) - np.min(surfs[0]))
    results = compute_similarity(normalized_im1, normalized_surf, surfs[1])
    rows_list.append(results)

df = pd.DataFrame(rows_list)
df.set_index('id', inplace=True)

print(df['cw_ssim_value'])
print(df['fsim_value'])

