import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hillshade import make_hillshade
from slope import make_slope, make_std_dem, make_std_slope, make_area_ratio
from normalize import make_normalized_array
from similarity_metrics import compute_similarity

dem = im1 = cv2.imread('/home/cparr/workspace/pattern_similarity/test_images/clpx_dem_1024.tif', cv2.IMREAD_UNCHANGED)
im1 = cv2.imread('/home/cparr/workspace/pattern_similarity/test_images/clpx_depth_1024.tif', cv2.IMREAD_UNCHANGED)
normed_depth = make_normalized_array(im1)

# If the image is RGB, convert to single band

if len(im1.shape) > 2:
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

slope = make_normalized_array(make_slope(dem))
sd_slope = make_normalized_array(make_std_slope(make_slope(dem), 5))
sd_dem = make_std_dem(make_normalized_array(dem), 5)
area_ratio = make_area_ratio(slope)
shade0 = make_normalized_array(make_hillshade(dem, 0, 45))
shade90 = make_normalized_array(make_hillshade(dem, 90, 45))
shade180 = make_normalized_array(make_hillshade(dem, 180, 45))
shade270 = make_normalized_array(make_hillshade(dem, 270, 45))
shade45 = make_normalized_array(make_hillshade(dem, 45, 45))
shade135 = make_normalized_array(make_hillshade(dem, 135, 45))
shade225 = make_normalized_array(make_hillshade(dem, 225, 45))
shade315 = make_normalized_array(make_hillshade(dem, 315, 45))

test_surfaces = [slope, sd_slope, sd_dem, area_ratio, shade0, shade90, shade180,
                 shade270, shade45, shade135, shade225, shade315]
ids = ['slope', 'sd_slope', 'sd_dem', 'area_ratio', 'shade0', 'shade90', 'shade180', 'shade270',
       'shade45', 'shade135', 'shade225', 'shade315']
rows_list = []

for surfs in zip(test_surfaces, ids):
    print("Processing... " + surfs[1])
    results = compute_similarity(normed_depth, surfs[0], surfs[1])
    rows_list.append(results)

df = pd.DataFrame(rows_list)
df.set_index('id', inplace=True)
df['MSE Rank'] = df['mse_value'].rank(ascending=True)
df['SSIM Rank'] = df['ssim_value'].rank(ascending=False)
df['CW-SSIM Rank'] = df['cw_ssim_value'].rank(ascending=False)
df['GMSD Rank'] = df['gms_value'].rank(ascending=False)
df['FSIM Rank'] = df['fsim_value'].rank(ascending=False)
df['Avg. Rank'] = (df['MSE Rank'] + df['SSIM Rank'] + df['CW-SSIM Rank'] + df['GMSD Rank'] + df['FSIM Rank'])
df.sort_values(['Avg. Rank'])


fig = plt.figure()

ax = fig.add_subplot(3, 5, 1)
ax.imshow(normed_depth)
ax.set_title('depth')
i = 2

for im, title in zip(test_surfaces, ids):

    ax = fig.add_subplot(3, 5, i)
    ax.imshow(im)
    ax.set_title(title + ' rank: ' + str(df.loc[title]['Avg. Rank']))
    i += 1

fig.show()

