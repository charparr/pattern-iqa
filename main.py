from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import zoom
from surface.make_terrain_surfs import make_terrain_tests
from util.normalize import make_normalized_array
from util.process_sim_and_df_results import perform_similarity
from util.snakes import (morph_acwe, checkerboard_level_set)
from util.export_to_csv import export_results, export_not_drift_results
from util.snakes_plot import plot_snakes_as_contours, plot_inverse_snakes_as_contours

start = timer()
dem = zoom(cv2.imread('/home/cparr/workspace/pattern_similarity/test_images'
                  '/clpx_dem_512.tif', cv2.IMREAD_UNCHANGED), 0.5)
im1 = zoom(cv2.imread('/home/cparr/workspace/pattern_similarity/test_images'
                  '/clpx_depth_512.tif', cv2.IMREAD_UNCHANGED), 0.5)
normed_depth = make_normalized_array(im1)

# If the image is RGB, convert to single band

if len(im1.shape) > 2:
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

print(dem.shape)
print(im1.shape)

ids, test_surfaces = make_terrain_tests(dem)
ids.append('normed_depth')
test_surfaces.append(normed_depth)

df, top5 = perform_similarity(normed_depth, test_surfaces, ids)
##
snake_list = []
inv_snake_list = []
for image in top5['test_im']:

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    ls = morph_acwe(image, 30, init_level_set=init_ls, smoothing=2,
                    lambda1=1, lambda2=1,
                    iter_callback=lambda x: evolution.append(np.copy(x)))

    snake_list.append(evolution[-1])
    not_drift_contour = 1 - evolution[-1]
    inv_snake_list.append(not_drift_contour)

top5 = top5.copy()
top5['snakes'] = snake_list
top5['inv_snakes'] = inv_snake_list

# df for ML. each row is a pixel sample.
learning_df = pd.DataFrame(columns=top5.index.values)
for t in top5.index.values:
    learning_df[t] = np.ravel(top5.loc[t]['test_im'])
learning_df['label'] = np.ravel(top5.loc['normed_depth']['snakes'])
learning_df['label'].replace(0, 'veneer or denuded', inplace=True)
learning_df['label'].replace(1, 'drift', inplace=True)
learning_df.head()
learning_df.to_csv('results/pixels_labeled_drift_or_not.csv')

##

# plot_snakes_as_contours(top5)
# plot_inverse_snakes_as_contours(top5)
#
# drift_df = export_results(top5, 'results/poly_drift_results.csv')
# not_drift_df = export_not_drift_results(top5,
#                                         'results/poly_not_drift_results.csv')

# combo = pd.concat([drift_df, not_drift_df])
# combo.set_index(pd.Series([i for i in range(0, combo.shape[0])]), inplace=True)
# combo.to_csv('results/labeled_drift_or_not.csv')

end = timer()
print(str(end-start)[0:6])