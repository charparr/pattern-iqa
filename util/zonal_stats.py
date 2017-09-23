import numpy as np
import pandas as pd
from skimage.measure import label
from skimage.measure import regionprops


def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """
    attributes_list = []
    for i, test_attribute in enumerate(dir(im_props[0])):
        # Attribute should not start with _ and cannot return an array
        # does not yet return tuples
        if test_attribute[:1] != '_' and not \
                isinstance(getattr(im_props[0], test_attribute), np.ndarray):
            attributes_list += [test_attribute]

    attributes_list=['max_intensity', 'mean_intensity', 'min_intensity']
    return attributes_list


def regionprops_to_df(im_props, im_var):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """
    attributes_list = scalar_attributes_list(im_props)
    # Initialise list of lists for parsed data
    parsed_data = []
    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]
        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]
    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=[str(im_var) + '_' + str(a) for a in attributes_list])


def get_zonal_stats(contour, image, im_var):

    label_img = label(contour)
    return regionprops_to_df(regionprops(label_img, image), im_var)
