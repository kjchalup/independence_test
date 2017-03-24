""" Establish the causal network between seven visual brain regions
resopnding to natural images, from Gallant's fMRI data. The data
unfortunately is not publicly available, but can be downloaded directly
from here https://crcns.org/data-sets/vc/vim-1 (you'll need 
to sign up first)."""
from collections import defaultdict
import numpy as np
from scipy.io import loadmat
import h5py
from independence_nn import indep_nn
from utils import nan_to_zero

MAX_TIME = 100

def load_activations():
    """ Load the fMRI responses into arrays, sorted
    by brain region.

    Returns:
        region_activations: A dictionary with roi names as keys and
            arrays of their activations as values.
"""
    data = h5py.File('neuro/EstimatedResponses.mat')
    rois = data['roiS1'].value  # roi id for each voxel.
    resps = data['dataTrnS1'].value  # neural activations.
    roi_names = ['other', 'V1', 'V2', 'V3', 'V3A', 'V3B', 'V4', 'LatOcc']
    region_activations = dict([(roi_names[roi_id], resps[:, np.where(
        rois == roi_id)[1]]) for roi_id in map(int, np.unique(rois))])
    # Change NaNs to zeros.
    for key in region_activations:
        region_activations[key] = nan_to_zero(region_activations[key])
    return region_activations


def load_stimuli():
    """ Load the stimuli.

    Returns:
        stimuli (n_stimuli, 128 * 128).
    """
    data = loadmat('neuro/trn_stim.mat')
    ims = data['stimTrn']
    return ims.reshape(ims.shape[0], -1)


def test_im_v1_v2():
    """ Check whether the expected set of dependencies
    holds between the image input, V1 and V2 activations.
    """
    ims = load_stimuli()
    all_regions = load_activations()
    v1 = all_regions['V1']
    v4 = all_regions['V4']
    print(indep_nn(ims, v4, v1, max_time=MAX_TIME))


if __name__=="__main__":
    test_im_v1_v2()
