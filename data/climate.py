""" Create a climate dataset. Requires NOAA reanalysis II data
(contact the author for .mat files). """
import os
import numpy as np
import scipy.io as sio

def make_climate_data(data_dir='./climate', delay=2):
    """ Create a climate dataset. X is a vector containing daily wind strengths
    over the globe. Y is the same, shifted `delay` days forwards. Z contains
    wind strengths in-between. The idea is that for a delay large enough, past
    is independent of future given present and surrounding times.
    """
    wind_x = sio.loadmat(os.path.join(
        data_dir, 'sfc_zonal_wind.mat'))['DataSeries']#[:, :10, :10]
    # temp = sio.loadmat(os.path.join(
    #     data_dir, 'sfc_temp.mat'))['DataSeries']#[:, :10, :10]
    n_samples, nx, ny = wind_x.shape

    x = wind_x[:-delay, :, :].reshape(-1, nx * ny)
    y = wind_x[delay:, :, :].reshape(-1, nx * ny)
    z = np.vstack([wind_x[start_id:start_id+delay-1, :, :].reshape(1, -1)
                   for start_id in range(1, n_samples-delay+1)])
    return x, y, z
