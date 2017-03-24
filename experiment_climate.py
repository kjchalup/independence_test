import os
import numpy as np
import scipy.io as sio
from independence_nn import indep_nn

def test_climate(data_dir='./climate', max_time=60, delay=2):
    wind_x = sio.loadmat(os.path.join(
        data_dir, 'sfc_zonal_wind.mat'))['DataSeries'][:, :10, :10]
    temp = sio.loadmat(os.path.join(
        data_dir, 'sfc_temp.mat'))['DataSeries'][:, :10, :10]
    n_samples, nx, ny = wind_x.shape
    
    x = wind_x[:-delay, :, :].reshape(-1, nx * ny)
    y = wind_x[delay:, :, :].reshape(-1, nx * ny)
    z = np.vstack([wind_x[start_id:start_id+delay-1, :, :].reshape(1, -1)
         for start_id in range(1, n_samples-delay+1)])
    z = z[np.random.permutation(z.shape[0])]
    pval = indep_nn(x, y, z, max_time=max_time, num_perms=2)
    return pval


if __name__ == "__main__":
    test_climate(max_time=240, delay=10)
