""" Various utility functions. """
import numpy as np
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF as ecdf
from scipy import integrate
from scipy.stats import kstest
import nn


def pc_ks(pvals):
    """ Compute the area under power curve and the Kolmogorov
    p-value of the hypothesis that pvals come from the uniform distro on (0, 1).
    """
    pvals = np.sort(pvals)
    cdf = ecdf(pvals)
    auc[0] = integrate.quad(cdf, 0, 1, points=pvals)
    _, ks = kstest(pvals, 'uniform')
    return auc, ks

def sample_random_fn(xmin=0, xmax=1, npts=10, ymin=0, ymax=1):
    """ Sample a random function defined on the (xmin, xmax) interval.
    
    Args:
        xmin (float): Function's domain's min.
        xmax (float): Function's domain's max.
        npts (int): Number of random points to interpolate in.
        ymin (float): The function's smallest value.
        ymax (float): The function's largest value.
    
    Returns:
        f (interpolation object): A function that can be applied in its domain.
    """
    f_base = np.sort(np.random.choice(
        np.linspace(xmin, xmax, 10000)[1:-1], npts-2))
    f_base = np.concatenate([[xmin], f_base, [xmax]])
    f = interp1d(f_base, np.random.rand(npts) * ymax + ymin, kind='linear')
    return f

def nan_to_zero(data):
    """ Convert all nans to zeros. """
    data[np.isnan(data)] = 0.
    return data

def equalize_dimensions(x, y, z=None):
    """ Reduplicate the data along axis 1 to make their
    dimensionalities similar.

    Args:
        x (n_samples, xdim): Data.
        y (n_samples, ydim): Data.
        z (n_samples, zdim): Data.
    
    Returns
        x, y, z concatenated with their own copies along the 1st 
            axis so that max(xdim, ydim, zdim) is not much bigger
            than min(xdim_new, ydim_new, zdim_new).
    """
    max_dim = max(x.shape[1], y.shape[1], z.shape[1] if z is not None else 0)
    x_duplicates = max_dim / x.shape[1]
    x_new = np.tile(x, [1, x_duplicates])

    y_duplicates = max_dim / y.shape[1]
    y_new = np.tile(y, [1, y_duplicates])

    if z is not None:
        z_duplicates = max_dim / z.shape[1]
        z_new = np.tile(z, [1, z_duplicates])
        return x_new, y_new, z_new
    else:
        return x_new, y_new
    
    
    
