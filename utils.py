""" Various utility functions. """
import sys
import numpy as np
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF as ecdf
from scipy import integrate
from scipy.spatial.distance import pdist
from scipy.stats import kstest
try:
    import rpy2.robjects as R
except ImportError:
    print("R-wrapper functionality will not be available. Please install rpy2.")

class TimeoutError(Exception):
    """ Exception to throw when a function times out. """
    pass


def np2r(x):
    """ Convert a numpy array to an R matrix.

    Args:
        x (dim0, dim1): A 2d numpy array.

    Returns:
        x_r: An rpy2 object representing an R matrix isometric to x.
    """
    if 'rpy2' not in sys.modules:
        raise ImportError(("rpy2 is not installed.",
                " Cannot convert a numpy array to an R vector."))
    try:
        dim0, dim1 = x.shape
    except IndexError:
        raise IndexError("Only 2d arrays are supported")
    return R.r.matrix(R.FloatVector(x.flatten()), nrow=dim0, ncol=dim1)


def pc_ks(pvals):
    """ Compute the area under power curve and the Kolmogorov-Smirnoff
    test statistic of the hypothesis that pvals come from the uniform
    distribution with support (0, 1).
    """
    if pvals.size == 0:
        return [-1, -1]
    if -1 in pvals or -2 in pvals:
        return [-1, -1]
    pvals = np.sort(pvals)
    cdf = ecdf(pvals)
    auc = 0
    for (pv1, pv2) in zip(pvals[:-1], pvals[1:]):
        auc += integrate.quad(cdf, pv1, pv2)[0]
    auc += integrate.quad(cdf, pvals[-1], 1)[0]
    ks, _ = kstest(pvals, 'uniform')
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
    """ Reduplicate the data along axis 1 to make the
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

_fs = [lambda x: x, lambda x: x**2, lambda x: x**3,
        lambda x: np.tanh(x), lambda x: np.exp(-np.abs(x))]

def sample_pnl(z, dim_out=1, lengthscale=1.):
    f = np.random.choice(_fs)
    return f(z)


def sample_gp(z, dim_out=1, lengthscale=1.):
    """ Sample from a Gaussian Process on the grid z. """
    dists = pdist(z).flatten()
    r = lengthscale * np.median(dists[np.where(dists != 0)[0]])
    n_samples, dim = z.shape
    rbf = lambda x, y: np.exp(-.5 * np.sum(((x - y) * (x - y))**2) / r**2)
    cov = np.array([[rbf(z[i], z[j]) for i in range(n_samples)]
        for j in range(n_samples)])
    samples = np.random.multivariate_normal(mean=np.zeros(n_samples),
        cov=cov, size=dim_out).T
    return samples
