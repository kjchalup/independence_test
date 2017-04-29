""" The Conditional Correlation Independence test.

Reference:
Ramsey, Joseph D.,
A Scalable Conditional Independence Test for Nonlinear, Non-Gaussian Data,
arXiv:1401.5031v2, 2014.
"""
import numpy as np
from scipy.stats import norm

def basis_fns(n=0):
    """ Returns the n+1'st-order monomial. If
    dim > 1, returns the sum of thie monomial 
    over all the coordinates.
    """
    return lambda x: np.sum(x ** (n+1), axis=1)


def mad(x):
    """ Return the mean absolute deviation of data in x. """
    median = np.median(x, axis=0)
    diff = np.median(np.abs(median - x), axis=0)
    mad = 1.4826 * np.max(diff) * np.sqrt(x.shape[1]) * ((4./3.) / x.shape[0]) ** (.2)
    return mad


def uniform_kernel(d, l):
    return 1 if d < l else 0


def residuals(x, z=None):
    """ Return the residuals of x given z. See Ramsey, 2014 for details."""
    if z is None:
        return x
    n_samples = x.shape[0]
    residuals = np.zeros_like(x)
    lengthscale = mad(z)
    for i in range(n_samples):
        sum = 0
        weight = 0
        for j in range(n_samples):
            d = z[i] - z[j]
            d = np.sqrt(np.sum(d * d))
            k = uniform_kernel(d, lengthscale)
            sum += k * x[j]
            weight += k
        residuals[i] = x[i] - sum / weight
    return residuals 


def fdr(plist, alpha):
    m = float(len(plist))
    plist = sorted(plist)
    for k in range(0, m):
        if (k+1) / m * alpha >= plist[k]:
            return k


def test(x, y, z, alpha=.05, n_basis=8):
    rx = residuals(x, z)
    ry = residuals(x, y)
    plist = []
    for basis_i in range(n_basis):
        for basis_y in range(n_basis):
            fx = basis_fns(basis_i)(x)
            fy = basis_fns(basis_j)(y)
            cov = np.mean((fx - fx.mean()) * (fy - fy.mean()))
            r = cov / np.sqrt(np.var(fx) * np.var(fy))
            z = .5 * np.log((1 + r) / (1 - r))
            x_prim = (fx - fx.mean()) / np.std(fx)
            y_prim = (fy - fy.mean()) / np.std(fy)
            tau = np.mean(x_prim**2 * y_prim**2)
            plist.append(2 * (1 - np.abs(norm.cdf(tau) * np.sqrt(n_samples) * z)))
    cutoff = fdr(plist, alpha)
    if cutoff >= len(plist):
        return 1
    else:
        return 0
