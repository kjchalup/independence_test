""" Wrapper for the RCIT conditional independence test.
Install the original R package before running this module.
It is available at https://github.com/ericstrobl/RCIT.

Reference:
Strobl, Eric V. and Zhang, Kun and Visweswaran, Shyam,
Approximate Kernel-based Conditional Independence Test for Non-Parametric Causal Discovery,
arXiv preprint arXiv:1202.3775 (2017).
"""
import rpy2.robjects as R
from rpy2.robjects.packages import importr
from independence_test.utils import np2r
importr('RCIT')

def test(x, y, z, **kwargs):
    """ Run the RCIT independence test.

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    x = np2r(x)
    y = np2r(y)
    z = np2r(z)
    res = R.r.RCIT(x, y, z) 
    return res[0][0]
