""" Wrapper for the RCIT conditional independence test.
You'll need R installed for this. See RCIT's github for
installation instructions: https://github.com/ericstrobl/RCIT.
"""
import rpy2.robjects as R
from rpy2.robjects.packages import importr
importr('RCIT')

def indep_rcit(x, y, z, max_time=60, **kwargs):
    """ Run the RCIT independence test.

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y. If execution time exceeds `max_time`,
            return -1. If Matlab fails, return -2.
    """
    n_samples = x.shape[0]
    xdim = x.shape[1]
    ydim = y.shape[1]
    zdim = z.shape[1]
    x = R.r.matrix(R.FloatVector(x.flatten()), nrow=n_samples, ncol=xdim)
    y = R.r.matrix(R.FloatVector(y.flatten()), nrow=n_samples, ncol=ydim)
    z = R.r.matrix(R.FloatVector(z.flatten()), nrow=n_samples, ncol=zdim)
    res = R.r.RCIT(x, y, z) 
    return res[0][0]
