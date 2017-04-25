""" Wrapper for the KCIT conditional independence test.
You'll need Matlab and the Python-Matlab engine installed.
Then, download this repository https://github.com/garydoranjr/kcipt.

Reference:
Kernel-based Conditional Independence Test and Application in Causal Discovery,
Zhang, Kun and Peters, Jonas and Janzing, Dominik and Scholkopf, Bernhard,
arXiv preprint arXiv:1202.3775, 2012.
"""
import time
try:
    import matlab
except ImportError:
    raise ImportError('Install the Matlab engine for Python to run CHSIC.')
from independence_test import MATLAB_ENGINE

def test(x, y, z, max_time=60, **kwargs):
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
    try:
        pval = MATLAB_ENGINE.CInd_test_new_withGP(
            matlab.double(x.tolist()), matlab.double(y.tolist()),
            matlab.double(z.tolist()), 0.05, float(0),
            nargout=5, async=True)

        for _ in range(max_time):
            time.sleep(1)
            if pval.done():
                return pval.result()[2]

    except matlab.engine.MatlabExecutionError:
        print('Matlab failure.')
        return -2

    print('Out of time.')
    pval.cancel()
    return -1
