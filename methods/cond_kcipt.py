""" Wrapper for the KCIPT conditional independence test.
You'll need Matlab and the Python-Matlab engine installed.
Then, download this repository https://github.com/garydoranjr/kcipt
first and set its path below. 

Reference:
A Permutation-Based Kernel Conditional Independence Test,
Doran, Gary and Muandet, Krikamol and Zhang, Kun and Sch{\"o}lkopf, Bernhard,
Proceedings of the 30th Conference on Uncertainty in Artificial Intelligence (UAI 2014).


"""

import time
try:
    import matlab
except ImportError:
    raise ImportError('Install the Matlab engine for Python to run CHSIC.')
from independence_test import MATLAB_ENGINE

def test(x, y, z, max_time=60, **kwargs):
    """ Run the CHSIC independence test.

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
        pval = MATLAB_ENGINE.kciptwrapper(
            matlab.double(x.tolist()), matlab.double(y.tolist()),
            matlab.double(z.tolist()), nargout=1, async=True)

        for _ in range(max_time):
            time.sleep(1)
            if pval.done():
                return pval.result()

    except matlab.engine.MatlabExecutionError:
        print('Matlab failure.')
        return -2

    print('Out of time.')
    pval.cancel()
    return -1
