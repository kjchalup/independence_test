""" Wrapper for the CHSIC conditional independence test.
You'll need Matlab and the Python-Matlab engine installed.
Then, download this repository https://github.com/garydoranjr/kcipt
first and set its path below. 

Reference:
Kernel Measures of Conditional Dependence,
Fukumizu, Kenji and Gretton, Arthur and Sun, Xiaohai and Scholkopf, Bernhard,
Twenty-First Annual Conference on Neural Information Processing Systems (NIPS 2007).
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
        pval = MATLAB_ENGINE.hsiccondTestIC(
            matlab.double(x.tolist()), matlab.double(y.tolist()),
            matlab.double(z.tolist()), 0.05, 1000.,
            nargout=3, async=True)

        for _ in range(max_time):
            time.sleep(1)
            if pval.done():
                return pval.result()[1]

    except matlab.engine.MatlabExecutionError:
        print('Matlab failure.')
        return -2

    print('Out of time.')
    pval.cancel()
    return -1
