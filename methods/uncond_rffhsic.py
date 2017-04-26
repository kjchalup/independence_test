""" Wrapper for the (Python) package for approximate HSIC
unconditional independence tests. Before using it, download 
the Kerpy package (https://github.com/oxmlcs/kerpy) and 
set KERPY_PATH below. Based on the results in the reference
below, I chose to use the random Fourier features approximation.

Reference:
    Q. Zhang, S. Filippi, A. Gretton, and D. Sejdinovic,
    Large-Scale Kernel Methods for Independence Testing,
    Statistics and Computing, to appear, 2017
"""
import os
import sys
KERPY_PATH = '~/projects/kerpy/'
sys.path.append(KERPY_PATH)
sys.path.append(os.join(KERPY_PATH, 'independence_testing'))
import kerpy.GaussianKernel
from HSICSpectralTestObject import HSICSpectralTestObject


def test(x, y, **kwargs):
    n_samples = x.shape[0]
    def data_gen(**kwargs):
        return x, y
    test = HSICSpectralTestObject(n_samples, data_gen, 
            kernelX=GaussianKernel(1.), kernelY=GaussianKernel(1.),
            kernelX_use_median=True, kernelY_use_median=True,
            rff=True, num_rfx=50, num_rfy=50, unbiased=False)
    return test.compute_pvalue()
