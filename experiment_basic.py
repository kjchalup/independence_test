""" Run the independence test on artificial data, used by other articles. """
import os
from collections import defaultdict
import numpy as np
import scipy as sp
from scipy import io as sio
from scipy.interpolate import interp1d
from independence_nn import indep_nn

MAX_TIME = 30  # indep_nn time limit.
TESTS = {'nn': indep_nn}  # Tests to compare.
RESULTS = defaultdict(list)  # Hold all the results here.

def test_discrete(n_samples=1000, xdim=1, ydim=1, type='dep'):
    assert xdim == ydim, 'x dimensionality must match y dimensionality.'
    assert type in ['dep', 'indep']
    z = np.random.dirichlet(alpha=np.ones(xdim+1), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    z = z[:, :-1]
    for tname in TESTS:
        if type == 'dep':
            pval = TESTS[tname](
                x, z, y, max_time=MAX_TIME, discrete=(True, False))
        else:
            pval = TESTS[tname](
                x, y, z, max_time=MAX_TIME, discrete=(True, True))
        RESULTS[tname + '_discrete_' + type].append(pval)


def test_adversarial():
    x = np.atleast_2d(np.linspace(0, 1, 1000)).T
    y = np.zeros_like(x)
    y[:500] = np.random.randn(500) * np.sqrt(.5)
    y[500:] = np.random.randn(500) * np.sqrt(2)
    return x, y

def test_gaussian(n_samples=1000, xdim=1, ydim=1, type='dep'):
    assert type in ['dep', 'indep']
    if type == 'dep':
        A = np.random.rand(xdim + ydim, xdim + ydim)
        xy = np.random.multivariate_normal(mean=np.zeros(xdim + ydim),
                                           cov=np.dot(A, A.T), size=n_samples)
        x = xy[:, :xdim]
        y = xy[:, xdim:]
    else:
        A = np.random.rand(xdim, xdim)
        B = np.random.rand(ydim, ydim)
        x = np.random.multivariate_normal(mean=np.zeros(xdim),
                                          cov=np.dot(A, A.T), size=n_samples)
        y = np.random.multivariate_normal(mean=np.zeros(xdim),
                                          cov=np.dot(B, B.T), size=n_samples)
    for tname in TESTS:
        pval = TESTS[tname](x, y, max_time=MAX_TIME, discrete=(True, False))
        RESULTS[tname + '_gaussian_' + type].append(pval)


def test_chaos(n_samples=1000, gamma=.5, type='dep'):
    assert type in ['dep', 'indep']
    x = np.zeros((n_samples, 4))
    y = np.zeros((n_samples, 4))
    x[-1, :] = np.random.randn(4) * .01
    y[-1, :] = np.random.randn(4) * .01
    for step_id in range(n_samples):
        x[step_id, 0] = 1.4 - x[step_id-1, 0]**2 + .3 * x[step_id-1, 1]
        y[step_id, 0] = (1.4 - (gamma * x[step_id-1, 0] * y[step_id-1, 0]
                                + (1 - gamma) * y[step_id-1, 0]**2) +
                         .1 * y[step_id-1, 1])
        x[step_id, 1] = x[step_id-1, 0]
        y[step_id, 1] = y[step_id-1, 0]
    x[:, 2:] = np.random.randn(n_samples, 2) * .5
    y[:, 2:] = np.random.randn(n_samples, 2) * .5

    for tname in TESTS:
        if type == 'dep':
            pval = TESTS[tname](
                y[1:], x[:-1], np.array(y[:-1]), max_time=MAX_TIME)
        else:
            pval = TESTS[tname](x[1:], y[:-1], 
                                np.array(x[:-1]), max_time=MAX_TIME)
        RESULTS[tname + '_chaos_' + type].append(pval)


def test_postnonlinear_zfull(n_samples=1000, zdim=100, type='dep'):
    assert type in ['dep', 'indep']
    e_x = np.random.randn(n_samples, 1)
    e_y = np.random.randn(n_samples, 1)
    z = np.random.randn(n_samples, zdim)
    x = np.zeros((n_samples, 1))
    y = np.zeros((n_samples, 1))

    # Make some random smooth nonlinear functions.
    for z_id in range(zdim):
        f_base = np.linspace(z[:, z_id].min(), z[:, z_id].max(), 10)
        fx = interp1d(f_base, np.random.rand(10), kind='cubic')
        fy = interp1d(f_base, np.random.rand(10), kind='cubic')

        # Make postnonlinear data.
        x += fx(z[:, z_id : z_id + 1]) 
        y += fx(z[:, z_id : z_id + 1])

    x += e_x
    gx_base = np.linspace(x.min(), x.max(), 10)
    gx = interp1d(gx_base, np.random.rand(10), kind='cubic')
    x = gx(x)
    y += e_y
    gy_base = np.linspace(y.min(), y.max(), 10)
    gy = interp1d(gy_base, np.random.rand(10), kind='cubic')
    y = gy(y)

    if type == 'dep':
        noise = np.random.randn(n_samples, 1)
        x += noise
        y += noise

    for tname in TESTS:
        pval = TESTS[tname](x, y, z, max_time=MAX_TIME)
        RESULTS[tname + '_pnl_' + type].append(pval)


def test_postnonlinear(n_samples=1000, zdim=100, type='dep'):
    assert type in ['dep', 'indep']
    e_x = np.random.randn(n_samples, 1)
    e_y = np.random.randn(n_samples, 1)
    z = np.random.randn(n_samples, zdim)

    # Make some random smooth nonlinear functions.
    f_base = np.linspace(z.min(), z.max(), 10)
    fx = interp1d(f_base, np.random.rand(10), kind='cubic')
    fy = interp1d(f_base, np.random.rand(10), kind='cubic')

    # Make postnonlinear data.
    x = fx(z[:, :1]) + e_x
    gx_base = np.linspace(x.min(), x.max(), 10)
    gx = interp1d(gx_base, np.random.rand(10), kind='cubic')
    x = gx(x)
    y = fy(z[:, :1]) + e_y
    gy_base = np.linspace(y.min(), y.max(), 10)
    gy = interp1d(gy_base, np.random.rand(10), kind='cubic')
    y = gy(y)

    if type == 'dep':
        noise = np.random.randn(n_samples, 1)
        x += noise
        y += noise

    for tname in TESTS:
        pval = TESTS[tname](x, y, z, max_time=MAX_TIME)
        RESULTS[tname + '_pnl_' + type].append(pval)


if __name__=="__main__":
    test_chaos(n_samples=1000, type='dep')
