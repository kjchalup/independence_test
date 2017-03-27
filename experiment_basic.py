""" Run the independence test on artificial data, used by other articles. """
import os
from collections import defaultdict
import numpy as np
import scipy as sp
from scipy import io as sio
from utils import sample_random_fn
from independence_nn import indep_nn

MAX_TIME = 60  # indep_nn time limit.
TESTS = {'nn': indep_nn}  # Tests to compare.
RESULTS = defaultdict(list)  # Hold all the results here.

def make_discrete_data(n_samples=1000, dim=1):
    assert type in ['dep', 'indep']
    z = np.random.dirichlet(alpha=np.ones(dim+1), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    z = z[:, :-1]
    if type == 'dep':
        return x, z, y
    else:
        return x, y, z


def test_discrete(n_samples=1000, type='dep', dim=1):
    make_discrete_data(n_samples, dim, type)
    for tname in TESTS:
        pval = TESTS[tname](x, y, z, max_time=MAX_TIME, discrete=(True, False))
        RESULTS[tname + '_discrete_' + type].append(pval)


def make_gaussian_data(n_samples=1000, type='dep', xdim=1, ydim=1):
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
    z = np.random.randn(n_samples, xdim)
    return x, y, z


def test_gaussian(n_samples=1000, xdim=1, ydim=1, type='dep'):
    x, y, z = make_gaussian_data(n_samples, type, xdim, ydim)
    for tname in TESTS:
        pval = TESTS[tname](x, y, z, max_time=MAX_TIME)
        RESULTS[tname + '_gaussian_' + type].append(pval)


def make_chaos_data(n_samples, gamma=.5, type='dep'):
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
    if type == 'dep':
        return y[1:], x[:-1], np.array(y[:-1])
    else:
        return x[1:], y[:-1], np.array(x[:-1])


def test_chaos(n_samples=1000, gamma=.5, type='dep'):
    x, y, z = make_chaos_data(n_samples, gamma, type)
    for tname in TESTS:
        pval = TESTS[tname](x, y, z, max_time=MAX_TIME)
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


def make_pnl_data(n_samples=1000, zdim=1, type='dep'):
    assert type in ['dep', 'indep']
    e_x = np.random.randn(n_samples, 1) * .1
    e_y = np.random.randn(n_samples, 1) * .1
    z = np.random.rand(n_samples, zdim)

    # Make some random smooth nonlinear functions.
    fx = sample_random_fn(z.min(), z.max(), 10)
    fy = sample_random_fn(z.min(), z.max(), 10)

    # Make postnonlinear data.
    x = fx(z[:, :1]) + e_x
    gx = sample_random_fn(x.min(), x.max(), 10)
    x = gx(x)
    y = fy(z[:, :1]) + e_y
    gy = sample_random_fn(y.min(), y.max(), 10)
    y = gy(y)

    if type == 'dep':
        # noise = np.random.randn(n_samples, 1)
        # x += noise
        # y += noise
        z = np.random.rand(n_samples, zdim)
    return x, y, z


def test_pnl(n_samples=1000, zdim=100, type='dep'):
    x, y, z = make_pnl_data(n_samples, zdim, type)
    for tname in TESTS:
        pval = TESTS[tname](x, y, z, max_time=MAX_TIME)
        RESULTS[tname + '_pnl_' + type].append(pval)


if __name__=="__main__":
    test_pnl(n_samples=1000, type='dep', zdim=1000)
    print(RESULTS)
