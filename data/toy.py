""" Artificial datasets used in previous work and new.

In each case, `strength` is a parameter that sets the difficulty
of the task. The larger `strength`, the larger and easier to detect
the independence between x and y given z.
"""
import numpy as np
from independence_test.utils import sample_gp, sample_pnl

def _sample_gmm(means, stds, coeffs, n_samples):
    """ Sample from a mixture of Gaussians. """
    mixs = np.random.choice(coeffs.size, n_samples, p=coeffs, replace=True)
    return np.random.randn(n_samples) * stds[mixs] + means[mixs] 


def make_gmm_data(n_samples, type='dep', dim=10, complexity=10, **kwargs):
    """ Z are the means, stds and coefficients of k one-dimensional Gaussian
    mixture components. X and Y are samples from the mixture. """
    x = np.zeros((n_samples, dim))
    y = np.zeros((n_samples, dim))
    z = np.zeros((n_samples, complexity * 3))
    coeffs = np.random.rand(n_samples, complexity)
    coeffs /= coeffs.sum(axis=1, keepdims=True)
    means = np.random.rand(n_samples, complexity)
    stds = np.random.randn(n_samples, complexity) * .5
    z = np.hstack([coeffs, means, stds])
    x = np.vstack([_sample_gmm(means[i], stds[i], coeffs[i], dim) for i in range(n_samples)])
    y = np.vstack([_sample_gmm(means[i], stds[i], coeffs[i], dim) for i in range(n_samples)])
    if type == 'dep':
        x, y, z = x, z, y
    return x, y, z


def make_chaos_data(n_samples, type='dep', complexity=.5, **kwargs):
    """ X and Y follow chaotic dynamics. """
    assert type in ['dep', 'indep']
    n_samples += 1
    x = np.zeros((n_samples, 4))
    y = np.zeros((n_samples, 4))
    x[-1, :] = np.random.randn(4) * .01
    y[-1, :] = np.random.randn(4) * .01
    for step_id in range(n_samples):
        x[step_id, 0] = 1.4 - x[step_id-1, 0]**2 + .3 * x[step_id-1, 1]
        y[step_id, 0] = (1.4 - (complexity * x[step_id-1, 0] * y[step_id-1, 0]
                                + (1 - complexity) * y[step_id-1, 0]**2) +
                         .1 * y[step_id-1, 1])
        x[step_id, 1] = x[step_id-1, 0]
        y[step_id, 1] = y[step_id-1, 0]
    x[:, 2:] = np.random.randn(n_samples, 2) * .5
    y[:, 2:] = np.random.randn(n_samples, 2) * .5
    if type == 'dep':
        return y[1:], x[:-1], np.array(y[:-1, :2])
    else:
        return x[1:], y[:-1], np.array(x[:-1, :2])


def make_pnl_data(n_samples=1000, type='dep', dim=1, complexity=0, **kwargs):
    """ Post-nonlinear model data. `dim` is the dimension of x, y and z.
    `dim` - `complexity` indicates the number of coordinates relevant to
    the dependence. 

    Note: `complexity` must be smaller or equal to `dim`. """

    assert type in ['dep', 'indep']
    assert 0 <= complexity < dim
    complexity = dim - complexity
    e_x = np.random.randn(n_samples, dim)
    e_y = np.random.randn(n_samples, dim)
    z = np.random.rand(n_samples, dim)


    # Make ANM data.
    x = sample_pnl(z[:, :complexity] + e_x, dim)
    y = sample_pnl(z[:, :complexity] + e_y, dim)
    
    if type == 'dep':
        #x, y, z = x, z, y
        e_xy = np.random.randn(n_samples, 1) * .5
        x += e_xy
        y += e_xy

    return x, y, z


def make_discrete_data(n_samples=1000, dim=1, type='dep', complexity=20, **kwargs):
    """ Each row of Z is a (continuous) vector sampled
    from the uniform Dirichlet distribution. Each row of
    X and Y is a (discrete) sample from a multinomial
    distribution in the corresponding row of Z.
    `complexity` indicates the number of multinomial samples in X and Y.
    """
    assert type in ['dep', 'indep']
    z = np.random.dirichlet(alpha=np.ones(dim+1), size=n_samples)
    x = np.vstack([np.random.multinomial(complexity, p) for p in z])[:, :-1].astype(float)
    y = np.vstack([np.random.multinomial(complexity, p) for p in z])[:, :-1].astype(float)
    z = z[:, :-1]
    if type == 'dep':
        x, y, z = x, z, y
    return x, y, z


def make_trivial_data(n_samples=1000, dim=1, type='dep', **kwargs):
    """ Make x = y if type = 'dep', else make x and y uniform random. """
    z = np.random.randn(n_samples, dim)
    if type == 'dep':
        x = np.random.randn(n_samples, dim)
        y = x
    else:
        x = np.random.randn(n_samples, dim)
        y = np.random.randn(n_samples, dim)
    return x, y, z
