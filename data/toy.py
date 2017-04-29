""" Artificial datasets used in previous work and new.

In each case, `strength` is a parameter that sets the difficulty
of the task. The larger `strength`, the larger and easier to detect
the independence between x and y given z.
"""
import numpy as np
from independence_test.utils import sample_gp


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
    e_x = np.random.randn(n_samples, dim) * .1
    e_y = np.random.randn(n_samples, dim) * .1
    z = np.random.rand(n_samples, dim)


    # Make ANM data.
    x = sample_gp(z[:, :complexity], dim, lengthscale=.1) + e_x
    y = sample_gp(z[:, :complexity], dim, lengthscale=.1) + e_y

    if type == 'dep':
        e_xy = np.random.randn(n_samples, 1) * .1
        x += e_xy
        y += e_xy

    return x, y, z


def make_discrete_data(n_samples=1000, dim=1, type='dep', complexity=20):
    """ Each row of Z is a (continuous) vector sampled
    from the uniform Dirichlet distribution. Each row of
    X and Y is a (discrete) sample from a multinomial
    distribution in the corresponding row of Z.
    `complexity` indicates the number of multinomial samples in X and Y.
    """
    assert type in ['dep', 'indep']
    z = np.random.dirichlet(alpha=np.ones(dim+1), size=n_samples)
    x = np.vstack([np.random.multinomial(complexity, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(complexity, p) for p in z])[:, :-1]
    z = z[:, :-1]
    if type == 'dep':
        e_xy = np.random.randn(n_samples, dim) * x.std(axis=0, keep_dims=True)
        x += e_xy
        y += e_xy

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
