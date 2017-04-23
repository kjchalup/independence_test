""" Artificial datasets used in previous work and new. """
import numpy as np
from independence_test.utils import sample_random_fn


def make_gaussian_data(
        n_samples=1000, type='dep', dim=1, complexity=None):
    """ X and Y are both Gaussian. Z is another random vector,
    independent of both X and Y. `complexity` indicates dim(Z)."""
    assert type in ['dep', 'indep']
    complexity = complexity or dim
    if type == 'dep':
        A = np.random.rand(2 * dim, 2 * dim)
        xy = np.random.multivariate_normal(mean=np.zeros(2 * dim),
                                           cov=np.dot(A, A.T), size=n_samples)
        x = xy[:, :dim]
        y = xy[:, dim:]
    else:
        A = np.random.rand(dim, dim)
        B = np.random.rand(dim, dim)
        x = np.random.multivariate_normal(mean=np.zeros(dim),
                                          cov=np.dot(A, A.T), size=n_samples)
        y = np.random.multivariate_normal(mean=np.zeros(dim),
                                          cov=np.dot(B, B.T), size=n_samples)

    C = np.random.rand(complexity, complexity)
    z = np.random.multivariate_normal(
        mean=np.zeros(complexity), cov=np.dot(C, C.T), size=n_samples)
    return x, y, z


def make_chaos_data(n_samples, type='dep', complexity=.5, **kwargs):
    """ X and Y follow chaotic dynamics. Larger complexity = easier task. """
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


def make_pnl_data(n_samples=1000, type='dep', complexity=1, **kwargs):
    """ Post-nonlinear model data. Only one coordinate of z
    is relevant. """
    assert type in ['dep', 'indep']
    e_x = np.random.randn(n_samples, 1) * .1
    e_y = np.random.randn(n_samples, 1) * .1
    z = np.random.rand(n_samples, complexity)

    # Make ANM data.
    fx = sample_random_fn(z.min(), z.max(), 10)
    x = fx(z[:, :1]) + e_x
    fy = sample_random_fn(z.min(), z.max(), 10)
    y = fy(z[:, :1]) + e_y

    # Make postnonlinear data.
    #gx = sample_random_fn(x.min(), x.max(), 10)
    #x = gx(x)
    #gy = sample_random_fn(y.min(), y.max(), 10)
    #y = gy(y)

    if type == 'dep':
        e_xy = np.random.randn(n_samples, 1) * .5
        x += e_xy
        y += e_xy

    return x, y, z


def make_pnl_zfull_data(n_samples=1000, complexity=100, type='dep', **kwargs):
    """ Post-nonlinear data, all coordinates of z are relevant. """
    assert type in ['dep', 'indep']
    e_x = np.random.randn(n_samples, 1) * .1
    e_y = np.random.randn(n_samples, 1) * .1
    z = np.random.rand(n_samples, complexity)

    # Make some random smooth nonlinear functions.
    x = y = 0
    for z_id in range(complexity):
        fx = sample_random_fn(z.min(), z.max(), 10)
        fy = sample_random_fn(z.min(), z.max(), 10)

        # Make postnonlinear data.
        x += fx(z[:, z_id : z_id + 1])
        y += fx(z[:, z_id : z_id + 1])

    x += e_x
    #gx = sample_random_fn(x.min(), x.max(), 10)
    #x = gx(x)
    y += e_y
    #gy = sample_random_fn(y.min(), y.max(), 10)
    #y = gy(y)

    if type == 'dep':
        e_xy = np.random.randn(n_samples, 1) * .5
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
        return x, z, y
    else:
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
