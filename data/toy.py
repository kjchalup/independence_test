""" Artificial datasets used in previous work and new. """
import numpy as np
from independence_test.utils import sample_random_fn


def make_gaussian_data(n_samples=1000, type='dep', xdim=1, ydim=1, zdim=1):
    """ X and Y are both Gaussian. Z is another random vector,
    independent of both X and Y. """
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

    C = np.random.rand(zdim, zdim)
    z = np.random.multivariate_normal(mean=np.zeros(zdim), cov=np.dot(C, C.T))
    return x, y, z


def make_chaos_data(n_samples, gamma=.5, type='dep'):
    """ X and Y follow chaotic dynamics. Larger gamma = easier task. """
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


def make_pnl_data(n_samples=1000, zdim=1, type='dep'):
    """ Post-nonlinear model data. Only one coordinate of z
    is relevant. """
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
        z = np.random.rand(n_samples, zdim)

    return x, y, z


def make_pnl_zfull_data(n_samples=1000, zdim=100, type='dep'):
    """ Post-nonlinear data, all coordinates of z are relevant. """
    assert type in ['dep', 'indep']
    e_x = np.random.randn(n_samples, 1)
    e_y = np.random.randn(n_samples, 1)
    z = np.random.randn(n_samples, zdim)

    # Make some random smooth nonlinear functions.
    for z_id in range(zdim):
        fx = sample_random_fn(z.min(), z.max(), 10)
        fy = sample_random_fn(z.min(), z.max(), 10)

        # Make postnonlinear data.
        x += fx(z[:, z_id : z_id + 1])
        y += fx(z[:, z_id : z_id + 1])

    x += e_x
    gx = sample_random_fn(x.min(), x.max(), 10)
    x = gx(x)
    y += e_y
    gy = sample_random_fn(y.min(), y.max(), 10)
    y = gy(y)

    if type == 'dep':
        z = np.random.rand(n_samples, zdim)

    return x, y, z


def make_discrete_data(n_samples=1000, dim=1, type='dep'):
    """ Each row of Z is a (continuous) vector sampled
    from the uniform Dirichlet distribution. Each row of
    X and Y is a (discrete) sample from a multinomial
    distribution in the corresponding row of Z.
    """
    assert type in ['dep', 'indep']
    z = np.random.dirichlet(alpha=np.ones(dim+1), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    z = z[:, :-1]
    if type == 'dep':
        return x, z, y
    else:
        return x, y, z
