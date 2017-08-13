""" Artificial datasets used in previous work and new.

In each case, `strength` is a parameter that sets the difficulty
of the task. The larger `strength`, the larger and easier to detect
the independence between x and y given z.
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from independence_test.utils import sample_pnl, sample_gp


def make_chaos_data(n_samples, type='dep', complexity=.5, **kwargs):
    """ X and Y follow chaotic dynamics. """
    assert type in ['dep', 'indep']
    if n_samples > 10**5+9:
        raise ValueError(
                'For Chaos data, only up to 10^5 samples can be created.')
    n_samples += 1
    x = np.zeros((10**5+10, 4))
    y = np.zeros((10**5+10, 4))
    x[-1, :] = np.random.randn(4) * .01
    y[-1, :] = np.random.randn(4) * .01
    for step_id in range(10**5+10):
        x[step_id, 0] = 1.4 - x[step_id-1, 0]**2 + .3 * x[step_id-1, 1]
        y[step_id, 0] = (1.4 - (complexity * x[step_id-1, 0] * y[step_id-1, 0]
                                + (1 - complexity) * y[step_id-1, 0]**2) +
                         .1 * y[step_id-1, 1])
        x[step_id, 1] = x[step_id-1, 0]
        y[step_id, 1] = y[step_id-1, 0]
    x[:, 2:] = np.random.randn(10**5+10, 2) * .5
    y[:, 2:] = np.random.randn(10**5+10, 2) * .5

    # Choose a random subset of required size.
    sample_ids = np.random.choice(10**5+9, int(n_samples), replace=False)
    if type == 'dep':
        return y[sample_ids+1], x[sample_ids], np.array(y[sample_ids, :2])
    else:
        return x[sample_ids+1], y[sample_ids], np.array(x[sample_ids, :2])


def make_pnl_data(n_samples=1000, type='dep', dim=1, complexity=0, **kwargs):
    """ Post-nonlinear model data. `dim` is the dimension of x, y and z.
    `dim` - `complexity` indicates the number of coordinates relevant to
    the dependence. 

    Note: `complexity` must be smaller or equal to `dim`. """

    assert type in ['dep', 'indep']
    e_x = np.random.randn(n_samples, 1)
    e_y = np.random.randn(n_samples, 1)

    s1 = np.random.randn(dim, dim)
    s1 = np.dot(s1, s1.T)
    z = np.random.multivariate_normal(np.zeros(dim), s1, n_samples)
    scaler = StandardScaler()

    # Make ANM data.
    # Normalize z[:, 0] so it has unit variance: this is to ensure
    # the PNL functions will be sampled over a reasonable domain.
    z[:, :1] = scaler.fit_transform(z[:, :1])
    x = sample_pnl(z[:, :1] + e_x)
    y = sample_pnl(z[:, :1] + e_y)
    
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    
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
    z = np.random.dirichlet(alpha=np.ones(dim), size=n_samples)
    x = np.vstack([np.random.multinomial(complexity-1, p) for p in z]).astype(float)
    y = np.vstack([np.random.multinomial(complexity-1, p) for p in z]).astype(float)
    x = OneHotEncoder(n_values=complexity, sparse=False).fit_transform(x)
    y = OneHotEncoder(n_values=complexity, sparse=False).fit_transform(y)
    if type == 'dep':
        copyvals = np.random.rand(n_samples) < .5
        x[copyvals] = y[copyvals]
    return x, y, z


def make_chain_data(n_samples=1000, dim=1, complexity=1, type='dep', **kwargs):
    """ Make x = y if type = 'dep', else make x and y uniform random. """
    A = np.random.randn(dim, dim)
    B  = np.random.randn(dim, dim)

    if type == 'dep':
        # x -> z <- y.
        z = np.random.randn(n_samples, dim)
        x = np.dot(z, A) + np.random.randn(n_samples, dim)
        y = np.dot(x, B) + np.random.randn(n_samples, dim)
        return x, y, z
    else:
        # x <- z -> y.
        z = np.random.randn(n_samples, dim)
        x = np.dot(z, A) + np.random.randn(n_samples, dim)
        y = np.dot(z, B) + np.random.randn(n_samples, dim)
        return x, y, z
