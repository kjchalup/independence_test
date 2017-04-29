""" A conditional (and unconditional!) independence test
based on neural network regression. This implementation
uses Tensorflow and sklearn.

Reference:
Chalupka, Krzysztof and Perona, Pietro and Eberhardt, Frederick, 2017.
"""
import sys
import time
import numpy as np
from independence_test.methods import nn
from independence_test.utils import equalize_dimensions

# Define available test statistic functions.
fs = {'min': lambda x, y: np.min(x) - np.min(y), 
      'mean': lambda x, y: np.mean(x) - np.mean(y)}

def mse(y_pred, y):
    """ Compute the mean squared error.

    Args:
        y_pred (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.

    Returns
        mse: The test statistic: mean-squared error on a validation set.
    """
    return np.mean((y - y_pred)**2)


def bootstrap(h0, h1, f, B=10000):
    """ Bootstrap the test statistic.

    Args:
        h0: Iterable of length m.
        h1: Iterable of length n.
        f: Function taking (h0, h1) to a test statistic.
        B (int): Number of bootstrap samples to create.

    Returns:
        t_star (B,): Bootstraped means of the two distributions.
    """
    t_star = np.zeros(B)
    m = len(h0)
    n = len(h1)
    all_h = np.concatenate([h0, h1])
    for b_id in range(B):
        b_data = np.random.choice(all_h, size=m + n, replace=True)
        t_star[b_id] = f(b_data[:m], b_data[m:])
    return t_star


def test(x, y, z=None, num_perm=10, prop_test=.1,
             max_time=60, discrete=(False, False),
             plot_return=False, test_type='min', verbose=False, **kwargs):
    """ The neural net probabilistic independence test.
    See Chalupka, Perona, Eberhardt 2017.

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        num_perm: Number of data permutations to estimate
            the p-value from marginal stats.
        prop_test (int): Proportion of data to evaluate test stat on.
        max_time (float): Time limit for the test (approximate).
        discrete (bool, bool): Whether x or y are discrete.
        plot_return (bool): If True, return statistics useful for plotting.
        test_type (str): Test statistic type, can be 'min', 'mean'.
        verbose (bool): Print out progress messages (or not).
        kwargs: Arguments to pass to the neural net constructor.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    kwargs['verbose'] = True
    # If x xor y is discrete, use the continuous variable as input.
    if discrete[0] and not discrete[1]:
        x, y = y, x

    # Adjust the dimensionalities of x, y, z to be on the same
    # order, by simple data duplication.
    if z is not None:
        x, y, z = equalize_dimensions(x, y, z)
    else:
        x, y = equalize_dimensions(x, y)

    # Use this many datapoints as a test set.
    n_samples = x.shape[0]
    n_test = int(n_samples * prop_test)

    # Attach the conditioning variable to the input.
    if z is not None:
        x_z = np.hstack([x, z])
    else:
        x_z = x

    # Create a neural net that predicts y from x and z.
    clf = nn.NN(x_dim=x_z.shape[1],
                y_dim=y.shape[1], **kwargs)
    kwargs['num_epochs'] = 10000  # Use max_time so this can be large.

    # Get params for D1.
    d1_preds = []
    d1_stats = np.zeros(num_perm)
    tr_losses, _ = clf.fit(x_z[n_test:], y[n_test:],
                           max_time=max_time / float(num_perm * 2), **kwargs)
    y_pred = clf.predict(x_z[:n_test])
    d1_preds.append(y_pred)
    num_epochs = (tr_losses != 0).sum()
    kwargs['num_epochs'] = num_epochs
    stat = mse(y_pred, y[:n_test])
    d1_stats[0] = stat
    if verbose:
        print('D1 statistic, permutation {}: {}'.format(
            0, d1_stats[0]))

    for perm_id in range(1, num_perm):
        clf.restart()
        clf.fit(x_z[n_test:], y[n_test:], **kwargs)
        y_pred = clf.predict(x_z[:n_test])
        d1_preds.append(y_pred)
        d1_stats[perm_id] = mse(y_pred, y[:n_test])
        if verbose:
            print('D1 statistic, permutation {}: {}'.format(
                perm_id, d1_stats[perm_id]))

    # Get params for D0.
    if z is not None:
        clf.close()
        clf = nn.NN(x_dim=z.shape[1],
                    y_dim=y.shape[1], **kwargs)
    d0_preds = []
    d0_stats = np.zeros(num_perm)
    for perm_id in range(num_perm):
        #x_noise = np.random.randn(*x.shape) * np.std(x, axis=0, keepdims=True) * .5
        perm_ids = np.random.choice(np.arange(n_samples), n_samples, replace=False)
        if z is not None:
             #x_z_bootstrap = np.hstack([x + x_noise, z])
            x_z_bootstrap = z#np.hstack([x[perm_ids], z])
        else:
            x_z_bootstrap = x + x_noise
        clf.restart()
        clf.fit(x_z_bootstrap[n_test:], y[n_test:], **kwargs)
        y_pred = clf.predict(x_z_bootstrap[:n_test])
        d0_preds.append(y_pred)
        d0_stats[perm_id] = mse(y_pred, y[:n_test])
        if verbose:
            print('D0 statistic, permutation {}: {}'.format(
                perm_id, d0_stats[perm_id]))

    # Bootstrap the difference in means of the two distributions.
    t_obs = fs[test_type](d0_stats, d1_stats)
    t_star = bootstrap(d0_stats, d1_stats, f=fs[test_type])
    p_value = np.sum(t_star > t_obs) / float(t_star.size)
    clf.close()
    if plot_return:
        return (p_value, x, y, x_z, d1_preds, d0_preds,
                d1_stats, d0_stats, t_obs, t_star, n_test)
    else:
        # Get the p-value.
        return p_value
