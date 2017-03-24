""" NN-based routines for independence testing. """
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from scipy.stats import norm
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture
import nn
from utils import equalize_dimensions

def test_stat(y_pred, y):
    """ Compute the independence test statistic.

    Args:
        y_pred (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.

    Returns
        stat: The test statistic: mean-squared error on a validation set.
    """
    return np.mean((y - y_pred)**2)


def bootstrap(h0, h1, B=10000):
    """ Bootstrap the mean between two distributions.

    Args:
        h0: Iterable of length m.
        h1: Iterable of length n.
        B (int): Number of bootstrap samples to create.

    Returns:
        t_star (B,): Bootstraped means of the two distributions.
    """
    t_star = np.zeros(B)
    m = len(h0)
    n = len(h1)
    all_h = np.concatenate([h0, h1])
    mu0s = np.zeros(B)
    mu1s = np.zeros(B)
    for b_id in range(B):
        b_data = np.random.choice(all_h, size=m + n, replace=True)
        mu0 = b_data[:m].mean()
        mu1 = b_data[m:].mean()
        t_star[b_id] = mu0 - mu1
        mu0s[b_id] = mu0
        mu1s[b_id] = mu1
    return t_star, mu0s, mu1s

def indep_nn(x, y, z=None, num_perm=10, prop_test=.1,
             max_time=60*5, discrete=(False, False), plot=True, **kwargs):
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
        plot (bool): If True, plot the predictions and errors.
        kwargs: Arguments to pass to the MDN constructor.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
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
    clf = nn.NN(x_dim=x_z[n_test:].shape[1],
                y_dim=y[n_test:].shape[1], **kwargs)
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
    stat = test_stat(y_pred, y[:n_test])
    d1_stats[0] = stat
    print('D1 statistic, permutation {}: {}'.format(
        0, d1_stats[0]))

    for perm_id in range(1, num_perm):
        clf.restart()
        clf.fit(x_z[n_test:], y[n_test:], **kwargs)
        y_pred = clf.predict(x_z[:n_test])
        d1_preds.append(y_pred)
        d1_stats[perm_id] = test_stat(y_pred, y[:n_test])
        print('D1 statistic, permutation {}: {}'.format(
            perm_id, d1_stats[perm_id]))

    # Get params for D0.
    d0_preds = []
    d0_stats = np.zeros(num_perm)
    predictions = []
    for perm_id in range(num_perm):
        perm_ids = np.random.choice(np.arange(n_test, n_samples), 
                                    n_samples - n_test, replace=True)
        if z is not None:
            x_z_bootstrap = np.hstack([x[perm_ids], z[n_test:]])
        else:
            x_z_bootstrap = x[perm_ids]
        clf.restart()
        clf.fit(x_z_bootstrap, y[n_test:], **kwargs)
        y_pred = clf.predict(x_z[:n_test])
        d0_preds.append(y_pred)
        d0_stats[perm_id] = test_stat(y_pred, y[:n_test])
        print('D0 statistic, permutation {}: {}'.format(
            perm_id, d0_stats[perm_id]))

    # Bootstrap the two difference in means of the two distributions.
    t_obs = np.mean(d0_stats) - np.mean(d1_stats)
    t_star, mu0s, mu1s = bootstrap(d0_stats, d1_stats)
    p_value = np.sum(t_star > t_obs) / float(t_star.size)
    print('p-value {}.'.format(p_value))
    if plot:
        font = FontProperties()

        plt.figure(figsize=(5.5, 5.5))
        plt.subplot(2, 2, 1)
        plt.plot(x[:, :1], y[:, :1], 'k.')
        sort_ids = np.argsort(x_z[:n_test,0])
        for y_pred_id, y_pred in enumerate(d1_preds):
            plt.plot(x_z[sort_ids, :1], y_pred[sort_ids, :1], '-',
                     color=(0, 1, 0, .4))
        plt.plot(x_z[:n_test, :1], y_pred[:, :1] * np.nan, 'g-', 
                 label='E[y|x,z]')

        for y_pred_id, y_pred in enumerate(d0_preds):
            plt.plot(x_z[sort_ids, :1], y_pred[sort_ids, :1], '-',
                     color=(1, 0, 0, .4))
        plt.plot(x_z[:n_test, :1], y_pred[:, :1] * np.nan, 'r-', 
                 label='E[y|x]')

        plt.legend(loc=0)

        plt.subplot(2, 2, 2)
        plt.hist(t_star, bins=50, normed=True)
        plt.plot(d0_stats, np.zeros_like(d0_stats), 'rx',
                 label=r'mse[y|x]$', clip_on=False, ms=10)
        plt.plot(d1_stats, np.zeros_like(d1_stats), 'gx',
                 label='mse[y|x,z]', clip_on=False, ms=10)
        plt.plot(t_obs, [0], 'k*', ms=10, clip_on=False)
        font.set_weight('heavy')
        plt.text(t_obs-.01, .1, r't', fontproperties=font)
        plt.text(-.3, .4, r'pdf(t$^*$)', fontproperties=font)
        plt.legend(loc=0)
        plt.show()

    # Get the p-value.
    return p_value
