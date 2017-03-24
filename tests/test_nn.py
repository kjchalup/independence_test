import os
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal
sys.path.insert(0, os.path.abspath('..'))
from nn import NN


def test_nn_learns():
    """ Test that the nn decreases training error on trivial data. """
    x = np.linspace(0, 10, 100).reshape(100, 1)
    y = 3 * x
    nn = NN(x_dim=1, y_dim=1)

    nn.fit(x, y, lr=0)
    out_init = nn.predict(x)
    err_init = np.mean((y - out_init)**2)

    # Train a bit.
    nn.fit(x, y)
    out_trained = nn.predict(x)
    err_trained = np.mean((y - out_trained)**2)
    
    assert err_trained < err_init, ('Neural net didnt learn '
                                    'a linear relationship.')

def test_nn_restarts():
    """ Test that restarting the weights works. """
    x = np.linspace(0, 10, 100).reshape(100, 1)
    y = 3 * x
    nn = NN(x_dim=1, y_dim=1)

    nn.fit(x, y, lr=0)
    out_init = nn.predict(x)
    nn.restart()
    out_restarted = nn.predict(x)
    
    assert np.sum(out_init != out_restarted) > 0
    
