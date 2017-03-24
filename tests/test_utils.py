import os
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal
sys.path.insert(0, os.path.abspath('..'))
from utils import nan_to_zero, equalize_dimensions

@pytest.fixture
def xyz_data():
    n_samples = 10
    x = np.ones((n_samples, 2))
    y = np.ones((n_samples, 4))
    z = np.ones((n_samples, 6))
    return x, y, z

def test_nan_to_zero_nonan_remains_unchanged():
    data = np.ones((10, 10))
    assert_array_equal(data, nan_to_zero(data), 'arrays should be equal.')

def test_nan_to_zero_nan_is_zeroed():
    data = np.ones(10)
    data[3] = np.nan
    correct_res = np.array(data)
    correct_res[3] = 0
    assert_array_equal(correct_res, nan_to_zero(data),
                       'arrays should be equal.')
    
def test_nan_to_zero_allnans_zeroed():
    data = np.nan * np.ones((3, 3))
    assert_array_equal(np.zeros((3, 3)), nan_to_zero(data),
                       'arrays should be equal')

def test_equalize_dim_xsmall(xyz_data):
    x, y, z = xyz_data
    x_new, y_new, z_new = equalize_dimensions(x, y, z)
    assert x_new.shape[1] == 6, 'x dim wrong.'
    assert y_new.shape[1] == 4, 'y dim wrong.'
    assert z_new.shape[1] == 6, 'z dim wrong.'

def test_equalize_dim_ysmall(xyz_data):
    x, y, z = xyz_data
    y_new, x_new, z_new = equalize_dimensions(y, x, z)
    assert x_new.shape[1] == 6, 'x dim wrong.'
    assert y_new.shape[1] == 4, 'y dim wrong.'
    assert z_new.shape[1] == 6, 'z dim wrong.'

def test_equalize_dim_zsmall(xyz_data):
    x, y, z = xyz_data
    z_new, y_new, x_new = equalize_dimensions(z, y, x)
    assert x_new.shape[1] == 6, 'x dim wrong.'
    assert y_new.shape[1] == 4, 'y dim wrong.'
    assert z_new.shape[1] == 6, 'z dim wrong.'
