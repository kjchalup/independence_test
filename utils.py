""" Various utility functions. """
import numpy as np
import nn


def nan_to_zero(data):
    """ Convert all nans to zeros. """
    data[np.isnan(data)] = 0.
    return data

def equalize_dimensions(x, y, z=None):
    """ Reduplicate the data along axis 1 to make their
    dimensionalities similar.

    Args:
        x (n_samples, xdim): Data.
        y (n_samples, ydim): Data.
        z (n_samples, zdim): Data.
    
    Returns
        x, y, z concatenated with their own copies along the 1st 
            axis so that max(xdim, ydim, zdim) is not much bigger
            than min(xdim_new, ydim_new, zdim_new).
    """
    max_dim = max(x.shape[1], y.shape[1], z.shape[1] if z is not None else 0)
    x_duplicates = max_dim / x.shape[1]
    x_new = np.tile(x, [1, x_duplicates])

    y_duplicates = max_dim / y.shape[1]
    y_new = np.tile(y, [1, y_duplicates])

    if z is not None:
        z_duplicates = max_dim / z.shape[1]
        z_new = np.tile(z, [1, z_duplicates])
        return x_new, y_new, z_new
    else:
        return x_new, y_new
    
    
    
