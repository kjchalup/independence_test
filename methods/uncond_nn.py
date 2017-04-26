""" The unconditional neural network independence test. """
from independence_test.methods import cond_nn

def test(x, y, **kwargs):
    return cond_nn.test(x, y)
