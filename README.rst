.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*A Deep-learning-based Conditional Independence Test.*

Usage
-----
Let *x, y, z* be random variables. Then deciding whether *P(x | y, z) = P(x | z)* 
can be highly non-trivial, especially if the variables are discrete. This code 
implements the Learning Conditional Independence Test (LCIT), described in 
[link to arXiv]. Basic usage is simple:

.. code:: python 

    from independence_nn import indep_nn
    n_samples = 300
    z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    z = z[:, :-1]
    pval = indep_nn(x, y, z, max_time=30, discrete=(True, False))

Here, we created discrete variables *x* and *y*, d-separated by a "common cause"
*z*. The null hypothesis is that *x* is independent of *y* given *z*. Since in this 
case the variables are independent given *z*, pval should be small. Specifying which 
variables are discrete is optional.

There are many more examples in `example_[abc].py` scripts.

Installation
------------
Simply clone this repository -- all the important code is in the
`independence_nn.py`_, `nn.py`_ and `utils.py`_ files, so put the repository
in your path and import indep_nn from independence_nn.
  
Requirements (all installable through `pip`_):
    * numpy >= 1.12
    * scikit-learn >= 0.18.1
    * tensorflow >= 1.0.0

.. _pip: http://www.pip-installer.org/en/latest/
.. _independence_nn.py: independence_nn.py
.. _utils.py: utils.py
.. _nn.py: nn.py
