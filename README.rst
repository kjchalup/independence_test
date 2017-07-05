.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*A Deep-learning-based Conditional Independence Test.*

Usage
-----
Let *x, y, z* be random variables. Then deciding whether *P(x | y, z) = P(x | z)* 
can be highly non-trivial, especially if the variables are continuous. This package 
implements a simple yet efficient and effective conditional independence test,
described in [link to arXiv]. Basic usage is simple:

.. code:: python 

    from independence_test.methods import cond_nn
    # Generate some data such that x is indpendent of y given z.
    n_samples = 300
    z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    z = z[:, :-1]
    
    # Run the conditional independence test.
    pval = cond_nn.test(x, y, z, max_time=30, discrete=(True, False))

Here, we created discrete variables *x* and *y*, d-separated by a "common cause"
*z*. The null hypothesis is that *x* is independent of *y* given *z*. Since in this 
case the variables are independent given *z*, pval shouldn't be too small. Specifying which 
variables are discrete is optional.

Implemented Methods
-------------------
I have implemented (wrappers for) many related methods for conditional and
unconditional independence tests. You can find references to relevant research
in the appropriate module docstrings.
independence_test.methods.cond_nn

Requirements
------------
To use the nn methods:
    * numpy >= 1.12
    * scikit-learn >= 0.18.1
    * tensorflow >= 1.0.0
    * https://github.com/kjchalup/neural_networks.git

To use Matlab wrappers (CHSIC, KCIT, KCIPT):
    * Matlab 2014a (might work with other versions, not tested)
    * Matlab engine for Python (available with Matlab)
    * The KCIPT package for Matlab (https://github.com/garydoranjr/kcipt/)

To use R wrappers (RCIT, RIT):
    * R 3.4 (might work with other versions, not tested)
    * The RCIT package for R (https://github.com/ericstrobl/RCIT)

.. _pip: http://www.pip-installer.org/en/latest/
.. _independence_nn.py: independence_nn.py
.. _utils.py: utils.py
