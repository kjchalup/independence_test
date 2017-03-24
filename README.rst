.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*A Deep-learning-based Conditional Independence Test*

Usage
-----
Let *x, y, z* be random variables. Then deciding whether *P(x | y, z) = P(x | z)* 
can be highly non-trivial, especially if the variables are discrete. This code 
implements the Learning Conditional Independence Test (LCIT), described in 
[link to arXiv]. Basic usage is simple:

.. code:: python 
    z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    z = z[:, :-1]
    pval = indep_nn(x, y, z, max_time=30, discrete=(True, False))

Here, we created discrete variables *x* and *y*, dependent through a "common cause"
*z*. The null hypothesis is that *x* is independent of *y* given *z*. Since in this 
case the variables are independent given *z*, pval should be small.

There are many more examples in `example_[abc].py` scripts.

Installation
------------
Use Python 2.7 and an up-to-date `pip`_ for easy installation.
The neural net code uses  `Tensorflow`_. If you can, use a
machine with a GPU. Clone this repository, then install the package
from within the repository directory::
  
  $ pip install .

This will install all the necessary dependencies.

.. _pip: http://www.pip-installer.org/en/latest/
.. _TensorFlow: https://www.tensorflow.org/
