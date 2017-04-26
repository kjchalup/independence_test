""" Wrapper for the Distance Correlation unconditional independence test.
Install the original R package before running this module.
It is available at https://cran.r-project.org/web/packages/energy/.

References: 
Szekely, G.J. and Rizzo, M.L. (2013). The distance correlation t-test of
independence in high dimension. Journal of Multivariate Analysis,
Volume 117, pp. 193-213.

Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007), Measuring and
Testing Dependence by Correlation of Distances, Annals of Statistics,
Vol. 35 No. 6, pp. 2769-2794.

Szekely, G.J. and Rizzo, M.L. (2009), Brownian Distance Covariance,
Annals of Applied Statistics, Vol. 3, No. 4, 1236-1265.
"""
import rpy2.robjects as R
from rpy2.robjects.packages import importr
from independence_test.utils import np2r
energy = importr('energy')

def test(x, y, **kwargs):
    """ Run the Distance Correlation independence test.

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    x = np2r(x)
    y = np2r(y)
    res = energy.dcor_ttest(x, y)
    return res[2][0]
