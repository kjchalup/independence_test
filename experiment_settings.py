""" Constants used throughout evaluation scripts. """
from independence_test.methods import cond_nn
from independence_test.methods import cond_cci
from independence_test.methods import cond_hsic
from independence_test.methods import cond_kcit
from independence_test.methods import cond_rcit
from independence_test.methods import cond_kcipt

from independence_test.data import make_chaos_data
from independence_test.data import make_pnl_data
from independence_test.data import make_gmm_data
from independence_test.data import make_discrete_data
from independence_test.data import make_trivial_data

SAVE_DIR = 'saved_data'
#SAMPLE_NUMS = [200, 400, 1000, 10000]
SAMPLE_NUMS = [10000]
COND_METHODS = {'cci': cond_cci,
                'nn': cond_nn,
                'rcit': cond_rcit,
                'chsic': cond_hsic,
                'kcit': cond_kcit,
                'kcipt': cond_kcipt}
COND_METHODS = {'nn': cond_nn}

# DSETS[`name`][1] is the "complexity" parameter, 
# DSETS[`name`][2] is the dimensionality.
DSETS = {'chaos': (make_chaos_data, [.1, .2, .3, .4, .5], [1]),
         'pnl': (make_pnl_data, [0], [1]),
         'discrete': (make_discrete_data, [3], [100]),
         'gmm': (make_gmm_data, [3], [100])}
