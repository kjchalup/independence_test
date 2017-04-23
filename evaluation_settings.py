""" Constants used throughout evaluation scripts. """
from independence_test.methods import indep_nn
from independence_test.methods import indep_chsic
from independence_test.methods import indep_kcit
from independence_test.methods import indep_rcit
from independence_test.methods import indep_kcipt

from independence_test.data import make_chaos_data
from independence_test.data import make_pnl_data
from independence_test.data import make_pnl_zfull_data
from independence_test.data import make_discrete_data
from independence_test.data import make_gaussian_data
from independence_test.data import make_trivial_data

SAVE_DIR = 'saved_data'
SAMPLE_NUMS = [200, 1000]
METHODS = {'nn': indep_nn}
           #'rcit': indep_rcit}
           #'chsic': indep_chsic,
           #'kcit': indep_kcit}
           #'kcipt': indep_kcipt}

# DSETS[`name`][1] is the "complexity" parameter, 
# DSETS[`name`][2] is the dimensionality.
DSETS = {'chaos': (make_chaos_data, [.1, .2, .3, .4, .5], [1]),
         'pnl': (make_pnl_data, [1, 100, 1000], [1]),
         'pnlzfull': (make_pnl_zfull_data, [1, 10, 100, 1000], [1]),
         'discrete': (make_discrete_data, [2, 1024], [2, 1024]),
         'gaussian': (make_gaussian_data, [None], [10]),
         'test': (make_trivial_data, [128], [1, 10, 100, 1000])}

N_TRIALS = 100
MAX_TIME = 10
