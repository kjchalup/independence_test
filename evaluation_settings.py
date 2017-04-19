""" Constants used throughout evaluation scripts. """
from independence_test.methods import indep_nn
from independence_test.methods import indep_chsic
from independence_test.methods import indep_kcit
from independence_test.methods import indep_kcipt

from independence_test.data import make_chaos_data
from independence_test.data import make_pnl_data
from independence_test.data import make_pnl_zfull_data
from independence_test.data import make_discrete_data
from independence_test.data import make_gaussian_data

SAVE_DIR = 'saved_data'
SAMPLE_NUMS = [200, 400]
METHODS = {'nn': indep_nn,
           'chsic': indep_chsic,
           'kcit': indep_kcit}
           #'kcipt': indep_kcipt}

DSETS = {'chaos': (make_chaos_data, [.1, .3, .5], [1]),
         'pnl': (make_pnl_data, [1, 10, 100, 1000], [1]),
         'pnlzfull': (make_pnl_zfull_data, [1, 10, 100, 1000], [1]),
         'discrete': (make_discrete_data, [1, 10, 100, 1000], [1]),
         'gaussian': (make_gaussian_data, [100, 1000], [100, 1000])}

N_TRIALS = 10
MAX_TIME = 60
