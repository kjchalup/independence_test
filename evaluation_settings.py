""" Constants used throughout evaluation scripts. """
from independence_test.methods import indep_nn
from independence_test.methods import indep_chsic
from independence_test.methods import indep_kcit
from independence_test.methods import indep_kcipt

SAVE_DIR = 'saved_data'
SAMPLE_NUMS = [200]
METHODS = {'nn': indep_nn, 
           # 'chsic': indep_chsic,
           # 'kcit': indep_kcit,
           'kcipt': indep_kcipt}
N_TRIALS = 10
MAX_TIME = 20
