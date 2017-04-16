""" Constants used throughout evaluation scripts. """
from independence_test.methods import indep_nn
from independence_test.methods import indep_chsic
from independence_test.methods import indep_kcit

SAVE_DIR = 'saved_data'
SAMPLE_NUMS = [200]
METHODS = {'nn': indep_nn, 'chsic': indep_chsic, 'kcit': indep_kcit}
N_TRIALS = 10
MAX_TIME = 10
