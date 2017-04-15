""" Constants used throughout evaluation scripts. """
from independence_test.methods import indep_nn
from independence_test.methods import indep_rand

SAVE_DIR = 'saved_data'
SAMPLE_NUMS = [10000]
METHODS = {'nn': indep_nn, 'rand': indep_rand}
N_TRIALS = 10
MAX_TIME = 120
