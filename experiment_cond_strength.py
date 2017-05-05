""" Evaluate available methods' power and size.
argv[1] should be the name of the dataset to use (see
experiment_settings.py).

The results are saved to SAVE_DIR/{argv[1]}_results.pkl.
"""
import os
import sys
import time
from collections import defaultdict
import joblib
import numpy as np

# Import all the conditional independence methods we have implemented.
from independence_test.methods import cond_nn
from independence_test.methods import cond_cci
from independence_test.methods import cond_hsic
from independence_test.methods import cond_kcit
from independence_test.methods import cond_rcit
from independence_test.methods import cond_kcipt

# Import all the datasets we have implemented
from independence_test.data import make_chaos_data
from independence_test.data import make_pnl_data
from independence_test.data import make_gmm_data
from independence_test.data import make_discrete_data

# Choose the sample numbers we will iterate over.
SAMPLE_NUMS = [200, 400, 1000, 10000, 100000]

# Set a limit (in seconds) on each trial. Methods that go over
# will be forcefully terminated and will return -1 as p-value.
MAX_TIME = 600

# Number of experimental trials, for each method / dataset setting.
N_TRIALS = 10

# Make a dict of methods.
COND_METHODS = {'nn': cond_nn,
                'cci': cond_cci,
                'rcit': cond_rcit,
                'chsic': cond_hsic,
                'kcit': cond_kcit,
                'kcipt': cond_kcipt}

# Make a dict of the datasets, as well as the values of the dataset 
# 'complexity' parameter we want to consider, and the dataset dimen-
# sionalities we want to consider (see documentation for each data-
# set for permissible complexity and dimensionality values).
DSETS = {'chaos': (make_chaos_data, [.01, .04, .16, .32, .68, .84, .96, .99], [1]),
         'pnl': (make_pnl_data, [0], [1]),
         'discrete': (make_discrete_data, [2, 8, 32], [2, 8, 32]),
         'gmm': (make_gmm_data, [1], [100])}



if __name__ == "__main__":
    dset = sys.argv[1]
    dataset = DSETS[dset]

    SAVE_FNAME = os.path.join('saved_data', '{}_results.pkl'.format(dset))

    try:
        RESULTS = joblib.load(SAVE_FNAME)
    except IOError:
        RESULTS = defaultdict(list)
    
    for trial_id in range(N_TRIALS):
        for n_samples in SAMPLE_NUMS:
            for dim in dataset[2]:
                for param in dataset[1]:
                    xd, yd, zd = dataset[0](type='dep', n_samples=n_samples,
                                            dim=dim, complexity=param, verbose=True)

                    xi, yi, zi = dataset[0](type='indep', n_samples=n_samples,
                                            dim=dim, complexity=param, verbose=True)
                    for method_name in COND_METHODS:
                        key = '{}_{}_{}mt_{}samples_{}dim_{}complexity'.format(
                            method_name, dset, MAX_TIME, n_samples, dim, param)
                        if len(RESULTS[key]) > 0 and RESULTS[key][-1][0] < 0:
                            continue
                        method = COND_METHODS[method_name]
                        print '=' * 70
                        print key
                        print '=' * 70

                        # Run the trials.
                        tic = time.time()
                        pval_d = method.test(xd, yd, zd, max_time=MAX_TIME)
                        #pval_d = 0
                        pval_i = method.test(xi, yi, zi, max_time=MAX_TIME)
                        toc = time.time() - tic
                        print('(trial {}, time {}) p_d {:.4g}, p_i {:.4g}.'.format(
                            trial_id, toc, pval_d, pval_i))

                        # Pickle the results.
                        RESULTS[key].append((pval_d, pval_i, toc))
                        joblib.dump(RESULTS, os.path.join(
                            'independence_test', SAVE_FNAME))
                print RESULTS
