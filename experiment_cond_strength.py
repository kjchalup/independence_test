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
from independence_test.experiment_settings import SAVE_DIR, SAMPLE_NUMS, COND_METHODS, DSETS
MAX_TIME = 600
N_TRIALS = 10

if __name__ == "__main__":
    dset = sys.argv[1]
    dataset = DSETS[dset]

    SAVE_FNAME = '{}_results.pkl'.format(dset)

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
                        # Set up the storage.
                        key = '{}_{}_{}mt_{}samples_{}dim_{}complexity'.format(
                            method_name, dset, MAX_TIME, n_samples, dim, param)
                        if len(RESULTS[key]) > 0 and RESULTS[key][-1][0] < 0:
                            # Once out of time, always out of time.
                            continue
                        method = COND_METHODS[method_name]
                        print '=' * 70
                        print key
                        print '=' * 70

                        # Run the trials.
                        tic = time.time()
                        #if method_name == 'nn':
                        #    MAX_TIME = 30
                        #else:
                        #    MAX_TIME = 200
                        pval_d = method.test(xd, yd, zd, max_time=MAX_TIME)
                        pval_i = method.test(xi, yi, zi, max_time=MAX_TIME)
                        toc = time.time() - tic
                        print('(trial {}, time {}) p_d {:.4g}, p_i {:.4g}.'.format(
                            trial_id, toc, pval_d, pval_i))

                        # Pickle the results.
                        RESULTS[key].append((pval_d, pval_i, toc))
                        joblib.dump(RESULTS, os.path.join(
                            'independence_test', SAVE_DIR, SAVE_FNAME))
                print RESULTS
