""" Evaluate available methods w.r.t. their scaling properties. """
import os
import time
import joblib
import numpy as np
from independence_test.data import make_gaussian_data
from independence_test.evaluation_settings import SAVE_DIR, SAMPLE_NUMS, METHODS
from independence_test.evaluation_settings import N_TRIALS, MAX_TIME, DSETS

SAVE_FNAME = 'time_results.pkl'
MAX_TIME = 10
try:
    RESULTS = joblib.load(SAVE_FNAME)
except IOError:
    RESULTS = {}

for n_samples in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
    for dim in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        x, y, z = make_gaussian_data(type='dep', n_samples=n_samples,
                                     dim=dim, complexity=dim)
        for method_name in METHODS:
            key = '{}_{}_{}'.format(method_name, n_samples, dim)
            method = METHODS[method_name]

            print '=' * 70
            print key
            print '=' * 70

            tic = time.time()
            pval = method(x, y, z, max_time=MAX_TIME)
            toc = time.time() - tic

            if pval == -1:
                toc = -1
            if pval == -2:
                toc = -2

            print 'Time = {}s.'.format(toc)
            RESULTS[key] = toc
            joblib.dump(RESULTS, os.path.join('independence_test',
                                              SAVE_DIR, SAVE_FNAME))
