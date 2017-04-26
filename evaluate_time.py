""" Evaluate available methods w.r.t. their scaling properties. """
import os
import time
from collections import defaultdict
import joblib
from independence_test.data import make_trivial_data
from independence_test.evaluation_settings import SAVE_DIR, SAMPLE_NUMS, METHODS
from independence_test.evaluation_settings import N_TRIALS, MAX_TIME, DSETS

N_SAMPLES = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
DIM = [8, 16, 32, 64, 128, 256, 512, 1024]
SAVE_FNAME = os.path.join('independence_test', SAVE_DIR, 'time_results_new.pkl')
N_ITERS = 10
MAX_TIME = 30

if __name__ == "__main__":
    try:
        RESULTS = joblib.load(SAVE_FNAME)
    except IOError:
        RESULTS = defaultdict(list)

    for n_samples in N_SAMPLES:
        for dim in DIM:
            for iter_id in range(N_ITERS):
                xd, yd, zd = make_trivial_data(type='dep', n_samples=n_samples, dim=dim)
                xi, yi, zi = make_trivial_data(type='indep', n_samples=n_samples, dim=dim)
                for method_name in METHODS:
                    key = '{}_{}_{}'.format(method_name, n_samples, dim)
                    if len(RESULTS[key]) > 0 and RESULTS[key][-1][0] <0:
                        # Once out of time, always out of time.
                        continue
                    method = METHODS[method_name]
                    print '=' * 70
                    print key
                    print '=' * 70
                    tic = time.time()
                    if method_name == 'nn':
                        MAX_TIME = 10
                    else:
                        MAX_TIME = 30
                    pval_d = method.test(xd, yd, zd, max_time=MAX_TIME)
                    pval_i = method.test(xi, yi, zi, max_time=MAX_TIME)
                    toc = time.time() - tic
                    toc /= 2
                    print 'Iter {}. Time = {}s, pval_d {}, pval_i {}.'.format(
                            iter_id, toc, pval_d, pval_i)
                    RESULTS[key].append((pval_d, pval_i, toc))
                    print RESULTS
                joblib.dump(RESULTS, SAVE_FNAME)
