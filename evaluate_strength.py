""" Evaluate available methods' power and size.
argv[1] should be the name of the dataset to use (see
evaluation_parameters.py).

The results are saved to SAVE_DIR/{argv[1]}_results.pkl.
"""
import os
import sys
import joblib
import numpy as np
from independence_test.utils import pc_ks
from independence_test.evaluation_settings import SAVE_DIR, SAMPLE_NUMS, METHODS
from independence_test.evaluation_settings import N_TRIALS, MAX_TIME, DSETS


if __name__ == "__main__":
    dset = sys.argv[1]
    dataset = DSETS[dset]

    SAVE_FNAME = '{}_results.pkl'.format(dset)

    try:
        RESULTS = joblib.load(SAVE_FNAME)
    except IOError:
        RESULTS = {}

    for n_samples in SAMPLE_NUMS:
        for dim in dataset[2]:
            for param in dataset[1]:
                xd, yd, zd = dataset[0](type='dep', n_samples=n_samples,
                                        dim=dim, complexity=param)

                xi, yi, zi = dataset[0](type='indep', n_samples=n_samples,
                                        dim=dim, complexity=param)

                for method_name in METHODS:
                    # Set up the storage.
                    key = '{}_{}_{}mt_{}samples_{}dim_{}complexity'.format(
                        method_name, dset, MAX_TIME, n_samples, dim, param)
                    method = METHODS[method_name]
                    print '=' * 70
                    print key
                    print '=' * 70
                    PVAL_D = np.zeros(N_TRIALS)
                    PVAL_I = np.zeros(N_TRIALS)

                    # Run the trials.
                    for trial_id in range(N_TRIALS):
                        PVAL_I[trial_id] = method(xi, yi, zi, max_time=MAX_TIME)
                        PVAL_D[trial_id] = method(xd, yd, zd, max_time=MAX_TIME)
                        print('Trial {}. p_d {:.4g}, p_i {:.4g}.'.format(
                            trial_id, PVAL_D[trial_id], PVAL_I[trial_id]))

                    # Compute the area under power curve (AUC) and test the
                    # independent case p-values for uniformity.
                    auc, _ = pc_ks(PVAL_D[PVAL_D >= 0])
                    _, ks = pc_ks(PVAL_I[PVAL_I >= 0])
                    print 'AUC: {}, KS: {}'.format(auc, ks)

                    # Pickle the results.
                    RESULTS[key] = [auc, ks]
                    joblib.dump(RESULTS, os.path.join(
                        'independence_test', SAVE_DIR, SAVE_FNAME))
