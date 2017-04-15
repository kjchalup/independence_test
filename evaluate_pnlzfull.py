""" Evaluate available methods on the post-nonlinear (z-full) dataset.

The results are saved to results/pnlzfull_results.pkl.
"""
import os
import joblib
from independence_test.utils import pc_ks
from independence_test.data import make_pnl_zfull_data as make_data
from independence_test.evaluation_settings import SAVE_DIR, SAMPLE_NUMS, METHODS
from independence_test.evaluation_settings import N_TRIALS, MAX_TIME

SAVE_FNAME = 'pnlzfull_results.pkl'

if __name__ == "__main__":
    try:
        RESULTS = joblib.load(SAVE_FNAME)
    except IOError:
        RESULTS = {}

    for n_samples in SAMPLE_NUMS:
        for dim in [1, 10, 100, 1000]:
            xd, yd, zd = make_data(
                type='dep', n_samples=n_samples, zdim=dim)

            xi, yi, zi = make_data(
                type='indep', n_samples=n_samples, zdim=dim)

            for method_name in METHODS:
                # Set up the storage.
                key = '{}_{}mt_{}samples_{}dim'.format(
                    method_name, MAX_TIME, n_samples, dim)
                method = METHODS[method_name]
                print '=' * 70
                print key
                print '=' * 70
                PVAL_D = []
                PVAL_I = []

                # Run the trials.
                for trial_id in range(N_TRIALS):
                    PVAL_D.append(method(xd, yd, zd, max_time=MAX_TIME,
                                         test_type='mean', verbose=False))
                    PVAL_I.append(method(xi, yi, zi, max_time=MAX_TIME,
                                         test_type='mean', verbose=False))
                    print('Trial {}. p_d {:.4g}, p_i {:.4g}.'.format(
                        trial_id, PVAL_D[-1], PVAL_I[-1]))

                # Compute the area under power curve (AUC) and test the
                # independent case p-values for uniformity.
                auc, _ = pc_ks(PVAL_D)
                _, ks = pc_ks(PVAL_I)
                print 'AUC: {}, KS: {}'.format(auc, ks)

                # Pickle the results.
                RESULTS[key] = [auc, ks]
                joblib.dump(RESULTS, os.path.join(
                    'independence_test', SAVE_DIR, SAVE_FNAME))
