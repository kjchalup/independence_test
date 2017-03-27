import gc
import experiment_basic
from experiment_basic import make_chaos_data
from independence_nn import indep_nn
from utils import pc_ks
n_trials = 30
max_time = 10
pval_d = []
pval_i = []

for trial_id in range(n_trials):
    # Compute the size of the test at alpha=.05 and alpha=.001
    x, y, z = make_chaos_data(type='dep', n_samples=300)
    pval_d.append(indep_nn(x, y, z, max_time=max_time, verbose=False))
    x, y, z = make_chaos_data(type='indep', n_samples=300)
    pval_i.append(indep_nn(x, y, z, max_time=max_time, verbose=False))
    print('Trial {}. p_d {:.4g}, p_i {:.4g}.'.format(
            trial_id, pval_d[-1], pval_i[-1]))

auc, _ = pc_ks(pval_d)
_, ks = pc_ks(pval_i)

print('AUC: {}, KS: {}'.format(auc, ks))
