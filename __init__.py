""" Set up Matlab's Python API to run the KCIT, KCIPT and CHSIC tests for
comparison purposes:
1) Have Matlab installed.
2) Install Matlab's Python API (it's available with Matlab).
3) Install kcipt, as explained in the README.
4) Go to kcipt/CI_PERM and change the filename rbf.m to rbf_ciperm.m.
5) Change all instances of rbf in .m files in kcipt/CI_PERM to rbf_ciperm.
The last two steps are necessary because kcipt contains two rbf.m
scripts and Matlab's lack of package management makes it hard to deal with
this in a more reasonable way.
"""
KCIPT_PATH = r'~/projects/kcipt'
import os
import tensorflow # Import tensorflow before Matlab to prevent GCC version issue.
try:
    import matlab.engine
    MATLAB_ENGINE = matlab.engine.start_matlab()
    MATLAB_ENGINE.addpath('independence_test/methods/', nargout=1)
    MATLAB_ENGINE.addpath(KCIPT_PATH, nargout=1)
    MATLAB_ENGINE.addpath(os.path.join(KCIPT_PATH, 'gpml-matlab/gpml'), nargout=1)
    MATLAB_ENGINE.addpath(os.path.join(KCIPT_PATH, 'kcipt'), nargout=1)
    MATLAB_ENGINE.addpath(os.path.join(KCIPT_PATH, 'algorithms'), nargout=1)
    MATLAB_ENGINE.addpath(os.path.join(KCIPT_PATH, 'data'), nargout=1)
    MATLAB_ENGINE.addpath(os.path.join(KCIPT_PATH, 'experiments'), nargout=1)
    MATLAB_ENGINE.addpath(os.path.join(KCIPT_PATH, 'CI_PERM'), nargout=1)
except ImportError:
    print(('Matlab engine for Python not available. You will not be able ', 
           'to use the following methods: cond_hcis, cond_kcit, cond_kcipt.'))

