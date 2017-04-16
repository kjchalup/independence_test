""" Wrapper for the KCIPT conditional independence test.
You'll need Matlab and the Python-Matlab engine installed.
Then, download this repository https://github.com/garydoranjr/kcipt
first and set its path below. """
import os
import signal
import matlab.engine


KCIPT_PATH = r'~/projects/kcipt/'
ENG = matlab.engine.start_matlab()
ENG.addpath('independence_test/methods/', nargout=1)
ENG.addpath(KCIPT_PATH, nargout=1)
ENG.addpath(os.path.join(KCIPT_PATH, 'gpml-matlab/gpml'), nargout=1)
ENG.addpath(os.path.join(KCIPT_PATH, 'kcipt'), nargout=1)
ENG.addpath(os.path.join(KCIPT_PATH, 'algorithms'), nargout=1)
ENG.addpath(os.path.join(KCIPT_PATH, 'data'), nargout=1)
ENG.addpath(os.path.join(KCIPT_PATH, 'experiments'), nargout=1)


def indep_kcipt(x, y, z, max_time=60, **kwargs):
    """ Run the CHSIC independence test.

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    pval = -1
    def signal_handler(signum, frame):
        raise StopIteration

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(max_time)
    try:
        pval = ENG.kciptwrapper(matlab.double(x.tolist()),
                                matlab.double(y.tolist()),
                                matlab.double(z.tolist()),
                                nargout=1)
    except StopIteration:
        print 'KCIPT timed out!'
    signal.alarm(0) # Cancel the alarm.

    return pval
