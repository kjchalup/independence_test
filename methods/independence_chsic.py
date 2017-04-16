""" Wrapper for the CHSIC conditional independence test. 
You'll need Matlab and the Python-Matlab engine installed.
Then, download this repository https://github.com/garydoranjr/kcipt
first and set its path below. """
import matlab.engine
import signal

KCIPT_PATH = r'~/projects/kcipt/'
ENG = matlab.engine.start_matlab()
ENG.addpath(ENG.genpath(KCIPT_PATH, nargout=1))

def indep_chsic(x, y, z, num_perm=100, max_time=60, **kwargs):
    """ Run the CHSIC independence test.

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        num_perm: Number of data permutations for bootstrap.
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
        _, pval, _ = ENG.hsiccondTestIC(matlab.double(x.tolist()),
                                        matlab.double(y.tolist()),
                                        matlab.double(z.tolist()),
                                        0.05,
                                        float(num_perm), nargout=3)
    except StopIteration:
        print 'CHSIC timed out!'
    signal.alarm(0) # Cancel the alarm.

    return pval
