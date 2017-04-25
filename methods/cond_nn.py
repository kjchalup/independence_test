""" A conditional (and unconditional!) independence test
based on neural network regression. This implementation
uses Tensorflow and sklearn.

Reference:
Chalupka, Krzysztof and Perona, Pietro and Eberhardt, Frederick, 2017.
"""
import sys
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from independence_test.utils import equalize_dimensions

# Define available test statistic functions.
fs = {'min': lambda x, y: np.min(x) - np.min(y), 
      'mean': lambda x, y: np.mean(x) - np.mean(y)}

def mse(y_pred, y):
    """ Compute the mean squared error.

    Args:
        y_pred (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.

    Returns
        mse: The test statistic: mean-squared error on a validation set.
    """
    return np.mean((y - y_pred)**2)


def bootstrap(h0, h1, f, B=10000):
    """ Bootstrap the test statistic.

    Args:
        h0: Iterable of length m.
        h1: Iterable of length n.
        f: Function taking (h0, h1) to a test statistic.
        B (int): Number of bootstrap samples to create.

    Returns:
        t_star (B,): Bootstraped means of the two distributions.
    """
    t_star = np.zeros(B)
    m = len(h0)
    n = len(h1)
    all_h = np.concatenate([h0, h1])
    for b_id in range(B):
        b_data = np.random.choice(all_h, size=m + n, replace=True)
        t_star[b_id] = f(b_data[:m], b_data[m:])
    return t_star


def test(x, y, z=None, num_perm=10, prop_test=.1,
             max_time=60, discrete=(False, False),
             plot_return=False, test_type='min', verbose=False, **kwargs):
    """ The neural net probabilistic independence test.
    See Chalupka, Perona, Eberhardt 2017.

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        num_perm: Number of data permutations to estimate
            the p-value from marginal stats.
        prop_test (int): Proportion of data to evaluate test stat on.
        max_time (float): Time limit for the test (approximate).
        discrete (bool, bool): Whether x or y are discrete.
        plot_return (bool): If True, return statistics useful for plotting.
        test_type (str): Test statistic type, can be 'min', 'mean'.
        verbose (bool): Print out progress messages (or not).
        kwargs: Arguments to pass to the neural net constructor.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    # If x xor y is discrete, use the continuous variable as input.
    if discrete[0] and not discrete[1]:
        x, y = y, x

    # Adjust the dimensionalities of x, y, z to be on the same
    # order, by simple data duplication.
    if z is not None:
        x, y, z = equalize_dimensions(x, y, z)
    else:
        x, y = equalize_dimensions(x, y)

    # Use this many datapoints as a test set.
    n_samples = x.shape[0]
    n_test = int(n_samples * prop_test)

    # Attach the conditioning variable to the input.
    if z is not None:
        x_z = np.hstack([x, z])
    else:
        x_z = x

    # Create a neural net that predicts y from x and z.
    clf = nn.NN(x_dim=x_z[n_test:].shape[1],
                y_dim=y[n_test:].shape[1], **kwargs)
    kwargs['num_epochs'] = 10000  # Use max_time so this can be large.

    # Get params for D1.
    d1_preds = []
    d1_stats = np.zeros(num_perm)
    tr_losses, _ = clf.fit(x_z[n_test:], y[n_test:],
                           max_time=max_time / float(num_perm * 2), **kwargs)
    y_pred = clf.predict(x_z[:n_test])
    d1_preds.append(y_pred)
    num_epochs = (tr_losses != 0).sum()
    kwargs['num_epochs'] = num_epochs
    stat = mse(y_pred, y[:n_test])
    d1_stats[0] = stat
    if verbose:
        print('D1 statistic, permutation {}: {}'.format(
            0, d1_stats[0]))

    for perm_id in range(1, num_perm):
        clf.restart()
        clf.fit(x_z[n_test:], y[n_test:], **kwargs)
        y_pred = clf.predict(x_z[:n_test])
        d1_preds.append(y_pred)
        d1_stats[perm_id] = mse(y_pred, y[:n_test])
        if verbose:
            print('D1 statistic, permutation {}: {}'.format(
                perm_id, d1_stats[perm_id]))

    # Get params for D0.
    d0_preds = []
    d0_stats = np.zeros(num_perm)
    perm_ids = np.random.choice(np.arange(n_samples), n_samples, replace=False)
    x_noise = np.random.randn(*x.shape) * np.std(x, axis=0, keepdims=True)
    for perm_id in range(num_perm):
        if z is not None:
            x_z_bootstrap = np.hstack([x + x_noise, z])
        else:
            x_z_bootstrap = x[perm_ids]
        clf.restart()
        clf.fit(x_z_bootstrap[n_test:], y[n_test:], **kwargs)
        y_pred = clf.predict(x_z_bootstrap[:n_test])
        d0_preds.append(y_pred)
        d0_stats[perm_id] = mse(y_pred, y[:n_test])
        if verbose:
            print('D0 statistic, permutation {}: {}'.format(
                perm_id, d0_stats[perm_id]))

    # Bootstrap the difference in means of the two distributions.
    t_obs = fs[test_type](d0_stats, d1_stats)
    t_star = bootstrap(d0_stats, d1_stats, f=fs[test_type])
    p_value = np.sum(t_star > t_obs) / float(t_star.size)
    clf.close()
    if plot_return:
        return (p_value, x, y, x_z, d1_preds, d0_preds,
                d1_stats, d0_stats, t_obs, t_star, n_test)
    else:
        # Get the p-value.
        return p_value


def define_nn(x_tf, y_dim, Ws, bs, keep_prob, tied_covar=False):
    """ Define a Neural Network.

    The architecture of the network is deifned by the Ws, list of weight
    matrices. The last matrix must be of shape (?, y_dim). If the number
    of layers is lower than 3, use the sigmoid nonlinearity.
    Otherwise, use the relu.

    Args:
        x_tf (n_samples, x_dim): Input data.
        y_dim: Output data dimensionality.
        Ws: List of weight tensors.
        bs: List of bias tensors.
        keep_prob: Dropout probability of keeping a unit on.

    Returns:
        out: Predicted y.

    Raises:
        ValueError: When the last weight tensor's output is not compatible
            with the input shape.
    """

    out_dim = Ws[-1].get_shape().as_list()[1]
    if out_dim != y_dim:
        raise ValueError('NN output dimension is not '
                         'compatible with input shape.')

    if len(Ws) < 2:
        nonlin = tf.nn.sigmoid
    else:
        nonlin = tf.nn.relu

    out = tf.add(tf.matmul(x_tf, Ws[0]), bs[0])
    for layer_id, (W, b) in enumerate(zip(Ws[1:], bs[1:])):
        with tf.name_scope('hidden{}'.format(layer_id)):
            out = nonlin(out)
            out = tf.nn.dropout(out, keep_prob=keep_prob)
            out = tf.add(tf.matmul(out, W), b)

    return out


class NN(object):
    """ A Neural Net object.

    The interface mimics that of sklearn: first,
    initialize the NN. Fit it with NN.fit(),
    and predict with NN.predict().

    Args:
        x_dim: Input data dimensionality.
        y_dim: Output data dimensionality.
        arch: A list of integers, each corresponding to the number
            of units in the MDN's hidden layer.
    """

    def __init__(self, x_dim, y_dim, arch=[128, 128], **kwargs):
        self.arch = arch
        self.x_tf = tf.placeholder(
            tf.float32, [None, x_dim], name='input_data')
        self.y_tf = tf.placeholder(
            tf.float32, [None, y_dim], name='output_data')
        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')
        self.y_dim = y_dim

        # Initialize the weights.
        self.Ws = [tf.truncated_normal([
            x_dim, arch[0]], stddev=1. / x_dim, dtype=tf.float32)]
        for layer_id in range(len(arch)-1):
            self.Ws.append(tf.truncated_normal(
                [arch[layer_id - 1], arch[layer_id]],
                stddev=1. / arch[layer_id - 1], dtype=tf.float32))
        self.Ws.append(tf.truncated_normal(
            [arch[-1], self.y_dim], stddev=1. / arch[-1], dtype=tf.float32))
        self.Ws = [tf.Variable(W_init) for W_init in self.Ws]

        # Initialize the biases.
        self.bs = [tf.Variable(tf.zeros(num_units), dtype=tf.float32)
                   for num_units in arch]
        self.bs.append(tf.Variable(tf.zeros(self.y_dim), dtype=tf.float32))

        # Initialize dropout keep_prob.
        self.keep_prob = tf.placeholder(tf.float32)

        # Define the MDN outputs as a function of input data.
        self.y_pred = define_nn(self.x_tf, y_dim, self.Ws, self.bs, self.keep_prob)

        # Define the loss function: MSE.
        self.loss_tf = tf.losses.mean_squared_error(self.y_tf, self.y_pred)

        # Define the optimizer.
        self.train_op_tf = tf.train.AdamOptimizer(
            self.lr_tf).minimize(self.loss_tf)

        # Define the data scaler.
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Define the saver object for model persistence.
        self.saver = tf.train.Saver(max_to_keep=1)

        # Define the Tensorflow session, and its initializer op.
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def close(self):
        """ Close the session and reset the graph. Note: this
        will make this neural net unusable. """
        self.sess.close()
        tf.reset_default_graph()

    def restart(self):
        """ Re-initialize the network. """
        self.sess.run(self.init_op)

    def predict(self, x):
        """ Compute the output for given data.

        Args:
            x (n_samples, x_dim): Input data.
                x_dim must agree with that passed to __init__.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        try:
            x = self.scaler_x.transform(x)
        except NotFittedError:
            print 'Warning: scalers are not fitted.'
        y_pred = self.sess.run(self.y_pred, {self.x_tf: x, self.keep_prob: 1.})
        return self.scaler_y.inverse_transform(y_pred)

    def fit(self, x, y, num_epochs=10, batch_size=32,
            lr=1e-3, max_time=np.inf, verbose=False,
            max_nonimprovs=30, **kwargs):
        """ Train the MDN to maximize the data likelihood.

        Args:
            x (n_samples, x_dim): Input data.
            y (n_samples, y_dim): Output data.
            num_epochs (int): Number of training epochs. Each epoch goes through
                the whole dataset (possibly leaving out
                mod(x.shape[0], batch_size) data).
            batch_size (int): Training batch size.
            lr (float): Learning rate.
            max_time (float): Maximum training time, in seconds. Training will
                stop if max_time is up OR num_epochs is reached.
            verbose (bool): Display training progress messages (or not).
            max_nonimprovs (int): Number of epochs allowed without improving
                the validation score before quitting.

        Returns:
            tr_losses (num_epochs,): Training errors (zero-padded).
            val_losses (num_epochs,): Validation errors (zero-padded).
        """
        # Split data into a training and validation set.
        n_samples = x.shape[0]
        n_val = int(n_samples * .1)
        ids_perm = np.random.permutation(n_samples)
        self.valid_ids = ids_perm[:n_val]
        x_val = x[self.valid_ids]
        y_val = y[self.valid_ids]
        x_tr = x[ids_perm[n_val:]]
        y_tr = y[ids_perm[n_val:]]
        x_tr = self.scaler_x.fit_transform(x_tr)
        y_tr = self.scaler_y.fit_transform(y_tr)
        x_val = self.scaler_x.transform(x_val)
        y_val = self.scaler_y.transform(y_val)

        # Train the neural net.
        tr_losses = np.zeros(num_epochs)
        val_losses = np.zeros(num_epochs)
        best_val = np.inf
        batch_num = int(np.floor((n_samples - n_val) / float(batch_size)))
        start_time = time.time()
        for epoch_id in range(num_epochs):
            ids_perm = np.random.permutation(n_samples - n_val)
            tr_loss = 0
            for batch_id in range(batch_num):
                batch_ids = ids_perm[batch_id * batch_size:
                                     (batch_id + 1) * batch_size]
                tr_loss += self.sess.run(
                    self.loss_tf, {self.x_tf: x_tr[batch_ids],
                                   self.y_tf: y_tr[batch_ids],
                                   self.keep_prob: 1.})
                self.sess.run(self.train_op_tf,
                              {self.x_tf: x_tr[batch_ids],
                               self.y_tf: y_tr[batch_ids],
                               self.keep_prob: .5,
                               self.lr_tf: lr})
            tr_loss /= batch_num
            val_loss = self.sess.run(self.loss_tf,
                                     {self.x_tf: x_val,
                                      self.y_tf: y_val,
                                      self.keep_prob: 1.})
            tr_losses[epoch_id] = tr_loss
            val_losses[epoch_id] = val_loss
            if val_loss < best_val:
                last_improved = epoch_id
                best_val = val_loss
                model_path = self.saver.save(
                    self.sess, 'independence_test/saved_data/model')

            tr_time = time.time() - start_time
            if verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. Tr loss '
                                  '{:.4g}, val loss {:.4g}, best val {:.4g}.'
                              ).format(epoch_id, int(tr_time),
                                       tr_loss, val_loss, best_val))
                sys.stdout.flush()

            if tr_time > max_time or epoch_id - last_improved > max_nonimprovs:
                break

        self.saver.restore(self.sess, model_path)
        if verbose:
            print('Trainig done in {} epochs, {}s. Validation loss {:.4g}.'.format(
                epoch_id, tr_time, best_val))
        return tr_losses, val_losses

    def early_stop(self, val_losses, thr=.1):
        """ Check if the validation error decreased recently.

        Args:
            val_losses (n_epochs,): validation errors, ordered
                by epoch.
            thr (int): Overfit threshold.

        Returns:
            stop (bool): True if val_losses hadn't 
                decreased for thr iterations.
        """
        if len(val_losses) < thr:
            return False

        if np.sum(np.diff(val_losses[-thr:]) < 0) == 0:
            return True
        else:
            return False
