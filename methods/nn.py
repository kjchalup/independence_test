""" Neural network routines. """
import sys
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf


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

    def fit(self, x, y, max_epochs=1000, min_epochs=10, batch_size=32,
            lr=1e-3, max_time=np.inf, verbose=False,
            max_nonimprovs=15, **kwargs):
        """ Train the MDN to maximize the data likelihood.

        Args:
            x (n_samples, x_dim): Input data.
            y (n_samples, y_dim): Output data.
            max_epochs (int): Max number of training epochs. Each epoch goes
                through the whole dataset (possibly leaving out
                mod(x.shape[0], batch_size) data).
            min_epochs (int): Do at least this many epochs (even if max_time
                already passed).
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
        tr_losses = np.zeros(max_epochs)
        val_losses = np.zeros(max_epochs)
        best_val = np.inf
        #batch_num = int(np.floor((n_samples - n_val) / float(batch_size)))
        batch_num = int(np.floor(1000 / float(batch_size)))
        if batch_num == 0:
            raise ValueError('Please choose batch_size < 1000!')
        start_time = time.time()
        for epoch_id in range(max_epochs):
            #ids_perm = np.random.permutation(n_samples - n_val)
            tr_loss = 0
            for batch_id in range(batch_num):
                #batch_ids = ids_perm[batch_id * batch_size:
                #                     (batch_id + 1) * batch_size]
                batch_ids = np.random.choice(n_samples-n_val, batch_size, replace=False)
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
                    self.sess, './tmp')

            tr_time = time.time() - start_time
            if verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. Tr loss '
                                  '{:.4g}, val loss {:.4g}, best val {:.4g}.'
                              ).format(epoch_id, int(tr_time),
                                       tr_loss, val_loss, best_val))
                sys.stdout.flush()
            # Finish training if:
            #   1) min_epochs are done, and
            #   2a) either we're out of time, or
            #   2b) there was no validation score
            #       improvement for max_nonimprovs epochs.
            if (epoch_id >= min_epochs and (time.time() - start_time > max_time
                or epoch_id - last_improved > max_nonimprovs)):
                break

        self.saver.restore(self.sess, model_path)
        if verbose:
            print('Trainig done in {} epochs, {}s. Validation loss {:.4g}.'.format(
                epoch_id, tr_time, best_val))
        return tr_losses, val_losses
