""" Tensorflow implementation of the Mixture Density Network. """
import sys
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf


def define_nn(x_tf, y_dim, Ws, bs, tied_covar=False):
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

        # Define the MDN outputs as a function of input data.
        self.y_pred = define_nn(self.x_tf, y_dim, self.Ws, self.bs)

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
        """ Close the session to free memory. Note: this 
        will make this neural net unusable. """
        self.sess.close()

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
            print('Warning: scalers are not fitted.')
        y_pred = self.sess.run(self.y_pred, {self.x_tf: x})
        return self.scaler_y.inverse_transform(y_pred)

    def fit(self, x, y, num_epochs=10, batch_size=32, 
            lr=1e-3, max_time=np.inf, verbose=True, **kwargs):
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
                                   self.y_tf: y_tr[batch_ids]})
                self.sess.run(self.train_op_tf,
                              {self.x_tf: x_tr[batch_ids],
                               self.y_tf: y_tr[batch_ids],
                               self.lr_tf: lr})
            tr_loss /= batch_num
            val_loss = self.sess.run(self.loss_tf,
                                     {self.x_tf: x_val,
                                      self.y_tf: y_val})
            tr_losses[epoch_id] = tr_loss
            val_losses[epoch_id] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                model_path = self.saver.save(self.sess, 'saved_data/model')

            tr_time = time.time() - start_time
            if verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. Tr loss '
                                  '{:.4g}, val loss {:.4g}.').format(
                                      epoch_id, int(tr_time), 
                                       tr_loss, val_loss))
                sys.stdout.flush()
        
            if tr_time > max_time:
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
