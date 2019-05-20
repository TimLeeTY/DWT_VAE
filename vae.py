import tensorflow as tf
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pickle


def xavier_init(fan_in, fan_out, constant=0.5):
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """
    Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    Adapted code from blogpost: https://jmetzen.github.io/2015-11-27/vae.html
    Credits to: Jan Hendrik Metzen

    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.relu,
                 learning_rate=0.001, batch_size=100, restore=None):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input (noiseless)
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_output"]])

        # Create autoencoder network
        self._create_network()

        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Add ops to save and restore all variables
        self.saver = tf.train.Saver()

        # Launch the session
        self.sess = tf.InteractiveSession()
        if restore is None:
            # Initializing the tensor flow variables
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            # Load tensorflow variables from disk
            self.saver.restore(self.sess, "tmp/model{}.ckpt".format(restore))

    def _create_network(self):
        # Initialize autoencoder network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"],
                                      network_weights["batch_norm_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"],
                                    network_weights["batch_norm_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,  n_hidden_gener_2,
                            n_input, n_output, n_z):
        """ Initialises all weights, biases, offsets and scaling factors for all
        layers as tf.Variables, parameters that will be optimised through training.

        Currently manages 2 fully connected hidden layers only, this can be improved to
        handle arbitrary architectures by looping through network_architecture appropriately
        """
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['batch_norm_recog'] = {
            'm1': tf.Variable(1, dtype=tf.float32),
            'b1': tf.Variable(0, dtype=tf.float32),
            'm2': tf.Variable(1, dtype=tf.float32),
            'b2': tf.Variable(0, dtype=tf.float32)}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_output))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_output], dtype=tf.float32))}
        all_weights['batch_norm_gener'] = {
            'm1': tf.Variable(1, dtype=tf.float32),
            'b1': tf.Variable(0, dtype=tf.float32),
            'm2': tf.Variable(1, dtype=tf.float32),
            'b2': tf.Variable(0, dtype=tf.float32)}
        return all_weights

    def _recognition_network(self, weights, biases, batch_norm):
        """ Probabilistic encoder, maps data point to a Gaussian distribution in latent space
        Currently manages 2 fully connected hidden layers only, this can be improved to
        handle arbitrary architectures by looping through network_architecture appropriately
        Batch normalisation is performed for each hidden layer
        """
        layer_1 = tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])
        layer_1_mean, layer_1_var = \
            tf.reduce_mean(layer_1, axis=1), tf.math.reduce_variance(layer_1, axis=1)
        layer_1 = tf.nn.batch_normalization(tf.transpose(layer_1), layer_1_mean, layer_1_var,
                                            batch_norm['b1'], batch_norm['m1'], 1e-10)
        layer_1 = tf.transpose(self.transfer_fct(layer_1))
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2_mean, layer_2_var = \
            tf.reduce_mean(layer_2, axis=1), tf.math.reduce_variance(layer_2, axis=1)
        layer_2 = tf.nn.batch_normalization(tf.transpose(layer_2), layer_2_mean, layer_2_var,
                                            batch_norm['b2'], batch_norm['m2'], 1e-10)
        layer_2 = tf.transpose(self.transfer_fct(layer_2))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                                biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases, batch_norm):
        """ Probabilistic decoder, maps point in latent space to a reconstructed data-point
        Currently manages 2 fully connected hidden layers only, this can be improved to
        handle arbitrary architectures by looping through network_architecture appropriately
        Batch normalisation is performed for each hidden layer
        """
        layer_1 = tf.add(tf.matmul(self.z, weights['h1']), biases['b1'])
        layer_1_mean, layer_1_var = \
            tf.reduce_mean(layer_1, axis=1), tf.math.reduce_variance(layer_1, axis=1)
        layer_1 = tf.nn.batch_normalization(tf.transpose(layer_1), layer_1_mean, layer_1_var,
                                            batch_norm['b1'], batch_norm['m1'], 1e-10)
        layer_1 = tf.transpose(self.transfer_fct(layer_1))
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2_mean, layer_2_var = \
            tf.reduce_mean(layer_2, axis=1), tf.math.reduce_variance(layer_2, axis=1)
        layer_2 = tf.nn.batch_normalization(tf.transpose(layer_2), layer_2_mean, layer_2_var,
                                            batch_norm['b2'], batch_norm['m2'], 1e-10)
        layer_2 = tf.transpose(self.transfer_fct(layer_2))
        x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        x_reconstr_mean = tf.nn.sigmoid(x_reconstr_mean)
        return x_reconstr_mean

    def _create_loss_optimizer(self, beta=0.1):
        """ Defines the cost functions assuming the data follows a
        multivariate Bernoulli Distribution: p(x) = x'^x * (1-x')^(1-x)
        1) Reconstruction loss term: E(log(p(x|z)))
        2) Latent loss (KL divergence term)
        """
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           axis=1)

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), axis=1)
        self.recon_loss = tf.reduce_mean(reconstr_loss)
        self.latent_loss = tf.reduce_mean(latent_loss)
        self.total_cost = self.recon_loss + beta * self.latent_loss
        self.cost = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_cost,
        }
        # Use ADAM optimiser
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost, x_reconstr_mean = self.sess.run((self.optimizer, self.cost,
                                                    tf.reduce_mean(self.x_reconstr_mean)),
                                                   feed_dict={self.x: X})
        return cost

    def encode(self, X):
        """Encodes data by mapping it into the mean of the
        Gaussian in the latent space.
        """
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})

    def reconstruct(self, x):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x: x})


def train(dataset_iterator, network_architecture, learning_rate=0.001,
          batch_size=50, total_batch=400, training_epochs=5,
          display_step=1, valid=None, preproc=''):
    """ Train VAE on input data over epochs,
    handles preprocessing and normalisation of data
    """

    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    cost_log = {
        'recon_loss': np.zeros(training_epochs),
        'latent_loss': np.zeros(training_epochs),
        'total_loss': np.zeros(training_epochs)
    }
    valid_cost_log = {
        'recon_loss': np.zeros(training_epochs),
        'latent_loss': np.zeros(training_epochs),
        'total_loss': np.zeros(training_epochs)
    }
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = {
            'recon_loss': 0,
            'latent_loss': 0,
            'total_loss': 0
        }
        valid_in = test_plot(vae, valid, epoch, preproc)
        # Loop over all batches
        for iter in range(total_batch):
            # Fit training using batch data
            # Data is sub-sampled to reduce dimension of training data
            batch = dataset_iterator.eval()[:, :16000*4//div:sub_samp]
            # Pre-process input data with relevant method
            batch = pre_process(batch, preproc)
            if np.count_nonzero(np.isnan(batch)) > 0:
                print('nan error, please check data')
                batch[np.isnan(batch)] = 0
            cost = vae.partial_fit(batch)
            print_str = 'Epoch {}-{:.3f}  '.format(epoch, iter)
            for key, value in cost.items():
                print_str += '{},{:.3f}  '.format(key, value)
                # Compute average losses
                avg_cost[key] += value / total_batch
            # Display logs per batch
            print(print_str)
        # Save variables to disk for use later
        save_path = vae.saver.save(vae.sess, 'tmp/model{}.ckpt'.format(preproc))
        print("Model saved in path: {}".format(save_path))
        # Display logs per epoch step
        print("Epoch {} - average cost {:.3f}".format(epoch, avg_cost["total_loss"]))
        valid_cost = vae.partial_fit(valid_in)
        for key, value in avg_cost.items():
            cost_log[key][epoch] = value
            valid_cost_log[key][epoch] = valid_cost[key]
    with open('cots_log.pckl', 'wb') as f:
        pickle.dump(cost_log, f)
    with open('valid_cost_log.pckl', 'wb') as f:
        pickle.dump(valid_cost_log, f)

    return vae


def test_plot(vae, valid, epoch, preproc=''):
    """ Plot a sample from valid dataset and its reconstruction
    for manual validation
    """
    valid_in = valid.eval()[:, :16000*4//div:sub_samp]
    valid_in = pre_process(valid_in, preproc)
    valid_recon = vae.reconstruct(valid_in)
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)
    ax1.plot(valid_in[0])
    ax2.plot(valid_recon[0])
    plt.savefig('plots/epoch{}-{}.pdf'.format(epoch, preproc))
    return(valid_in)


def _parse_input(serialised_input):
    """ Parse serialised input into labelled dataset with NSynth features """
    parsed_output = tf.parse_single_example(serialised_input, nsynth_feature)
    return parsed_output


def filter_func(features):
    """ Filter input data"""
    result = tf.reshape(tf.equal(features['qualities'][2], 1), [])
    return result


def pre_process(batch, preproc='', wavelet='db1'):
    """ Handles pre-processing of input data, using DWT or DFT
    Normalises data to fit into [0, 1] range
    """
    if preproc == 'DWT':
        wt = pywt.Wavelet(wavelet)
        batch = pywt.wavedec(batch, wt, axis=1, level=6)
        batch = np.hstack(batch)[:, :time_steps]
        batch = np.square(batch)
    elif preproc == 'DFT':
        batch = np.absolute(np.fft.fft(batch, axis=1))
    elif preproc == '':
        batch = np.square(batch)
    else:
        print('invalid preproc')
        batch = np.square(batch)
    batch = np.divide(batch, np.amax(batch, axis=1)[:, None]+1e-10)
    return batch


def main():
    global time_steps, wavelet, sub_samp, div, n_samples, nsynth_feature
    time_steps = 16000 * 4
    wavelet = 'db6'
    nsynth_feature = {
        'note': tf.FixedLenFeature([], tf.int64),
        'note_str': tf.FixedLenFeature([], tf.string),
        'instrument': tf.FixedLenFeature([], tf.int64),
        'instrument_str': tf.FixedLenFeature([], tf.string),
        'pitch': tf.FixedLenFeature([], tf.int64),
        'velocity': tf.FixedLenFeature([], tf.int64),
        'sample_rate': tf.FixedLenFeature([], tf.int64),
        'audio': tf.FixedLenFeature([time_steps], tf.float32),
        'qualities': tf.FixedLenFeature([10], tf.int64),
        'qualities_str': tf.VarLenFeature(tf.string),
        'instrument_family': tf.FixedLenFeature([], tf.int64),
        'instrument_family_str': tf.FixedLenFeature([], tf.string),
        'instrument_source': tf.FixedLenFeature([], tf.int64),
        'instrument_source_str': tf.FixedLenFeature([], tf.string),
    }
    sub_samp = 2
    div = 8
    time_steps = time_steps // sub_samp // div
    batch_size = 50
    n_samples = 289205
    nsynth_dataset = tf.data.TFRecordDataset("nsynth-train.tfrecord")
    nsynth_valid = tf.data.TFRecordDataset("nsynth-valid.tfrecord")
    nsynth_dataset = nsynth_dataset.map(_parse_input)
    nsynth_valid = nsynth_valid.map(_parse_input)
    nsynth_dataset = nsynth_dataset.filter(filter_func).shuffle(buffer_size=10000).repeat().batch(batch_size)
    nsynth_valid = nsynth_valid.filter(filter_func).shuffle(buffer_size=1000).repeat().batch(batch_size)

    nsynth_iterator = nsynth_dataset.make_one_shot_iterator().get_next()['audio']
    nsynth_valid_iterator = nsynth_valid.make_one_shot_iterator().get_next()['audio']

    # Define number of nodes in each layer
    vae_architecture = dict(
        n_hidden_recog_1=2000,
        n_hidden_recog_2=1000,
        n_hidden_gener_1=1000,
        n_hidden_gener_2=2000,
        n_input=time_steps,
        n_output=time_steps,
        n_z=64
    )

    vae = train(nsynth_iterator, vae_architecture, training_epochs=40,
                valid=nsynth_valid_iterator, preproc='', batch_size=batch_size)
    return(vae)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()
