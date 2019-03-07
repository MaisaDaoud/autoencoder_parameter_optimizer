import numpy as np
import os
import tensorflow as tf
from six.moves import xrange
from sklearn.cluster import KMeans
from sklearn import preprocessing



class Autoencoder(object):

    def __init__(self, sess, epochs=1, run=1, learning_rate=0.001, batch_size=100, n_layers=2, rep_length=250,
                 checkpoint_dir="checkpoint",
                 training_data="data_train",testing_data="data_test", test=False,
                 train=True, dataset_name="READ", dataset_dir="READ"):

        """
        :param sess: session
        :param epochs: int, num training epohcs
        :param run: 1
        :param learning_rate: float
        :param batch_size:int
        :param n_layers: :int, number of layers in the encoder side
        :param checkpoint_dir: the name of the checkpoint directory
        :param training_data:list, training_data
        :param cancer_training_size: int, the size of training cancer data
        :param cacner_testing_size: int, the size of testing cancer data
        :param testing_data: int
        :param train: boolean
        :param dataset_name: the name of the dataset file
        :param generate: boolean, True to generate synthetic representations
        :param s_number: int, number of samples to generateF


        """
        self.test = test
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.run = run
        self.sess = sess
        self.testing_data = preprocessing.scale(testing_data)
        self.training_data = preprocessing.scale(training_data)
        if len(self.testing_data.shape) == 1:
            self.input_length = len(self.testing_data)
        else:
            self.input_length = self.testing_data.shape[1]

        self.output_length = self.input_length
        self.train = train
        if self.train:
            self.training_size = len(self.training_data)  # training_size
        self.test = test
        self.rep_length = rep_length
        # define the layers
        n_clusters = self.n_layers
        self.mu, self.sigma = self.clustering(n_clusters)
        self.weights, self.baiases = self.initialize_layers(self.n_layers, self.input_length, self.output_length)
        print("$$$$$$$$$$ System Parameters $$$$$$$$$$$$$$$$$$")
        print("[*] n_RBF_layers", self.n_layers)
        print("[*] n_epochs", self.epochs)
        print("[*] learning_rate", self.learning_rate)
        print("[*] input_length", self.input_length)
        print("[*] training  data size=", self.training_size)
        print("[*] testing data shape is ", len(self.testing_data.shape))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.build_model()

    def clustering(self, num_clusters):
        mu = {}
        sigma = {}
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.training_data)
        sigma_0 = {i: np.std(self.training_data[np.where(kmeans.labels_ == i)], axis=0) for i in
                   range(kmeans.n_clusters)}
        for key in sigma_0.keys():
            for i in range(len(sigma_0[key])):
                if sigma_0[key][i] == 0:
                    sigma_0[key][i] = 1
        for l in range(0, num_clusters):
            mu['cluster_' + str(l+1)] = tf.constant(kmeans.cluster_centers_[l], tf.float32,
                                                      [1, self.input_length])
            sigma['cluster_' + str(l + 1)] = tf.constant(sigma_0[l],
                                                         tf.float32, [1, self.input_length])
        return mu, sigma

    def initialize_layers(self, n_layers, input_length, output_length):
        weights = {}
        biases = {}
        for l in range(0, n_layers):
            weights['encoder_h' + str(l + 1)] = tf.Variable(
                tf.random_normal([input_length, self.rep_length]) * tf.sqrt(
                    2.0 / self.rep_length))  # tf.random_normal([n_layers[l], n_layers[l + 1]]))
            biases['encoder_b' + str(l + 1)] = tf.Variable(tf.random_normal([self.rep_length]))
        weights['decoder_h' + str(1)] = tf.Variable(
            tf.random_normal([self.rep_length, output_length]) * tf.sqrt(
                2.0 / self.rep_length))  # tf.random_normal([n_layers[0], output_length]))
        biases['decoder_b' + str(1)] = tf.Variable(tf.random_normal([output_length]))

        return weights, biases

    def build_model(self):
        """ build the models"""
        self.saver = tf.train.Saver()
        if (self.train):

            self.training()
        else:

            self.high_dim_outputs, self.rep = self.testing(self.testing_data, "testing_reps")

    def training(self):
        """ train the model"""
        X = tf.placeholder('float', [None, self.input_length])
        self.e = self.encoder(X)
        self.d = self.decoder(self.e)
        beta = 0.000001
        regularizers = 0.0  # tf.nn.l2_loss(self.weights['encoder_h1'])
        for l in range(0, self.n_layers):
            regularizers = regularizers + tf.nn.l2_loss(self.weights['encoder_h' + str(l + 1)])

        cost = tf.reduce_sum(tf.pow(tf.reduce_mean(tf.pow(self.d - X, 2)), 0.5) + beta * regularizers)

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            cost)

        # Initializing the variables
        try:
            tf.global_variables_initializer().run(session=self.sess)
        except:
            tf.initialize_all_variables().run(session=self.sess)

        counter = 1
        display_step = 10
        # start_time = time.time()
        could_load = self.load(self.checkpoint_dir, "training", self.run)

        if could_load:

         print(" [*] Load SUCESS")
        else:

         print(" [!] Load failed...")

        for epoch in xrange(self.epochs):

            # number of patches
            if self.training_size < self.batch_size:
                self.batch_size = self.training_size
                batch_idxs = 1
            else:
                batch_idxs = min(len(self.training_data), self.training_size) // self.batch_size

            for idx in xrange(batch_idxs):

                batch_samples = self.training_data[
                                idx * self.batch_size:(idx + 1) * self.batch_size]
                _, d_print, c, x_print, e_print, weights_1 = self.sess.run(
                    [optimizer, self.d, cost, X, self.e, self.weights['encoder_h1']], feed_dict={
                        X: batch_samples})  # self.mask_noise(batch_samples, int(self.input_length * 0.1))})  # ,int(self.input_length * 0.1)

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

            if np.mod(counter, 5) == 0:
                self.save(self.checkpoint_dir, counter, self.run)
            counter += 1

        reps_dir = "Representations/"+self.dataset_dir+"/" + self.dataset_name + "/run_" + str(self.run) + "/Autoencoder"
        file_dir = reps_dir
        if not os.path.exists(file_dir):
            os.makedirs(reps_dir)
        high_dim_outputs, reps = self.testing(self.training_data,
                                              self.dataset_dir)  # self.sess.run([self.d, self.e], feed_dict={X: np.reshape(self.training_data, [len(self.training_data), self.input_length])})
        np.savetxt(os.path.join(file_dir, "{}_{}".format("data_train", "reps.csv")), reps, delimiter=",")
        np.savetxt(os.path.join(file_dir, "{}_{}".format("data_train", "high_dim_outputs.csv")), high_dim_outputs,
                   delimiter=",")
        np.savetxt(os.path.join(file_dir, "{}_{}".format("weighs_first", "layer.csv")), weights_1,
                   delimiter=",")
        high_dim_outputs, reps = self.testing(self.testing_data,
                                              self.dataset_dir)  # self.sess.run([self.d, self.e], feed_dict={X: np.reshape(self.testing_data, [len(self.testing_data), self.input_length]),})
        np.savetxt(os.path.join(file_dir, "{}_{}".format("data_test", "reps.csv")), reps,
                   delimiter=",")
        np.savetxt(os.path.join(file_dir, "{}_{}".format("data_test", "high_dim_outputs.csv")), high_dim_outputs,
                   delimiter=",")

    def encoder(self, x, train=True):
        """ encoding """
        layer = 0
        activation_sum = 0

        for l in xrange(0, self.n_layers):

            name_mu = "cluster_" + str(l + 1)
            activation_sum += tf.exp(-1 * tf.pow(x - self.mu[name_mu], 2) / (2 * tf.pow(self.sigma[name_mu],
                                                                                        2)))  # tf.exp(-1 * tf.pow(x - self.mu[name_mu],2)/(2*tf.                                                  pow(self.sigma[name_mu],2)))#                               self.a / tf.pow((tf.pow(x - self.mu[name_mu], 2)                               + tf.pow(self.a, 2)), 0.5) #tf.exp(-1 * tf.pow(x                               - self.mu[name_mu],2)/(2*tf.pow(self.sigma[name_                               mu],2)))

        for l in xrange(0, self.n_layers):

            name_w = "encoder_h" + str(l + 1)
            name_mu = "cluster_" + str(l + 1)
            # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
            this_layer = tf.exp(-1 * tf.pow(x - self.mu[name_mu], 2) / (2 * tf.pow(self.sigma[name_mu],
                                                                                   2))) / activation_sum
            this_layer = tf.matmul(this_layer, self.weights[name_w])
            layer += this_layer

        return layer

    def decoder(self, x, train=True):
        """ decoding """
        layer = x

        for l in xrange(0, 1):

            name_w = "decoder_h" + str(l + 1)
            name_b = "decoder_b" + str(l + 1)

            if train:
                mmm = 1

                batch_mean, batch_var = tf.nn.moments(layer, [0])
                scale = tf.Variable(tf.ones([self.weights[name_w].shape[0]]))
                beta = tf.Variable(tf.zeros([self.weights[name_w].shape[0]]))
                epsilon = 1e-3
                layer = tf.nn.batch_normalization(layer, batch_mean, batch_var, beta, scale, epsilon)

        name_w = "decoder_h" + str(1)
        nanme_b = "decoderr_b" + str(1)
        layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights[name_w]), self.baiases[
            name_b]))  # tf.nn.relu(tf.add(tf.matmul(layer, self.weights[name_w]), self.baiases[name_b]))

        return layer

    def testing(self, data, name):
        """
        generate testing representations, the file is written at
        Representations/dataset_name/Autoencoder/data_test_reps.csv

        representations for training data is generated at
        Representations/dataset_name/Autoencoder/data_train_reps.csv

        """
        X = tf.placeholder('float', [None, self.input_length])
        self.e = self.encoder(X, False)
        self.d = self.decoder(self.e, False)
        cost = tf.pow(tf.reduce_mean(tf.pow(self.d - X, 2)), 0.5)
        could_load = self.load(self.checkpoint_dir, "testing", self.run)
        if could_load:
            print(" [*] Load SUCCESS")
            reps_dir = "Representations/"+self.dataset_dir+"/" + self.dataset_name + "/run_" + str(self.run) + "/Autoencoder"
            file_dir = reps_dir
            if not os.path.exists(file_dir):
                os.makedirs(reps_dir)
            if len(self.testing_data.shape) == 1:
                high_dim_outputs, reps, c = self.sess.run([self.d, self.e, cost], feed_dict={
                    X: np.reshape(data, [1,
                                         self.input_length])})
            else:
                high_dim_outputs, reps, c = self.sess.run([self.d, self.e, cost], feed_dict={
                    X: np.reshape(data, [len(data), self.input_length])})
            if self.train:
                np.savetxt(os.path.join(file_dir, "{}_{}".format(name, "reps.csv")), np.transpose(reps, (0, 1)),
                           delimiter=",")
            return high_dim_outputs, reps

        else:
            print(" [!] Load failed...")

    def xavier_init(self, nin, nout, const=1):
        low = -const * np.sqrt(1 / (nin + nout))
        high = const * np.sqrt(1 / (nin + nout))

        return tf.random_uniform((nin, nout), minval=low, maxval=high)

    def mask_noise(self, x, v):
        """ add salt-and-pepper noise to the data"""
        x_noise = x.copy()

        n_samples = x.shape[0]
        n_features = x.shape[1]

        for i in range(n_samples):
            mask = np.random.randint(0, n_features, v)

            for m in mask:
                x_noise[i][m] = 0

        return x_noise

    @property
    def model_dir(self):
        """ the name of the  directory """
        return "{}_{}_{}_{}_{}".format(
            "dataset_train", self.batch_size, self.input_length, self.rep_length, self.run)

    def save(self, checkpoint_dir, step, run):
        """ save the trained model """
        model_name = "Autoencoder.model"
        run = "run_" + str(run)
        checkpoint_dir = os.path.join(checkpoint_dir, run, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name))


    def load(self, checkpoint_dir, phase, run):
        """ load a previously trined model """
        run = "run_" + str(run)
        print("[*] Reading checkpoints for", phase)
        checkpoint_dir = os.path.join(checkpoint_dir, run, self.model_dir)
        print("[*] Lading previously trained model from", checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False


