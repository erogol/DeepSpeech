import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import time
import absl.app
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from tensorflow import keras
from functools import partial
from six.moves import zip, range
from tensorflow.keras import layers
from util.config import Config, initialize_globals
from util.feedingTF2 import create_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_progress, create_progressbar


class DenseDropout(keras.layers.Layer):
    def __init__(self, num_hidden, dropout_rate, relu_clip):
        super().__init__()
        self.relu_clip = relu_clip
        self.layer = keras.layers.Dense(num_hidden)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x):
        return self.dropout(keras.activations.relu(self.layer(x), max_value=self.relu_clip))


class RNN(keras.layers.Layer):
    def __init__(self, num_hidden):
        super().__init__()
        # parameter to enable CUDNN
        self.layer = keras.layers.LSTM(num_hidden, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True, return_sequences=True)

    def call(self, x):
        states = self.layer(x)
        return states


class CreateOverlappingWindows(keras.layers.Layer):
    def __init__(self, Config, FLAGS):
        super(CreateOverlappingWindows, self).__init__()
        self.window_width = 2 * Config.n_context + 1
        self.num_channels = Config.n_input
        identity = (np.eye(self.window_width * self.num_channels)
                      .reshape(self.window_width, self.num_channels, self.window_width * self.num_channels))
        self.identity_filter = tf.constant(identity, tf.float32)

    def call(self, x):
        x = tf.nn.conv1d(input=x, filters=self.identity_filter, stride=1, padding='SAME')
        return tf.reshape(x, (-1, self.window_width * self.num_channels))


class DeepSpeech(keras.Model):
    def __init__(self, Config, FLAGS, overlap=True):
        super(DeepSpeech, self).__init__()
        self.window_width = 2 * Config.n_context + 1 
        self.num_channels = Config.n_input
        self.overlap = Config.n_input
        if overlap:
            # Create a constant convolution filter using an identity matrix, so that the
            # convolution returns patches of the input tensor as is, and we can create
            # overlapping windows over the MFCCs.
           self.overlap_layer = CreateOverlappingWindows(Config, FLAGS) 
        self.layer_1 = DenseDropout(Config.n_hidden_1, FLAGS.dropout_rate, FLAGS.relu_clip)
        self.layer_2 = DenseDropout(Config.n_hidden_2, FLAGS.dropout_rate2, FLAGS.relu_clip)
        self.layer_3 = DenseDropout(Config.n_hidden_3, FLAGS.dropout_rate3, FLAGS.relu_clip)
        self.layer_4 = RNN(Config.n_cell_dim)
        self.layer_5 = DenseDropout(Config.n_hidden_5, FLAGS.dropout_rate, FLAGS.relu_clip)
        self.layer_6 = keras.layers.Dense(Config.n_hidden_6)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        if self.overlap:
            x = self.overlap_layer(x)
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = tf.reshape(out, [batch_size, -1, tf.shape(out)[-1]])
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        return out


def main(_):
    use_cuda = tf.test.is_gpu_available()
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Using CUDA: ", use_cuda)
    print("Number of GPUs: ", num_gpus)

    initialize_globals()
    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               cache_path=FLAGS.feature_cache)

    ####################
    # Single GPU
    ####################
    # optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    # model = DeepSpeech(Config, FLAGS)

    ## tf.function gives ~9x speech boost per iteration
    # @tf.function
    # def train_step(model, data):
    #     batch_x, batch_x_lens, batch_y, batch_y_lens = data
    
    #     with tf.GradientTape() as tape:
    #         logits = model(batch_x)

    #         loss = tf.nn.ctc_loss(labels=batch_y,
    #                                 label_length=batch_y_lens,
    #                                 logits=logits,
    #                                 logit_length=batch_x_lens,
    #                                 blank_index=Config.n_hidden_6 - 1,
    #                                 logits_time_major=False)
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     return loss

    # for epoch in range(FLAGS.epochs):
    #     for step, data in enumerate(train_set.take(4)):
    #         start_time = time.time()
    #         loss = train_step(model, data)
    #         step_time = time.time() - start_time
    #         print('Epoch {:>3} - Step {:>3} - Training loss: {:.3f} - Step Time: {:.2f}'.format(epoch, int(step), float(loss), step_time))

    #########################
    # DISTRIBUTED
    #########################
    
    strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices in use: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE_PER_REPLICA = FLAGS.train_batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=GLOBAL_BATCH_SIZE,
                               cache_path=FLAGS.feature_cache)

    train_set_dist = strategy.experimental_distribute_dataset(train_set)

    with strategy.scope():
        # loss
        def compute_loss(batch_y, batch_y_lens, logits, batch_x_lens, Config):
            per_example_loss = tf.nn.ctc_loss(labels=batch_y,
                        label_length=batch_y_lens,
                        logits=logits,
                        logit_length=batch_x_lens,
                        blank_index=Config.n_hidden_6 - 1,
                        logits_time_major=False)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        # metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        # primitives
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
        model = DeepSpeech(Config, FLAGS)

        def train_step(inputs):
            batch_x, batch_x_lens, batch_y, batch_y_lens = data
            with tf.GradientTape() as tape:
                logits = model(batch_x)

                loss = compute_loss(batch_y,
                                    batch_y_lens,
                                    logits,
                                    batch_x_lens,
                                    Config,
                                    )
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss 

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                            args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
        
        # @tf.function
        # def distributed_test_step(dataset_inputs):
        #     return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

        for epoch in range(FLAGS.epochs):
            # TRAIN LOOP
            for step, data in enumerate(train_set_dist):
                start_time = time.time()
                loss = distributed_train_step(data)
                step_time = time.time() - start_time
                print('Epoch {:>3} - Step {:>3} - Training loss: {:.3f} - Step Time: {:.2f}'.format(epoch, int(step), float(loss), step_time))


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)