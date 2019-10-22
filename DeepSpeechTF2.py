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
        return self.dropout(
            keras.activations.relu(self.layer(x), max_value=self.relu_clip))


class CreateOverlappingWindows(tf.keras.Model):
    def __init__(self):
        super(CreateOverlappingWindows, self).__init__()
        window_width = 2 * Config.n_context + 1
        num_channels = Config.n_input
        identity = (np.eye(window_width * num_channels).reshape(
            window_width, num_channels, window_width * num_channels))
        self.identity_filter = tf.constant(identity, tf.float32)
        self.reshape = layers.Reshape((-1, window_width * num_channels))

    def call(self, x):
        x = tf.nn.conv1d(input=x,
                         filters=self.identity_filter,
                         stride=1,
                         padding='SAME')
        return self.reshape(x)

# def create_model():
#     inputs = tf.keras.Input(shape=(None, None, Config.n_input))

#     x = CreateOverlappingWindows()(inputs)
#     x = layers.Masking()(x)

#     x = DenseDropout(Config.n_hidden_1, FLAGS.dropout_rate, FLAGS.relu_clip)(x)
#     x = DenseDropout(Config.n_hidden_2, FLAGS.dropout_rate2, FLAGS.relu_clip)(x)
#     x = DenseDropout(Config.n_hidden_3, FLAGS.dropout_rate3, FLAGS.relu_clip)(x)
#     # cudnn enabling setting for LSTM
#     x = keras.layers.LSTM(Config.n_cell_dim, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True, return_sequences=True)(x)

#     x = DenseDropout(Config.n_hidden_5, FLAGS.dropout_rate, FLAGS.relu_clip)(x)
#     x = layers.Dense(Config.n_hidden_6)(x)
#     return tf.keras.Model(inputs=inputs, outputs=x, name='DeepSpeechModel')


class DeepSpeech(keras.Model):
    def __init__(self):
        super(DeepSpeech, self).__init__()
        self.overlap_layer = CreateOverlappingWindows()
        self.masking = layers.Masking()
        self.dense1 = DenseDropout(Config.n_hidden_1, FLAGS.dropout_rate,
                                   FLAGS.relu_clip)
        self.dense2 = DenseDropout(Config.n_hidden_2, FLAGS.dropout_rate2,
                                   FLAGS.relu_clip)
        self.dense3 = DenseDropout(Config.n_hidden_3, FLAGS.dropout_rate3,
                                   FLAGS.relu_clip)
        self.rnn = keras.layers.LSTM(Config.n_cell_dim,
                                     activation='tanh',
                                     recurrent_activation='sigmoid',
                                     recurrent_dropout=0.0,
                                     unroll=False,
                                     use_bias=True,
                                     return_sequences=True)
        self.dense5 = DenseDropout(Config.n_hidden_5, FLAGS.dropout_rate,
                                   FLAGS.relu_clip)
        self.dense_out = layers.Dense(Config.n_hidden_6)

    @tf.function(input_signature=(tf.TensorSpec([None, None, None], dtype=tf.float32), ))
    def call(self, x):
        x = self.overlap_layer(x)
        x = self.masking(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.rnn(x)
        x = self.dense5(x)
        return self.dense_out(x)


def main(_):
    use_cuda = tf.test.is_gpu_available()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)
    print("Using CUDA: ", use_cuda)
    print("Number of GPUs: ", num_gpus)
    initialize_globals()

    ####################
    # Single GPU
    ####################
    # train_set = create_dataset(FLAGS.train_files.split(','),
    #                            batch_size=FLAGS.train_batch_size,
    #                            cache_path=FLAGS.feature_cache)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    # model = DeepSpeech(Config, FLAGS)
    # # model = create_model()

    # # tf.function gives ~9x speech boost per iteration
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

    # for epoch in range(FLAGS.epochs):c
    #     for step, data in enumerate(train_set.take(4)):
    #         start_time = time.time()
    #         loss = train_step(model, data)
    #         step_time = time.time() - start_time
    #         print('Epoch {:>3} - Step {:>3} - Training loss: {:.3f} - Step Time: {:.2f}'.format(epoch, int(step), float(loss), step_time))

    #########################
    # DISTRIBUTED
    #########################
    strategy = tf.distribute.MirroredStrategy()  #devices=["/gpu:0", "/gpu:1"])
    print('Number of devices in use: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE_PER_REPLICA = FLAGS.train_batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # create training set
    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=GLOBAL_BATCH_SIZE,
                               cache_path=FLAGS.feature_cache)
    train_set_dist = strategy.experimental_distribute_dataset(train_set)

    # create development set
    if FLAGS.dev_files:
        dev_csvs = FLAGS.dev_files.split(',')
        dev_set = create_dataset(dev_csvs, batch_size=FLAGS.dev_batch_size)
        dev_set_dist = strategy.experimental_distribute_dataset(dev_set)

    with strategy.scope():
        # primitives
        model = DeepSpeech()
        # model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

        # create checkpoint path
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'train', 'checkpoint')
        ckpt = tf.train.Checkpoint(model=model,
                                   optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path)
        best_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'train', 'best')
        best_ckpt = tf.train.Checkpoint(model=model,
                                   optimizer=optimizer)
        best_ckpt_manager = tf.train.CheckpointManager(best_ckpt, best_checkpoint_path, max_to_keep=1)


        # if a checkpoint exists, restore the latest checkpoint.
        # ckpt.restore(ckpt_manager.latest_checkpoint)
        # print ('Latest checkpoint restored!!')

        # metrics
        avg_train_loss = keras.metrics.Mean(name='avg_train_loss')
        avg_eval_loss = keras.metrics.Mean(name='avg_eval_loss')

        # https://github.com/tensorflow/tensorflow/issues/29911
        # @tf.function
        def distributed_train_step(batch_x, batch_x_lens, batch_y,
                                   batch_y_lens):
            def train_step(batch_x, batch_x_lens, batch_y, batch_y_lens):
                with tf.GradientTape() as tape:
                    logits = model(batch_x, training=True)
                    ctc_loss = tf.nn.ctc_loss(labels=batch_y,
                                              label_length=batch_y_lens,
                                              logits=logits,
                                              logit_length=batch_x_lens,
                                              blank_index=Config.n_hidden_6 -
                                              1,
                                              logits_time_major=False)
                    loss = tf.nn.compute_average_loss(
                        ctc_loss, global_batch_size=GLOBAL_BATCH_SIZE)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads,
                                              model.trainable_variables))
                return loss

            per_replica_loss = strategy.experimental_run_v2(
                train_step,
                args=(batch_x, batch_x_lens, batch_y, batch_y_lens))
            final_loss = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                         per_replica_loss,
                                         axis=None)
            avg_train_loss(final_loss)
            return final_loss

        # https://github.com/tensorflow/tensorflow/issues/29911
        # @tf.function
        def distributed_test_step(batch_x, batch_x_lens, batch_y,
                                  batch_y_lens):
            def eval_step(batch_x, batch_x_lens, batch_y, batch_y_lens):
                logits = model(batch_x, training=False)
                ctc_loss = tf.nn.ctc_loss(labels=batch_y,
                                          label_length=batch_y_lens,
                                          logits=logits,
                                          logit_length=batch_x_lens,
                                          blank_index=Config.n_hidden_6 - 1,
                                          logits_time_major=False)
                loss = tf.nn.compute_average_loss(
                    ctc_loss, global_batch_size=GLOBAL_BATCH_SIZE)
                return loss

            per_replica_loss = strategy.experimental_run_v2(
                eval_step, args=(batch_x, batch_x_lens, batch_y, batch_y_lens))
            final_eval_loss = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                              per_replica_loss,
                                              axis=None)
            avg_eval_loss(final_eval_loss)
            return final_eval_loss

    with strategy.scope():
        best_eval_loss = float('Inf')
        for epoch in range(FLAGS.epochs):
            # TRAIN LOOP
            loader_time = 0
            for step, data in enumerate(train_set_dist):
                loader_time = time.time() - loader_time
                start_time = time.time()
                batch_x, batch_x_lens, batch_y, batch_y_lens = data
                loss = distributed_train_step(batch_x, batch_x_lens, batch_y,
                                              batch_y_lens)
                step_time = time.time() - start_time
                print(
                    'Epoch {:>3} - Step {:>3} - Loss: {:.3f}/{:.3f} - Loader Time: {:.2} - Step Time: {:.2f}'
                    .format(epoch, int(step), float(loss),
                            avg_train_loss.result(), loader_time, step_time))
                ckpt_manager.save()
                loader_time = time.time()

            # EVAL LOOP
            if FLAGS.dev_files:
                for step, data in enumerate(dev_set_dist):
                    batch_x, batch_x_lens, batch_y, batch_y_lens = data
                    _ = distributed_test_step(batch_x, batch_x_lens, batch_y, batch_y_lens)
                print('Avg. Eval Loss: {:.3f} - Best Avg. Eval Loss: {:.3f}'.format(avg_eval_loss.result(), best_eval_loss))
                if avg_eval_loss.result() < best_eval_loss:
                    best_eval_loss = avg_eval_loss.result()
                    best_ckpt_manager.save()


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)