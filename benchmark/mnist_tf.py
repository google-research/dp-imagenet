# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs MNIST training with differential privacy using Tensorflow."""

import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPSequential


flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .25, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.3, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.5, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 15, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches (must evenly divide batch_size)')

FLAGS = flags.FLAGS


def load_mnist():
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    """Loads MNIST and preprocesses to combine training and validation data."""
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = (np.array(train_data, dtype=np.float32) / 255 - MNIST_MEAN) / MNIST_STD
    test_data = (np.array(test_data, dtype=np.float32) / 255 - MNIST_MEAN) / MNIST_STD

    train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
    test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)
    return train_data, train_labels, test_data, test_labels


def main(unused_argv):
    logging.set_verbosity(logging.ERROR)
    if FLAGS.microbatches == -1:
        FLAGS.microbatches = FLAGS.batch_size
    if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Number of microbatches should divide evenly batch_size')

    # Load training and test data.
    train_data, train_labels, test_data, test_labels = load_mnist()

    # Define a sequential Keras model
    layers = [
        tf.keras.layers.Conv2D(16, 8,
                               strides=2,
                               padding='same',
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4,
                               strides=2,
                               padding='valid',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ]
    if FLAGS.dpsgd:
        model = DPSequential(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            layers=layers)
    else:
        model = tf.keras.Sequential(layers=layers)

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train model with Keras
    epoch_time = []
    for epoch in range(FLAGS.epochs):
        start_time = time.time()
        model.fit(train_data, train_labels,
                  epochs=1,
                  validation_data=None,
                  batch_size=FLAGS.batch_size)
        epoch_time.append(time.time() - start_time)
        print(f"Train Epoch: {epoch} \t took {epoch_time[-1]} seconds")

    print('Average epoch time (all epochs): ', np.average(epoch_time))
    print('Median epoch time (all epochs): ', np.median(epoch_time))
    print('Average epoch time (except first): ', np.average(epoch_time[1:]))
    print('Median epoch time (except first): ', np.median(epoch_time[1:]))

    model.evaluate(test_data, test_labels, batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    app.run(main)