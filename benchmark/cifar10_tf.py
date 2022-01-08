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

"""Runs CIFAR10 training with differential privacy using Tensorflow."""

import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPSequential


flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training.')
flags.DEFINE_float('momentum', 0.9, 'Optimizer momentum.')
flags.DEFINE_string('lr_schedule', 'cos', 'Learning rate schedule: "constant" or "cos".')
flags.DEFINE_float('noise_multiplier', 1.5, 'Ratio of noise standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 10.0, 'Clipping norm.')
flags.DEFINE_integer('epochs', 90, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 2000, 'Batch size.')
flags.DEFINE_integer('microbatches', -1, 'Number of microbatches (must evenly divide batch_size).')
flags.DEFINE_string('data_dir', None, 'Data directory.')

FLAGS = flags.FLAGS


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STDDEV = (0.2023, 0.1994, 0.2010)


def normalize_images(x):
    return (x - np.reshape(CIFAR_MEAN, [1, 1, 1, 3])) / np.reshape(CIFAR_STDDEV, [1, 1, 1, 3])


def get_cifar10_data(data_dir=None):
    data = tfds.as_numpy(tfds.load(name='cifar10', batch_size=-1, data_dir=data_dir))
    x_train = data['train']['image'] / 255.0
    y_train = data['train']['label']
    x_test = data['test']['image'] / 255.0
    y_test = data['test']['label']
    return normalize_images(x_train).astype(np.float32), y_train, normalize_images(x_test).astype(np.float32), y_test


def main(unused_argv):
    logging.set_verbosity(logging.ERROR)
    if FLAGS.microbatches == -1:
        FLAGS.microbatches = FLAGS.batch_size
    if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Number of microbatches should divide evenly batch_size')

    # Load training and test data.
    train_data, train_labels, test_data, test_labels = get_cifar10_data(data_dir=FLAGS.data_dir)

    # Define a sequential Keras model
    layers = [
        tf.keras.layers.Conv2D(32, 3,
                               strides=1,
                               padding='same',
                               activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        tf.keras.layers.Conv2D(64, 3,
                               strides=1,
                               padding='same',
                               activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        tf.keras.layers.Conv2D(64, 3,
                               strides=1,
                               padding='same',
                               activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        tf.keras.layers.Conv2D(128, 3,
                               strides=1,
                               padding='same',
                               activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ]
    if FLAGS.dpsgd:
        model = DPSequential(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            layers=layers)
    else:
        model = tf.keras.Sequential(layers=layers)

    # Define learning rate schedule, optimizer and loss function
    if FLAGS.lr_schedule == 'cos':
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            FLAGS.learning_rate, FLAGS.epochs * len(train_data) // FLAGS.batch_size)
    else:
        learning_rate = FLAGS.learning_rate
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=FLAGS.momentum)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train model with Keras
    epoch_time = []
    for epoch in range(1, FLAGS.epochs + 1):
        start_time = time.time()
        model.fit(train_data, train_labels,
                  epochs=1,
                  verbose=0,
                  validation_data=None,
                  batch_size=FLAGS.batch_size)
        epoch_time.append(time.time() - start_time)
        print(f'Train Epoch: {epoch} \t took {epoch_time[-1]} seconds')
        test_accuracy = model.evaluate(test_data, test_labels, verbose=0, batch_size=FLAGS.batch_size)[1]
        print(f'    Test acc = {test_accuracy * 100:.3f}%.')

    print('Average epoch time (all epochs): ', np.average(epoch_time))
    print('Median epoch time (all epochs): ', np.median(epoch_time))
    print('Average epoch time (except first): ', np.average(epoch_time[1:]))
    print('Median epoch time (except first): ', np.median(epoch_time[1:]))
    print('Total training time (excluding evaluation): ', np.sum(epoch_time))


if __name__ == '__main__':
    app.run(main)
