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

"""Runs MNITS training with differential privacy using JAX/Objax."""

import argparse
from functools import partial
import numpy as np

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds

import objax
from objax.functional import flatten, max_pool_2d
from objax.nn import Conv2D, Linear, Sequential


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(Sequential):
    def __init__(self, nclass=10):
        ops = [
                Conv2D(1, 16, 8, padding=objax.ConvPadding.SAME, strides=2),
                objax.functional.relu,
                partial(max_pool_2d, size=2, strides=1),

                Conv2D(16, 32, 4, padding=objax.ConvPadding.VALID, strides=2),
                objax.functional.relu,
                partial(max_pool_2d, size=2, strides=1),

                flatten,
                Linear(32 * 16, 32),
                objax.functional.relu,
                Linear(32, nclass),
        ]
        super().__init__(ops)

    def name(self):
        return "SampleConvNet"


def get_parameters():
    # Training settings
    parser = argparse.ArgumentParser(description="Objax MNIST Example")
    parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=250,
            metavar="B",
            help="input batch size for training",
    )
    parser.add_argument(
            "-n",
            "--epochs",
            type=int,
            default=15,
            metavar="N",
            help="number of epochs to train",
    )
    parser.add_argument(
            "--lr",
            type=float,
            default=0.25,
            metavar="LR",
            help="learning rate",
    )
    parser.add_argument(
            "--sigma",
            type=float,
            default=1.3,
            metavar="S",
            help="Noise multiplier",
    )
    parser.add_argument(
            "-c",
            "--max-per-sample-grad_norm",
            type=float,
            default=1.5,
            metavar="C",
            help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
            "--disable-dp",
            action="store_true",
            default=False,
            help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
            "--data-root",
            type=str,
            default=".",
            help="Where MNIST is/will be stored",
    )
    args = parser.parse_args()
    return args


def main():
    tf.config.experimental.set_visible_devices([], "GPU")  # prevent tfds from using GPU
    args = get_parameters()

    data = tfds.as_numpy(tfds.load(name='mnist', batch_size=-1, data_dir=args.data_root))
    train_image = (data['train']['image'].transpose(0, 3, 1, 2) / 255 - MNIST_MEAN) / MNIST_STD
    test_image = (data['test']['image'].transpose(0, 3, 1, 2) / 255 - MNIST_MEAN) / MNIST_STD
    train_label = data['train']['label']
    test_label = data['test']['label']
    ntrain = train_image.shape[0]
    ntest = test_image.shape[0]
    del data

    def data_stream():
        while True:
            perm = np.random.permutation(ntrain)
            for i in range(num_batches):
                batch_idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
                yield train_image[batch_idx], train_label[batch_idx]

    num_batches = ntrain // args.batch_size
    model = SampleConvNet()
    model_vars = model.vars()
    opt = objax.optimizer.SGD(model_vars)
    predict = objax.Jit(lambda x: objax.functional.softmax(model(x, training=False)),
                        model_vars)

    def loss(x, label):
        logit = model(x, training=True)
        return objax.functional.loss.cross_entropy_logits_sparse(logit, label).mean()

    if not args.disable_dp:
        gv = objax.privacy.dpsgd.PrivateGradValues(loss,
                                                   model_vars,
                                                   noise_multiplier=args.sigma,
                                                   l2_norm_clip=args.max_per_sample_grad_norm,
                                                   microbatch=1,
                                                   batch_axis=(0, 0))
    else:
        gv = objax.GradValues(loss, model_vars)

    def train_op(x, y):
        g, v = gv(x, y)
        opt(args.lr, g)
        return v

    train_op = objax.Jit(train_op, gv.vars() + opt.vars())
    batches = data_stream()
    epoch_time = []
    for epoch in range(args.epochs):
        start_time = time.time()
        losses = []
        for _ in range(num_batches):
            x, y = next(batches)
            losses.append(train_op(x, y))
        epoch_time_cur = time.time() - start_time
        epoch_time.append(epoch_time_cur)
        print(f'Train Epoch: {epoch} \t took {epoch_time_cur} seconds')

    print('Average epoch time (all epochs): ', np.average(epoch_time))
    print('Median epoch time (all epochs): ', np.median(epoch_time))
    print('Average epoch time (except first): ', np.average(epoch_time[1:]))
    print('Median epoch time (except first): ', np.median(epoch_time[1:]))

    # Evaluation
    correct = (np.argmax(predict(test_image), axis=1) == test_label).sum()
    test_loss = loss(test_image, test_label)
    print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, correct, ntest, 100.0 * correct / ntest)
    )


if __name__ == "__main__":
    main()
