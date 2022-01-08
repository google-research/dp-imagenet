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

"""Runs CIFAR10 training with differential privacy using JAX/Objax."""

import argparse
import math
import random
import time

from functools import partial

import jax
import jax.numpy as jn

import numpy as np
import tensorflow_datasets as tfds

import objax


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STDDEV = (0.2023, 0.1994, 0.2010)


def normalize_images(x):
    return (x - np.reshape(CIFAR_MEAN, [1, 3, 1, 1])) / np.reshape(CIFAR_STDDEV, [1, 3, 1, 1])


def get_cifar10_data():
    data = tfds.as_numpy(tfds.load(name="cifar10", batch_size=-1))
    x_train = data["train"]["image"].transpose(0, 3, 1, 2) / 255.0
    y_train = data["train"]["label"]
    x_test = data["test"]["image"].transpose(0, 3, 1, 2) / 255.0
    y_test = data["test"]["label"]
    return normalize_images(x_train), y_train, normalize_images(x_test), y_test


class NumpyDatasetLoader:

    def __init__(self, x, y, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(y)
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=int)

    def __iter__(self):
        example_indices = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(example_indices)
        for idx in range(0, self.size, self.batch_size):
            indices = example_indices[idx:idx + self.batch_size]
            yield (self.x[indices], self.y[indices])

    def __len__(self):
        return self.size


class Cifar10ConvNet(objax.nn.Sequential):
    def __init__(self, nclass=10):
        ops = [
            objax.nn.Conv2D(3, 32, k=3, strides=1, padding=1),
            objax.functional.relu,
            partial(objax.functional.average_pool_2d, size=2, strides=2),

            objax.nn.Conv2D(32, 64, k=3, strides=1, padding=1),
            objax.functional.relu,
            partial(objax.functional.average_pool_2d, size=2, strides=2),

            objax.nn.Conv2D(64, 64, k=3, strides=1, padding=1),
            objax.functional.relu,
            partial(objax.functional.average_pool_2d, size=2, strides=2),

            objax.nn.Conv2D(64, 128, k=3, strides=1, padding=1),
            objax.functional.relu,
            lambda x: x.mean((2, 3)),

            objax.functional.flatten,
            objax.nn.Linear(128, nclass, use_bias=True),
        ]
        super().__init__(ops)


def train(train_loader, train_op, lr):
    start_time = time.time()
    for x, y in train_loader:
        train_op(x, y, lr)
    return time.time() - start_time


def test(test_loader, predict_op):
    num_correct = 0
    for x, y in test_loader:
        y_pred = predict_op(x)
        num_correct += np.count_nonzero(np.argmax(y_pred, axis=1) == y)
    print(f"\tTest set:\tAccuracy: {num_correct/len(test_loader)}")


def main(args):
    x_train, y_train, x_test, y_test = get_cifar10_data()
    train_loader = NumpyDatasetLoader(x_train, y_train, batch_size=args.batch_size)
    test_loader = NumpyDatasetLoader(x_test, y_test, batch_size=args.batch_size_test, shuffle=False)

    # Model
    model = Cifar10ConvNet()
    model_vars = model.vars()

    # Optimizer
    opt = objax.optimizer.Momentum(model_vars, momentum=args.momentum, nesterov=False)

    # Prediction operation
    predict_op = objax.Jit(lambda x: objax.functional.softmax(model(x)), model_vars)

    # Loss and training op
    @objax.Function.with_vars(model_vars)
    def loss_fn(x, label):
        logit = model(x)
        return objax.functional.loss.cross_entropy_logits_sparse(logit, label).mean()

    if args.disable_dp:
        loss_gv = objax.GradValues(loss_fn, model.vars())
    else:
        loss_gv = objax.privacy.dpsgd.PrivateGradValues(loss_fn,
                                                        model_vars,
                                                        args.sigma,
                                                        args.max_per_sample_grad_norm,
                                                        microbatch=1,
                                                        batch_axis=(0, 0),
                                                        use_norm_accumulation=args.norm_acc)

    @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
    def augment_op(x):
        # random flip
        x = jax.lax.cond(objax.random.uniform(()) < 0.5,
                        lambda t: t,
                        lambda t: t[:, :, ::-1],
                        operand=x)
        # random crop
        x_pad = jn.pad(x, [[0, 0], [4, 4], [4, 4]], 'reflect')
        offset = objax.random.randint((2,), 0, 4)
        return jax.lax.dynamic_slice(x_pad, (0, offset[0], offset[1]), (3, 32, 32))

    augment_op = objax.Vectorize(augment_op)

    @objax.Function.with_vars(model_vars + loss_gv.vars() + opt.vars())
    def train_op(x, y, learning_rate):
        if args.disable_dp:
            x = augment_op(x)
        grads, loss = loss_gv(x, y)
        opt(learning_rate, grads)
        return loss

    train_op = objax.Jit(train_op)

    epoch_time = []
    for epoch in range(args.epochs):
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * (epoch + 1) / (args.epochs + 1)))
        else:
            lr = args.lr
        cur_epoch_time = train(train_loader, train_op, lr)
        print(f"Train Epoch: {epoch+1} \t took {cur_epoch_time} seconds")
        epoch_time.append(cur_epoch_time)
        test(test_loader, predict_op)
        if not args.disable_dp:
            epsilon = objax.privacy.dpsgd.analyze_dp(
                q=args.batch_size / len(train_loader),
                noise_multiplier=args.sigma,
                steps=len(train_loader) // args.batch_size * (epoch + 1),
                delta=args.delta)
            print(f"\tPrivacy: (ε = {epsilon:.2f}, δ = {args.delta})")

    print("Average epoch time (all epochs): ", np.average(epoch_time))
    print("Median epoch time (all epochs): ", np.median(epoch_time))
    print("Average epoch time (except first): ", np.average(epoch_time[1:]))
    print("Median epoch time (except first): ", np.median(epoch_time[1:]))
    print("Total training time (excluding evaluation): ", np.sum(epoch_time))


def parse_args():
    parser = argparse.ArgumentParser(description="Objax CIFAR10 DP Training")
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        default=2000,
        type=int,
        metavar="N",
        help="Batch size for training",
    )
    parser.add_argument(
        "--batch-size-test",
        default=200,
        type=int,
        metavar="N",
        help="Batch size for test",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="cos"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.5)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 10.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--norm-acc",
        action="store_true",
        default=False,
        help="Enables norm accumulation in Objax DP-SGD code.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())