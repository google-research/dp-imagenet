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

"""
Runs MNIST training with differential privacy using Opacus.

Code is based on https://github.com/pytorch/opacus/blob/master/examples/mnist.py
and modified for our benchmarks.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))    # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)    # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))    # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)    # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)    # -> [B, 512]
        x = F.relu(self.fc1(x))    # -> [B, 32]
        x = self.fc2(x)    # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


def train(model, device, train_loader, optimizer, num_batches):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    start_time = time.time()
    for _batch_idx in range(num_batches):
        data, target = next(train_loader)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    epoch_time = time.time() - start_time
    return epoch_time


def test(model, device, test_loader, ntest):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()    # sum up batch loss
            pred = output.argmax(
                    dim=1, keepdim=True
            )    # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= ntest
    print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                    test_loss,
                    correct,
                    ntest,
                    100.0 * correct / ntest,
                    )
    )
    return correct / ntest


def main():
    tf.config.experimental.set_visible_devices([], "GPU")  # prevent tfds from using GPU
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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
    device = torch.device('cuda')

    # -------- change data loader -------- #
    data = tfds.as_numpy(tfds.load(name='mnist', batch_size=-1, data_dir=args.data_root))
    train_image = (data['train']['image'].transpose(0, 3, 1, 2) / 255 - MNIST_MEAN) / MNIST_STD
    test_image = (data['test']['image'].transpose(0, 3, 1, 2) / 255 - MNIST_MEAN) / MNIST_STD
    ntrain = train_image.shape[0]
    ntest = test_image.shape[0]
    train_image = torch.from_numpy(train_image).float()
    test_image = torch.from_numpy(test_image).float()
    train_label = torch.from_numpy(data['train']['label']).long()
    test_label = torch.from_numpy(data['test']['label']).long()
    del data

    num_batches = ntrain // args.batch_size

    def data_stream():
        while True:
            perm = np.random.permutation(ntrain)
            for i in range(num_batches):
                batch_idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
                yield train_image[batch_idx], train_label[batch_idx]
    train_loader = data_stream()

    def data_stream_test():
        yield test_image, test_label
    test_loader = data_stream_test()

    model = SampleConvNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
                model,
                batch_size=args.batch_size,
                sample_size=ntrain,
                alphas=[],
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
        )
        privacy_engine.attach(optimizer)
    else:
        print('run nonprivate')

    epoch_time = []
    for epoch in range(args.epochs):
        epoch_time_cur = train(model, device, train_loader, optimizer, num_batches)
        epoch_time.append(epoch_time_cur)
        print(f'Train Epoch: {epoch} \t took {epoch_time_cur} seconds')

    print('Average epoch time (all epochs): ', np.average(epoch_time))
    print('Median epoch time (all epochs): ', np.median(epoch_time))
    print('Average epoch time (except first): ', np.average(epoch_time[1:]))
    print('Median epoch time (except first): ', np.median(epoch_time[1:]))

    test(model, device, test_loader, ntest)


if __name__ == "__main__":
    main()