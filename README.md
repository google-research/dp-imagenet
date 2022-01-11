
# Scaling differentially-private image classification.

Code for the tech report "Training large scale image models with differential privacy."
by Alexey Kurakin, Steve Chien, Shuang Song, Roxana Geambasu, Andreas Terzis and Abhradeep Thakurta.

This is not an officially supported Google product.

## Repository structure

* `benchmarks` directory contains code which we used to compare performance of various DP-SGD
  frameworks on CIFAR10 and MNIST
* `imagenet` directory contains Imagenet trainign code.

## Installation

1. If you are going to use NVIDIA GPU then install latest NVIDIA drivers,
   [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#introduction)
   and [CuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.htm).
   While latest versions are not strictly necessary to run the code, we sometimes observed
   slower performance with older versions of CUDA and CuDNN.

2. Set up Python virtual environment with all necessary libraries:

    ```bash
    # Create virtualenv
    virtualenv -p python3 ~/.venv/dp_imagenet
    source ~/.venv/dp_imagenet/bin/activate
    # Install Objax with CUDA
    pip install --upgrade objax
    pip install --upgrade jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
    # Tensorflow and TFDS (for datasets readers)
    pip install tensorflow
    pip install tensorflow-datasets
    ```

3. Extra libraries for TF and Opacus benchmarks:

    ```bash
    pip install tensorflow-privacy
    pip install opacus
    pip install torchvision
    pip install tensorboard
    ```

4. Follow instructions at https://www.tensorflow.org/datasets/catalog/imagenet2012 to download Imagenet dataset for TFDS.


Before running any code, make sure to enter virtual environment and setup `PYTHONPATH`:

```bash
# Enter virtual env, set up path
source ~/.venv/dp_imagenet/bin/activate
cd ${REPOSITORY_DIRECTORY}
export PYTHONPATH=$PYTHONPATH:.
```

## Training Imagenet models with DP

Here are few examples showing how to run Imagenet training with and without DP:

```bash
# Resnet50 without DP
python imagenet/imagenet_train.py --tfds_data_dir="${TFDS_DATA_DIR}" --max_eval_batches=10 --eval_every_n_steps=100 --train_device_batch_size=64 --disable_dp

# Resnet18 without DP
python imagenet/imagenet_train.py --tfds_data_dir="${TFDS_DATA_DIR}" --max_eval_batches=10 --eval_every_n_steps=100 --model=resnet18 --train_device_batch_size=64 --disable_dp

# Resnet18 with DP
python imagenet/imagenet_train.py --tfds_data_dir="${TFDS_DATA_DIR}" --max_eval_batches=10 --eval_every_n_steps=100 --model=resnet18 --train_device_batch_size=64
```

To pre-train model on Places365 and finetune with differential privacy on Imagenet use the following commands:

```bash
# Prepare directory for Places365 checkpoint
PLACES_CHECKPOINT_DIR="${HOME}/experiments/places365"
mkdir -p "${PLACES_CHECKPOINT_DIR}"

# Pre-train model on Places365 without differential privacy
python imagenet/imagenet_train.py \
  --tfds_data_dir="${TFDS_DATA_DIR}" \
  --dataset=places365 \
  --eval_every_n_steps=1024 \
  --model=resnet18 \
  --num_train_epochs=80 \
  --lr_warmup_epochs=4 \
  --base_learning_rate=0.05 \
  --disable_dp \
  --train_device_batch_size=128 \
  --model_dir="${PLACES_CHECKPOINT_DIR}"

# Prepare directory for Imagenet checkpoint
IMAGENET_DP_CHECKPOINT_DIR="${HOME}/experiments/imagenet_dp"
mkdir -p "${IMAGENET_DP_CHECKPOINT_DIR}"

# Finetune model on Imagenet with differential privacy
python imagenet/imagenet_train.py \
  --tfds_data_dir="${TFDS_DATA_DIR}" \
  --eval_every_n_steps=1024 \
  --model=resnet18 \
  --num_train_epochs=40 \
  --dp_clip_norm=20 \
  --dp_sigma=0.032 \
  --grad_acc_steps=1024 \
  --base_learning_rate=1e-5 \
  --weight_decay=0 \
  --lr_schedule="fixed" \
  --optimizer="adam" \
  --lr_warmup_epochs=4 \
  --num_layers_to_freeze=8 \
  --keep_ckpts=100 \
  --finetune_path="${PLACES_CHECKPOINT_DIR}/ckpt/0000141312.npz" \
  --model_dir="${IMAGENET_DP_CHECKPOINT_DIR}"
```

## Running DP-SGD benchmarks

Following commands were used to obtain benchmarks of various frameworks for the tech report.
All of them were run on `n1-standard-96` Google Cloud machine with 8 v100 GPUs.
All numbers were obtains with CUDA 11.4 and CuDNN 8.2.2.26.

Objax benchmarks:

```bash
# MNIST benchmark without DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/mnist_objax.py --disable-dp

# MNIST benchmark with DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/mnist_objax.py

# CIFAR10 benchmark without DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/cifar10_objax.py --disable-dp

# CIFAR10 benchmark with DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/cifar10_objax.py

# Imagenet benchmark Resnet18 without DP
python imagenet/imagenet_train.py --tfds_data_dir="${TFDS_DATA_DIR}" --disable_dp --base_learning_rate=0.2

# Imagenet benchmark Resnet18 with DP
python imagenet/imagenet_train.py --tfds_data_dir="${TFDS_DATA_DIR}" --base_learning_rate=2.0
```

Opacus benchmarks:

```bash
# MNIST benchmark without DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/mnist_opacus.py --disable-dp

# MNIST benchmark with DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/mnist_opacus.py

# CIFAR10 benchmark without DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/cifar10_opacus.py --disable-dp

# CIFAR10 benchmark with DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/cifar10_opacus.py
```

Tensorflow benchmarks:

```bash
# MNIST benchmark without DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/mnist_tf.py --dpsgd=False

# MNIST benchmark with DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/mnist_tf.py

# CIFAR10 example without DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/cifar10_tf.py --dpsgd=False

# CIFAR10 example with DP
CUDA_VISIBLE_DEVICES=0 python benchmarks/cifar10_tf.py
```
