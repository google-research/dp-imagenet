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

"""Imagenet training with differential privacy using JAX/Objax."""

import os
import time

import jax
import jax.numpy as jn
import numpy as np
from absl import app, flags, logging

import objax
from objax.typing import JaxArray
from imagenet import imagenet_data
from objax.zoo.resnet_v2 import ResNet50, ResNet18
from objax.zoo.wide_resnet import WideResNet

flags.DEFINE_string('model_dir', '', 'Model directory.')
flags.DEFINE_integer('keep_ckpts', 4, 'Number of checkpoints to keep.')
flags.DEFINE_integer('train_device_batch_size', 128, 'Per-device training batch size.')
flags.DEFINE_integer('grad_acc_steps', 1,
                     'Number of steps for gradients accumulation, used to simulate large batches.')
flags.DEFINE_integer('eval_device_batch_size', 250, 'Per-device eval batch size.')
flags.DEFINE_integer('max_eval_batches', -1, 'Maximum number of batches used for evaluation, '
                                             'zero or negative number means use all batches.')
flags.DEFINE_integer('eval_every_n_steps', 1000, 'How often to run eval.')
flags.DEFINE_float('num_train_epochs', 10, 'Number of training epochs.')
flags.DEFINE_float('base_learning_rate', 2.0, 'Base learning rate.')
flags.DEFINE_float('lr_warmup_epochs', 1.0,
                   'Number of learning rate warmup epochs.')
flags.DEFINE_string('lr_schedule', 'cos', 'Learning rate schedule: "cos" or "fixed"')
flags.DEFINE_string('optimizer', 'momentum', 'Optimizer to use: "momentum" or "adam"')

flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay (L2 loss) coefficient.')
flags.DEFINE_string('model', 'resnet18', 'Model to use.')
flags.DEFINE_string('tfds_data_dir', None, 'Optional TFDS data directory.')
flags.DEFINE_string('dataset', 'imagenet', 'Dataset to use')

flags.DEFINE_boolean('disable_dp', False, 'If true then train without DP.')
flags.DEFINE_float('dp_sigma', 0.00001, 'DP noise multiplier.')
flags.DEFINE_float('dp_clip_norm', 1.0, 'DP gradient clipping norm.')

flags.DEFINE_float('dp_delta', 1e-6, 'DP-SGD delta for eps computation.')

flags.DEFINE_string('finetune_path', '',
                    'Path to checkpoint which is used as finetuning initialization.')
flags.DEFINE_boolean('finetune_cut_last_layer', True,
                     'If True then last layer will be cut for finetuning.')
flags.DEFINE_integer('num_layers_to_freeze', 0, 'Number of layers to freeze for finetuning.')

FLAGS = flags.FLAGS

NUM_CLASSES = 1000


class GradientAccumulationOptimizerWrapper(objax.Module):

    def __init__(self, optimizer, scaler=1.0):
        self.scaler = scaler
        self.optimizer = optimizer
        self.accum = objax.ModuleList(objax.StateVar(jn.zeros_like(x.value)) for x in self.optimizer.train_vars)

    def __call__(self, lr, grads, apply_updates):
        assert len(grads) == len(self.optimizer.train_vars), 'Expecting as many gradients as trainable variables'
        for g, a in zip(grads, self.accum):
            a.value += g * self.scaler
        if apply_updates:
            self.optimizer(lr, [a.value for a in self.accum])
            for a in self.accum:
                a.value = jn.zeros_like(a.value)


class CustomResnet(objax.zoo.resnet_v2.ResNetV2):
  """Implementation of custom ResNet v2."""

  def __init__(self,
               name: str,
               in_channels: int,
               num_classes: int):
    """Creates custom resnet instance.

    Args:
        name: resnet name
        in_channels: number of channels in the input image.
        num_classes: number of output classes.
    """
    # Name syntax:
    # "rn_cX_bA_B_C_D" - resnet without bottleneck
    # "rnb_cX_bA_B_C_D" - resnet with bottleneck
    # X - number of channels, (A, B, C, D) - number of blocks per group
    # parsing the name
    assert name.startswith('rn_c') or name.startswith('rnb_c')
    bottleneck = name.startswith('rnb_')
    name = name[name.find('_')+2:]
    num_ch, name = name.split('_', 1)
    num_ch = int(num_ch)
    assert name.startswith('b')
    blocks_per_group = list(int(b) for b in name[1:].split('_'))
    assert len(blocks_per_group) == 4
    super().__init__(in_channels=in_channels,
                     num_classes=num_classes,
                     blocks_per_group=blocks_per_group,
                     bottleneck=bottleneck,
                     channels_per_group=(num_ch, num_ch*2, num_ch*4, num_ch*8),
                     group_use_projection=(True, True, True, True),
                     normalization_fn=objax.nn.GroupNorm2D,
                     activation_fn=objax.functional.relu)


def make_model(model_name, num_classes):
  model_name = model_name.lower()
  if model_name.startswith('rn'):
    return CustomResnet(model_name, in_channels=3, num_classes=num_classes)
  if model_name.startswith('wrn'):
    # syntax is 'wrn_D_W' where D - depth, W - width
    d, w = (int(val) for val in model_name.split('_')[1:])
    return WideResNet(nin=3, nclass=num_classes, depth=d, width=w, bn=objax.nn.GroupNorm2D)
  if model_name == 'resnet50':
    return ResNet50(in_channels=3,
                    num_classes=num_classes,
                    normalization_fn=objax.nn.GroupNorm2D)
  if model_name == 'resnet18':
    return ResNet18(in_channels=3,
                    num_classes=num_classes,
                    normalization_fn=objax.nn.GroupNorm2D)
  raise ValueError(f'Unsupported model type: {model_name}')


class Experiment:
    """Class with all code to run experiment."""

    def __init__(self):
        # Some constants
        self.total_num_replicas = jax.device_count()
        self.total_batch_size = FLAGS.train_device_batch_size * self.total_num_replicas
        self.base_learning_rate = FLAGS.base_learning_rate * self.total_batch_size * FLAGS.grad_acc_steps / 256
        self.save_summaries = bool(FLAGS.model_dir)
        self.lr_warmup_epochs = FLAGS.lr_warmup_epochs
        self.num_train_epochs = FLAGS.num_train_epochs
        # Dataset
        self.train_split = imagenet_data.get_train_dataset_split(FLAGS.dataset)
        self.eval_split = imagenet_data.get_eval_dataset_split(FLAGS.dataset)
        # Create model
        self.model = make_model(FLAGS.model, self.train_split.num_classes)
        model_vars = self.model.vars()
        print('Model variables:')
        print(model_vars)
        print(flush=True)
        if FLAGS.num_layers_to_freeze > 0:
            self.trainable_vars = self.model[FLAGS.num_layers_to_freeze:].vars()
            print(f'Keeping {FLAGS.num_layers_to_freeze} layers frozen')
            print('Trainable variables:')
            print(self.trainable_vars)
            print(flush=True)
        else:
            self.trainable_vars = model_vars
        # Create parallel eval op
        self.evaluate_batch_parallel = objax.Parallel(self.evaluate_batch, model_vars,
                                                      reduce=lambda x: x.sum(0))
        # Create parallel training op
        if FLAGS.optimizer == 'momentum':
            self.optimizer = objax.optimizer.Momentum(self.trainable_vars, momentum=0.9, nesterov=True)
        elif FLAGS.optimizer == 'adam':
            self.optimizer = objax.optimizer.Adam(self.trainable_vars)
        else:
            raise ValueError(f'Unsupported optimizer: {FLAGS.optimizer}')
        if FLAGS.grad_acc_steps > 1:
            self.optimizer = GradientAccumulationOptimizerWrapper(self.optimizer, 1. / FLAGS.grad_acc_steps)
        elif FLAGS.grad_acc_steps != 1:
            raise ValueError('--grad_acc_steps has to be greater or equal to 1.')

        if FLAGS.disable_dp:
            self.compute_grads_loss = objax.GradValues(self.loss_fn, self.trainable_vars)
        else:
            self.compute_grads_loss = objax.privacy.dpsgd.PrivateGradValues(
                self.loss_fn,
                self.trainable_vars,
                FLAGS.dp_sigma,
                FLAGS.dp_clip_norm,
                microbatch=1,
                batch_axis=(0, 0))
        self.all_vars = model_vars + self.optimizer.vars() + objax.random.DEFAULT_GENERATOR.vars()
        self.train_op_parallel = objax.Parallel(
            self.train_op, self.all_vars, static_argnums=(3,), reduce=lambda x: x[0])
        # Summary writer
        if self.save_summaries:
            self.summary_writer = objax.jaxboard.SummaryWriter(os.path.join(
                FLAGS.model_dir, 'tb'))
        else:
            self.summary_writer = None

    def evaluate_batch(self, images, labels):
        logits = self.model(images, training=False)
        num_correct = jn.count_nonzero(jn.equal(jn.argmax(logits, axis=1), labels))
        return num_correct

    def run_eval(self):
        """Runs evaluation and returns top-1 accuracy."""
        test_ds = imagenet_data.load(
            self.eval_split,
            is_training=False,
            batch_dims=[jax.local_device_count() * FLAGS.eval_device_batch_size],
            tfds_data_dir=FLAGS.tfds_data_dir)

        correct_pred = 0
        total_examples = 0
        for batch_index, batch in enumerate(test_ds):
            correct_pred += self.evaluate_batch_parallel(batch['images'],
                                                         batch['labels'])
            total_examples += batch['images'].shape[0]
            if ((FLAGS.max_eval_batches > 0)
                    and (batch_index + 1 >= FLAGS.max_eval_batches)):
                break

        return correct_pred / total_examples

    def loss_fn(self, images, labels):
        """Computes loss function.

        Args:
          images: tensor with images NCHW
          labels: tensors with dense labels, shape (batch_size,)

        Returns:
          Tuple (total_loss, losses_dictionary).
        """
        logits = self.model(images, training=True)
        xent_loss = objax.functional.loss.cross_entropy_logits_sparse(logits, labels).mean()
        wd_loss = FLAGS.weight_decay * 0.5 * sum((v.value ** 2).sum()
                                                 for k, v in self.trainable_vars.items()
                                                 if k.endswith('.w'))
        total_loss = xent_loss + wd_loss
        return total_loss, {'total_loss': total_loss,
                            'xent_loss': xent_loss,
                            'wd_loss': wd_loss}

    def learning_rate(self, epoch: float):
        """Computes learning rate for given fractional epoch."""
        epoch = jn.minimum(epoch, self.num_train_epochs)
        if FLAGS.lr_schedule == 'cos':
            cos_decay_epochs = self.num_train_epochs - self.lr_warmup_epochs
            lr_multiplier = jn.where(
              epoch < self.lr_warmup_epochs,
              x=epoch / self.lr_warmup_epochs,
              y=0.5 * (1 + jn.cos(jn.pi * (epoch - self.lr_warmup_epochs) / cos_decay_epochs))
            )
        elif FLAGS.lr_schedule == 'fixed':
            lr_multiplier = jn.where(
              epoch < self.lr_warmup_epochs,
              x=epoch / self.lr_warmup_epochs,
              y=1.0)
        else:
            raise ValueError(f'Unsupported LR schedule: {FLAGS.lr_schedule}')
        return self.base_learning_rate * lr_multiplier

    def train_op(self, images, labels, cur_epoch, apply_updates):
        cur_epoch = cur_epoch[0]  # because cur_epoch is array of size 1
        grads, (_, losses_dict) = self.compute_grads_loss(images, labels)
        grads = objax.functional.parallel.pmean(grads)
        losses_dict = objax.functional.parallel.pmean(losses_dict)
        learning_rate = self.learning_rate(cur_epoch)
        if FLAGS.grad_acc_steps > 1:
            self.optimizer(learning_rate, grads, apply_updates)
        else:
            self.optimizer(learning_rate, grads)
        return dict(**losses_dict, learning_rate=learning_rate, epoch=cur_epoch)

    def train_and_eval(self):
        """Runs training and evaluation."""
        train_ds = imagenet_data.load(
            self.train_split,
            is_training=True,
            batch_dims=[jax.local_device_count() * FLAGS.train_device_batch_size],
            tfds_data_dir=FLAGS.tfds_data_dir)

        # in case of gradient accumulation - this is number of virtual steps per epoch
        steps_per_epoch = self.train_split.num_examples / self.total_batch_size
        # total number of virtual training steps
        total_train_steps = int(steps_per_epoch * FLAGS.num_train_epochs)
        eval_every_n_steps = FLAGS.eval_every_n_steps

        if self.save_summaries:
            checkpoint = objax.io.Checkpoint(FLAGS.model_dir, keep_ckpts=FLAGS.keep_ckpts)
            start_step, _ = checkpoint.restore(self.all_vars)
        else:
            checkpoint = None
            start_step = 0

        if (start_step == 0) and FLAGS.finetune_path:
          # All supported models are subclasses of objax.nn.Sequential
          # Last layer of these sequential models is classification head.
          # Thus truncating classification head to restore finetuning
          # initialization.
          print('Finetuning initialization from checkpoint: ', FLAGS.finetune_path)
          if FLAGS.finetune_cut_last_layer:
              objax.io.Checkpoint.LOAD_FN(
                  FLAGS.finetune_path,
                  self.model[:-1].vars(),
                  objax.util.Renamer({f'({self.model.__class__.__name__})': '(Sequential)'}))
          else:
              objax.io.Checkpoint.LOAD_FN(
                  FLAGS.finetune_path,
                  self.model.vars())

        cur_epoch = np.zeros([jax.local_device_count()], dtype=np.float32)
        total_training_time = 0.0
        train_time_per_epoch = 0.0
        for big_step in range(start_step, total_train_steps, eval_every_n_steps):
            print(f'Running training steps {big_step + 1} - {big_step + eval_every_n_steps}', flush=True)
            with self.all_vars.replicate():
                # training
                start_time = time.time()
                for cur_step in range(big_step + 1, big_step + eval_every_n_steps + 1):
                    # with gradient accumulation, cur_step is virtual step
                    batch = next(train_ds)
                    next_update_step = (cur_step + FLAGS.grad_acc_steps - 1) // FLAGS.grad_acc_steps * FLAGS.grad_acc_steps
                    cur_epoch[:] = next_update_step / steps_per_epoch
                    apply_updates = (cur_step % FLAGS.grad_acc_steps) == 0
                    monitors = self.train_op_parallel(
                        batch['images'],
                        batch['labels'],
                        cur_epoch,
                        apply_updates)
                elapsed_train_time = time.time() - start_time
                total_training_time += elapsed_train_time
                train_time_per_epoch = total_training_time / cur_epoch[0]
                # eval
                start_time = time.time()
                accuracy = self.run_eval()
                elapsed_eval_time = time.time() - start_time
            if not FLAGS.disable_dp:
                dp_epsilon = objax.privacy.dpsgd.analyze_dp(
                    q=self.total_batch_size * FLAGS.grad_acc_steps / self.train_split.num_examples,
                    noise_multiplier=FLAGS.dp_sigma * np.sqrt(self.total_num_replicas * FLAGS.grad_acc_steps),
                    steps=int(cur_step / FLAGS.grad_acc_steps),
                    delta=FLAGS.dp_delta)
            else:
               dp_epsilon = None
            if self.save_summaries:
                # In multi-host setup only first host saves summaries and checkpoints.
                if jax.host_id() == 0:
                    # save summary
                    summary = objax.jaxboard.Summary()
                    for k, v in monitors.items():
                        summary.scalar(f'train/{k}', v)
                    if dp_epsilon is not None:
                        summary.scalar('dp/epsilon', dp_epsilon)
                        summary.scalar('dp/delta', FLAGS.dp_delta)
                    summary.scalar('test/accuracy', accuracy * 100)
                    self.summary_writer.write(summary, step=int(cur_step / FLAGS.grad_acc_steps))
                    # save checkpoint
                    checkpoint.save(self.all_vars, cur_step)
            # print info
            print(f'Step {cur_step} -- '
                  f'Epoch {cur_step / steps_per_epoch:.2f} -- '
                  f'Loss {monitors["total_loss"]:.2f}  '
                  f'Accuracy {accuracy * 100:.2f}')
            if dp_epsilon is not None:
                print(f'    DP: ε={dp_epsilon:.2f}  δ={FLAGS.dp_delta}')
            print(f'    Training took {elapsed_train_time:.1f} seconds, '
                  f'eval took {elapsed_eval_time:.1f} seconds')
            print(f'    Total training time so far {total_training_time:.1f} seconds, '
                  f'avg {train_time_per_epoch:.2f} seconds per epoch', flush=True)


def main(argv):
    del argv
    print('JAX host: %d / %d' % (jax.process_index(), jax.process_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()),
          flush=True)
    experiment = Experiment()
    experiment.train_and_eval()
    objax.util.multi_host_barrier()


if __name__ == '__main__':
    logging.set_verbosity(logging.ERROR)
    jax.config.config_with_absl()
    app.run(main)
