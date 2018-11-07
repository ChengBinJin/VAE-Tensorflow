# ---------------------------------------------------------
# Tensorflow VAE Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size, default: 128')
tf.flags.DEFINE_string('dataset', 'mnist', 'dataset name from [mnist, freyface], default: mnist')

tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_bool('add_noise', False, 'boolean for adding salt & pepper noise to input image, default: False')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for Adam, default: 0.001')
tf.flags.DEFINE_integer('z_dim', 20, 'dimension of z vector, default: 20')

tf.flags.DEFINE_integer('iters', 20000, 'number of iterations, default: 20000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 5000, 'save frequency for model, default: 5000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_batch', 100, 'number of sampling images for check generator quality, default: 100')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                       '(e.g. 20181107-2106_False_2), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
