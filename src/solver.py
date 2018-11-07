# ---------------------------------------------------------
# Tensorflow VAE Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

# noinspection PyPep8Naming
from dataset import Dataset
from vae import VAE
import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_time = 0
        self._make_folders()
        self._init_logger()

        self.dataset = Dataset(self.sess, self.flags, self.flags.dataset, log_path=self.log_out_dir)
        self.model = VAE(self.sess, self.flags, self.dataset, log_path=self.log_out_dir)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "{}/model/{}_{}_{}".format(self.flags.dataset, cur_time,
                                                                str(self.flags.add_noise), str(self.flags.z_dim))
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}_{}_{}".format(self.flags.dataset, cur_time,
                                                                str(self.flags.add_noise), str(self.flags.z_dim))

            self.sample_out_dir = "{}/sample/{}_{}_{}".format(self.flags.dataset, cur_time,
                                                              str(self.flags.add_noise), str(self.flags.z_dim))
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.log_out_dir = "{}/logs/{}_{}_{}".format(self.flags.dataset, cur_time,
                                                         str(self.flags.add_noise), str(self.flags.z_dim))
            self.train_writer = tf.summary.FileWriter("{}/logs/{}_{}_{}".format(
                self.flags.dataset, cur_time, str(self.flags.add_noise), str(self.flags.z_dim)),
                                                      graph_def=self.sess.graph_def)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model,)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, self.flags.load_model)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def _init_logger(self):
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(self.log_out_dir, 'solver.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        if self.flags.is_train:
            logger.info('gpu_index: {}'.format(self.flags.gpu_index))
            logger.info('batch_size: {}'.format(self.flags.batch_size))
            logger.info('dataset: {}'.format(self.flags.dataset))

            logger.info('is_train: {}'.format(self.flags.is_train))
            logger.info('add_noise: {}'.format(self.flags.add_noise))
            logger.info('learning_rate: {}'.format(self.flags.learning_rate))
            logger.info('z_dim: {}'.format(self.flags.z_dim))

            logger.info('iters: {}'.format(self.flags.iters))
            logger.info('print_freq: {}'.format(self.flags.print_freq))
            logger.info('save_freq: {}'.format(self.flags.save_freq))
            logger.info('sample_freq: {}'.format(self.flags.sample_freq))
            logger.info('sample_size: {}'.format(self.flags.sample_batch))
            logger.info('load_model: {}'.format(self.flags.load_model))

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                logger.info(' [*] Load SUCCESS!\n')
            else:
                logger.info(' [!] Load failed...\n')

        while self.iter_time < self.flags.iters:
            # samppling images and save them
            self.sample(self.iter_time)

            # next batch
            batch_imgs, batch_labels = self.dataset.train_next_batch(batch_size=self.flags.batch_size)

            # train_step
            loss, summary = self.model.train_step(batch_imgs)
            self.model.print_info(loss, self.iter_time)
            self.train_writer.add_summary(summary, self.iter_time)
            self.train_writer.flush()

            # save model
            self.save_model(self.iter_time)
            self.iter_time += 1

        self.save_model(self.flags.iters)

    def test(self):
        if self.load_model():
            logger.info(' [*] Load SUCCESS!')
        else:
            logger.info(' [!] Load Failed...')

        num_iters = 20
        for iter_time in range(num_iters):
            print('iter_time: {}'.format(iter_time))
            # next batch
            batch_imgs, batch_labels = self.dataset.test_next_batch(batch_size=self.flags.sample_batch)
            y_imgs, y_noise = self.model.sample_imgs(batch_imgs)

            self.model.plots(batch_imgs, iter_time, self.test_out_dir, prefix='input')  # clean input images
            self.model.plots(y_imgs, iter_time, self.test_out_dir, prefix='recon')  # decoded images
            if self.flags.add_noise is True:
                self.model.plots(y_noise, iter_time, self.test_out_dir, prefix='noise')  # noise input images

            if self.flags.z_dim == 2:
                self.latent_vector_walk(iter_time, save_file=self.test_out_dir)
                if self.flags.dataset == 'mnist':
                    self.embedding(iter_time, save_file=self.test_out_dir)

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            batch_imgs, _ = self.dataset.train_next_batch(batch_size=self.flags.sample_batch)
            y_imgs, y_noise = self.model.sample_imgs(batch_imgs)

            self.model.plots(batch_imgs, iter_time, self.sample_out_dir, prefix='input')  # clean input images
            self.model.plots(y_imgs, iter_time, self.sample_out_dir, prefix='recon')  # decoded images
            if self.flags.add_noise is True:
                self.model.plots(y_noise, iter_time, self.sample_out_dir, prefix='noise')  # noise input images

            if self.flags.z_dim == 2:
                self.latent_vector_walk(iter_time, save_file=self.sample_out_dir)
                if self.flags.dataset == 'mnist':
                    self.embedding(iter_time, save_file=self.sample_out_dir)

    def latent_vector_walk(self, iter_time, save_file=None):
        z_range = 1.5 + np.random.randn()
        # z_range = 2.
        z_vectors = np.rollaxis(np.mgrid[z_range:-z_range:20 * 1j, z_range:-z_range:20 * 1j], 0, 3).reshape([-1, 2])
        y_decodes = self.model.decoder_y(z_vectors)
        self.model.plots(y_decodes, iter_time, save_file, prefix='latent')  # noise input images

    def embedding(self, iter_time, save_file=None):
        batch_imgs, batch_labels = self.dataset.test_next_batch(batch_size=5000)
        z_embed = self.model.decoder_z(batch_imgs)
        utils.save_scattered_image(z_embed, batch_labels, iter_time, save_file)

    def save_model(self, iter_time):
        if np.mod(iter_time, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)
            logger.info('[*] Model saved! Iter: {}'.format(iter_time))

    def load_model(self):
        logger.info(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            logger.info('[*] Load iter_time: {}'.format(self.iter_time))
            return True
        else:
            return False
