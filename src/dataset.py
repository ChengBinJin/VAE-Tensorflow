# ---------------------------------------------------------
# Tensorflow VAE Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


def _init_logger(flags, log_path):
    if flags.is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(log_path, 'dataset.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


class MnistDataset(object):
    def __init__(self, sess, flags, dataset_name):
        self.sess = sess
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (32, 32, 1)
        self.img_buffle = 100000  # image buffer for image shuflling
        self.num_trains, self.num_tests = 0, 0

        self.mnist_path = os.path.join('../../Data', self.dataset_name)
        self._load_mnist()

    def _load_mnist(self):
        logger.info('Load {} dataset...'.format(self.dataset_name))
        self.train_data, self.test_data = tf.keras.datasets.mnist.load_data()
        # self.train_data is tuple: (image, label)
        self.num_trains = self.train_data[0].shape[0]
        self.num_tests = self.test_data[0].shape[0]

        # TensorFlow Dataset API
        train_x, train_y = self.train_data
        self.test_x, self.test_y = self.test_data
        dataset = tf.data.Dataset.from_tensor_slices(({'image': train_x}, train_y))
        dataset = dataset.shuffle(self.img_buffle).repeat().batch(self.flags.batch_size)

        iterator = dataset.make_one_shot_iterator()
        self.next_batch = iterator.get_next()

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))
        logger.info('Img size: {}'.format(self.image_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self, batch_size):
        batch_data = self.sess.run(self.next_batch)
        batch_imgs = batch_data[0]["image"]
        batch_labels = batch_data[1]

        if self.flags.batch_size > batch_size:
            # reshape 784 vector to 28 x 28 x 1
            batch_imgs = np.reshape(batch_imgs[:batch_size], [batch_size, 28, 28])
        else:
            batch_imgs = np.reshape(batch_imgs, [self.flags.batch_size, 28, 28])

        imgs_32 = [scipy.misc.imresize(batch_imgs[idx], self.image_size[0:2])
                   for idx in range(batch_imgs.shape[0])]  # scipy.misc.imresize convert to uint8 type
        imgs_array = np.expand_dims(np.asarray(imgs_32).astype(np.float32), axis=3)
        imgs_array = imgs_array / 255.  # from [0. 255.] to [0., 1.]

        return imgs_array, batch_labels

    def test_next_batch(self, batch_size):
        idxs = np.random.randint(low=0, high=self.num_tests, size=batch_size)
        batch_imgs, batch_labels = self.test_x[idxs], self.test_y[idxs]

        imgs_32 = [scipy.misc.imresize(batch_imgs[idx], self.image_size[0:2])
                   for idx in range(batch_imgs.shape[0])]  # scipy.misc.imresize convert to uint8 type
        imgs_array = np.expand_dims(np.asarray(imgs_32).astype(np.float32), axis=3)
        imgs_array = imgs_array / 255.  # from [0. 255.] to [0., 1.]

        # one-hot representations
        labels_array = np.zeros((batch_labels.shape[0], 10))
        labels_array[range(batch_labels.shape[0]), batch_labels] = 1

        return imgs_array, labels_array


class FreyFace(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (28, 20, 1)
        self.num_trains, self.num_tests = 0, 0

        self.freyface_path = os.path.join('../../Data', self.dataset_name)
        self._load_freyface()

    def _load_freyface(self):
        logger.info('Load {} dataset...'.format(self.dataset_name))

        self.data = scipy.io.loadmat(os.path.join(self.freyface_path, 'frey_rawface.mat'))
        self.data = self.data['ff']
        self.data = self.data.transpose()
        self.data = self.data.reshape((-1, *self.image_size))

        self.test_data = self.data[-100:, :, :, :]
        self.train_data = self.data[:-100, :, :, :]

        self.num_trains = self.train_data.shape[0]
        self.num_tests = self.test_data.shape[0]

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))
        logger.info('Img size: {}'.format(self.image_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self, batch_size):
        idxs = np.random.randint(low=0, high=self.num_trains, size=batch_size)
        batch_imgs = self.train_data[idxs]
        batch_imgs = batch_imgs / 255.  # from [0. 255.] to [0., 1.]

        return batch_imgs, None

    def test_next_batch(self, batch_size):
        idxs = np.random.randint(low=0, high=self.num_tests, size=batch_size)
        batch_imgs = self.test_data[idxs]
        batch_imgs = batch_imgs / 255.  # from [0. 255.] to [0., 1.]

        return batch_imgs, None


# noinspection PyPep8Naming
def Dataset(sess, flags, dataset_name, log_path=None):
    if flags.is_train:
        _init_logger(flags, log_path)  # init logger

    if dataset_name == 'mnist':
        return MnistDataset(sess, flags, dataset_name)
    elif dataset_name == 'freyface':
        return FreyFace(flags, dataset_name)
    else:
        raise NotImplementedError
