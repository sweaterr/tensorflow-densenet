# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tarfile
import glob
import numpy as np

import tensorflow as tf
import urllib
from scipy.io import loadmat

_NUM_SHARDS = 5

from datasets import dataset_utils

labels = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells','sweet pea',
          'english marigold','tiger lily','moon orchid','bird of paradise','monkshood',
          'globe thistle','snapdragon',"colt's foot",'king protea','spear thistle',
          'yellow iris','globe-flower','purple coneflower','peruvian lily',
          'balloon flower','giant white arum lily','fire lily', 'pincushion flower',
          'fritillary', 'red ginger', 'grape hyacinth','corn poppy', 'prince of wales feathers',
          'stemless gentian', 'artichoke', 'sweet william', 'carnation',
          'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
          'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
          'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
          'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
          'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
          'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
          'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone',
          'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum',
          'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania',
          'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower',
          'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine',
          'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily',
          'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
          'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  flower_root = os.path.join(dataset_dir, 'flower_photos')
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'flowers102_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, file_and_classid, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(file_and_classid) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(file_and_classid))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(file_and_classid), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(file_and_classid[i][0], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_id = int(file_and_classid[i][1])

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()



def download_file(url, dataset_dir, dest=None):
  if not dest:
    dest = dataset_dir + url.split('/')[-1]
  if not os.path.exists(dest):
    urllib.urlretrieve(url, dest)

def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  print("Downloading images...")
  download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', dataset_dir)
  tarfile.open(dataset_dir + "/102flowers.tgz").extractall(path=dataset_dir)

  print("Downloading image labels...")
  download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat', dataset_dir)

  print("Downloading train/test/valid splits...")
  download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat', dataset_dir)

  # Read .mat file containing training, testing, and validation sets.
  setid = loadmat(dataset_dir + '/setid.mat')

  # The .mat file is 1-indexed, so we subtract one to match Caffe's convention.
  # train <-> test swap
  idx_train = setid['tstid'][0] - 1
  idx_test = setid['trnid'][0] - 1
  idx_valid = setid['valid'][0] - 1

  # Read .mat file containing image labels.
  image_labels = loadmat(dataset_dir + '/imagelabels.mat')['labels'][0]

  files = sorted(glob.glob(dataset_dir + '/jpg/*.jpg'))

  # Images are ordered by species, so shuffle them
  np.random.seed(777)
  idx_train = idx_train[np.random.permutation(len(idx_train))]
  idx_test = idx_test[np.random.permutation(len(idx_test))]
  # idx_valid = idx_valid[np.random.permutation(len(idx_valid))]

  labels = np.array(list(zip(files, image_labels)))

  # Divide into train and test:
  # random.seed(_RANDOM_SEED)
  # random.shuffle(photo_filenames)
  # training_filenames = photo_filenames[_NUM_VALIDATION:]
  # validation_filenames = photo_filenames[:_NUM_VALIDATION]
  train_set = labels[idx_train, :]
  test_set = labels[idx_test, :]

  # First, convert the training and validation sets.
  _convert_dataset('train', train_set, dataset_dir)
  _convert_dataset('validation', test_set,dataset_dir)

  # Finally, write the labels file:
  # labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  # _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers dataset!')

