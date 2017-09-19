# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images in CIFAR-10.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


slim = tf.contrib.slim

def preprocess_for_train(image,
                         output_height,
                         output_width):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    padding: The amound of padding before and after each dimension of the image.

  Returns:
    A preprocessed image.
  """
  tf.summary.image('image', tf.expand_dims(image, 0))
  image = tf.to_float(image)
  image = tf.div(image, 255.0)

  # Subtract off the mean and divide by the variance of the pixels.
  return image


def preprocess_for_eval(image, output_height, output_width):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
  tf.summary.image('image', tf.expand_dims(image, 0))
  # Transform the image to floats.
  image = tf.to_float(image)
  image = tf.div(image, 255.0)

  # Subtract off the mean and divide by the variance of the pixels.
  return image


def preprocess_image(image, output_height, output_width, is_training=False):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, output_height, output_width)
  else:
    return preprocess_for_eval(image, output_height, output_width)
