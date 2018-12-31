# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import os

import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.platform import gfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

labels = [
    "cat",
    "dog"
]
label_reverse={
    "cat":0,
    "dog":1,
}

IMAGE_SIZE = 256
# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


class Image:
    name = ""
    label = ""


def generate_image_list(data_dir, _label_range, _range):
    #
    #   本函数生成所有待读取的图片列表
    #
    image_list = []

    for label_i in _label_range:
        for i in _range:
            image = os.path.join(data_dir, labels[label_i], '%d.jpg' % i)
            image_list += [image]

    for f in image_list:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    return image_list


def readImage(image_list, jpgReadResult, num_epochs):
    """
        读取文件并且返回

    """
    result = jpgReadResult()

    filename_queue = tf.train.string_input_producer(image_list)


    reader = tf.WholeFileReader()
    result.name, value = reader.read(filename_queue)


    result.name = tf.cast(result.name,tf.string)

    result.label = 0

    # 这里有一个作用就是根据文件名进行分类 真的很搞
    for i,label in enumerate(labels):
        prevValue = result.label
        cond = tf.strings.regex_full_match(result.name, (".*"+label+".*"))
        result.label=tf.cond(cond,lambda:i,lambda:prevValue)

    result.label = tf.cast(result.label, tf.int32)

    img_data_jpg = tf.image.decode_jpeg(value)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)


    result.uint8image = img_data_jpg

    return result


def generate_image_batch(image, min_queue_examples, batch_size, num_preprocess_threads):
    images, label_batch = tf.train.shuffle_batch(
        [image.image,image.label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        shapes=[tf.TensorShape([tf.Dimension(128), tf.Dimension(128), tf.Dimension(3)]),tf.TensorShape([])],
        min_after_dequeue=min_queue_examples)
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(
        data_dir,
        _range=xrange(1, 1000),
        _label_range=[0, 1],
        num_epochs=5,
        batch_size=32,
        _width=128,
        _height=128,
        _depth=3,
        num_preprocess_threads=16,
        min_fraction_of_examples_in_queue=0.4):
    min_queue_examples = int(num_epochs * min_fraction_of_examples_in_queue)

    class jpgReadResult:
        height = _height
        width = _width
        depth = _depth

    image_list = generate_image_list(data_dir, _label_range, _range)

    read_result = readImage(image_list, jpgReadResult, num_epochs)

    reshaped_image = tf.cast(read_result.uint8image, tf.float32)
    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    # Randomly crop a [height, width] section of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               _width, _height)

    
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(resized_image)
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)

    min_queue_examples = int(num_epochs *
                             min_fraction_of_examples_in_queue)

    image = Image()
    image.image = float_image
    image.name = read_result.name
    image.label = read_result.label


    return generate_image_batch(image,min_queue_examples, batch_size, num_preprocess_threads)


def inputs(
        data_dir,
        _range=xrange(1, 1000),
        _label_range=[0, 1],
        num_epochs=5,
        batch_size=128,
        _width=128,
        _height=128,
        _depth=3,
        num_preprocess_threads=16,
        min_fraction_of_examples_in_queue=0.4):
    min_queue_examples = int(num_epochs * min_fraction_of_examples_in_queue)

    class jpgReadResult:
        height = _height
        width = _width
        depth = _depth

    image_list = generate_image_list(data_dir, _label_range, _range)


    read_result = readImage(image_list, jpgReadResult, num_epochs)

    reshaped_image = tf.cast(read_result.uint8image, tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               _width, _height)

    float_image = tf.image.per_image_standardization(resized_image)


    image = Image()
    image.image = float_image
    image.name = read_result.name
    image.label = read_result.label

    return generate_image_batch(image, min_queue_examples, batch_size, num_preprocess_threads)
