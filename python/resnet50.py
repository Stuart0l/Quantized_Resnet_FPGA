import tensorflow as tf
import numpy as np
import os
from imagenet_preprocessing import preprocess_image

block_sizes = [3, 4, 6, 3]
block_strides = [1, 2, 2, 2]


def batch_norm(inputs, param):

    gamma = tf.Variable(param[0], name='scale')
    beta = tf.Variable(param[1], name='offset')
    mu = tf.Variable(param[2], name='mean')
    sigma = tf.Variable(param[3], name='variance')

    return tf.nn.batch_normalization(x=inputs,
                                     scale=gamma,
                                     offset=beta,
                                     mean=mu,
                                     variance=sigma,
                                     variance_epsilon=1e-3)


def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    kernel_filter = tf.Variable(filters, name='filter')
    return tf.nn.conv2d(input=inputs,
                        filter=kernel_filter,
                        strides=[1, strides, strides, 1],
                        padding=('SAME' if strides == 1 else 'VALID'))


def bottleneck_block_v1(inputs, filters, projection_shortcut, strides):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut,
                              param=filters[3]['bn'])

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters[0]['conv'],
                                  kernel_size=1,
                                  strides=1)
    inputs = batch_norm(inputs, filters[0]['bn'])
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters[1]['conv'],
                                  kernel_size=3,
                                  strides=strides)
    inputs = batch_norm(inputs, filters[1]['bn'])
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters[2]['conv'],
                                  kernel_size=1,
                                  strides=1)
    inputs = batch_norm(inputs, filters[2]['bn'])
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def block_layer(inputs, filters, blocks, strides):
    def projection_shortcut(_input):
        return conv2d_fixed_padding(inputs=_input,
                                    filters=filters[0][3]['conv'],
                                    kernel_size=1,
                                    strides=strides)

    inputs = bottleneck_block_v1(inputs, filters[0], projection_shortcut, strides)

    for i in range(1, blocks):
        inputs = bottleneck_block_v1(inputs, filters[i], None, 1)

    return inputs


def resnet_50(inputs, weights):
    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=weights[0][0]['conv'],
                                  kernel_size=7,
                                  strides=2)

    inputs = batch_norm(inputs, weights[0][0]['bn'])
    inputs = tf.nn.relu(inputs)

    inputs = tf.nn.max_pool(value=inputs,
                            ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    for i, num_blocks in enumerate(block_sizes):
        inputs = block_layer(inputs=inputs,
                             filters=weights[0][i + 1],
                             blocks=num_blocks,
                             strides=block_strides[i])

    axes = [1, 2]
    inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)

    inputs = tf.squeeze(inputs, axes)
    inputs = tf.matmul(inputs, weights[1][0])
    inputs = tf.add(inputs, weights[1][1], name='output')

    return inputs
