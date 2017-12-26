import numpy as np
import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    # fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)
    fc_h = tf.matmul(inpt, fc_w) + fc_b
    return fc_h


def conv_layer_res(inpt, filter_shape, stride, is_training):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)

    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")

    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training)
    # batch_norm = tf.contrib.layers.batch_norm(conv)
    out = tf.nn.relu(batch_norm)

    return out


def residual_block(inpt, output_depth, down_sample, projection=False, is_training=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1, 2, 2, 1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer_res(inpt, [3, 3, input_depth, output_depth], 1, is_training)
    conv2 = conv_layer_res(conv1, [3, 3, output_depth, output_depth], 1, is_training)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer_res(inpt, [1, 1, input_depth, output_depth], 2, is_training)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    return res