import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pc_distance'))
from pc_distance import tf_nndistance

def mlp(features, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, num_outputs in enumerate(layer_dims[:-1]):
            features = tf.contrib.layers.fully_connected(
                features, num_outputs,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='fc_%d' % i)
        outputs = tf.contrib.layers.fully_connected(
            features, layer_dims[-1],
            activation_fn=None,
            scope='fc_%d' % (len(layer_dims) - 1))
    return outputs

def mlp_conv(inputs, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf.contrib.layers.conv1d(
                inputs, num_out_channel,
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_%d' % i)
        outputs = tf.contrib.layers.conv1d(
            inputs, layer_dims[-1],
            kernel_size=1,
            activation_fn=None,
            scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    return dist1 , dist2

def gen_grid(num_grid_point):
    x = tf.linspace(-0.05, 0.05, num_grid_point)
    x, y = tf.meshgrid(x, x)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])
    return grid

def gen_1d_grid(num_grid_point):
    x = tf.linspace(-0.05, 0.05, num_grid_point)
    grid = tf.reshape(x, [-1, 1])
    return grid