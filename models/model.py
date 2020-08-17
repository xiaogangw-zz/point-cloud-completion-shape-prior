import tensorflow as tf
import numpy as np
from tf_util import mlp, mlp_conv,gen_grid, gen_1d_grid
from tf_sampling import farthest_point_sample, gather_point

def encoder(inputs, embed_size=1024):
    with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
        features = mlp_conv(inputs, [128, 256], bn=None, bn_params=None)
        features_global = tf.reduce_max(features, axis=1, keepdims=True, name='maxpool_0')
        features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
    with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
        features = mlp_conv(features, [512, embed_size], bn=None, bn_params=None)
        features = tf.reduce_max(features, axis=1, name='maxpool_1')
    return features_global,features

def symmetric_sample(points, num):
    p1_idx = farthest_point_sample(num, points)
    input_fps = gather_point(points, p1_idx)
    input_fps_flip = tf.concat(
        [tf.expand_dims(input_fps[:, :, 0], axis=2), tf.expand_dims(input_fps[:, :, 1], axis=2),
         tf.expand_dims(-input_fps[:, :, 2], axis=2)], axis=2)
    input_fps = tf.concat([input_fps, input_fps_flip], 1)
    return input_fps

def decoder(inputs, features, step_ratio=16, num_fine=16 * 1024):
    num_coarse=1024
    assert num_fine == num_coarse * step_ratio
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        coarse = mlp(features, [1024, 1024, num_coarse * 3], bn=None, bn_params=None)
        coarse = tf.reshape(coarse, [-1, num_coarse, 3])

    p1_idx = farthest_point_sample(512, coarse)
    coarse_1 = gather_point(coarse, p1_idx)
    input_fps = symmetric_sample(inputs, int(512 / 2))
    coarse = tf.concat([input_fps, coarse_1], 1)

    with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
        if not step_ratio ** .5 % 1 == 0:
            grid = gen_1d_grid(step_ratio)
        else:
            grid = gen_grid(np.round(np.sqrt(step_ratio)).astype(np.int32))
        grid = tf.expand_dims(grid, 0)
        grid_feat = tf.tile(grid, [features.shape[0], num_coarse, 1])
        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, step_ratio, 1])
        point_feat = tf.reshape(point_feat, [-1, num_fine, 3])
        global_feat = tf.tile(tf.expand_dims(features, 1), [1, num_fine, 1])
        feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)
        fine = mlp_conv(feat, [512, 512, 3], bn=None, bn_params=None) + point_feat
    return coarse, fine