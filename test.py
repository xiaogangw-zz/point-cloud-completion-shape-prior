import argparse
import importlib
import models
import numpy as np
import sys
import tensorflow as tf
import h5py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append('./utils')
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
from tf_util import chamfer

objects = ['plane', 'cabinet', 'car', 'chair', 'lamp', 'couch', 'table', 'watercraft', 'speaker', 'firearm', 'cellphone', 'bench', 'monitor']
snc_synth_id_to_category = {
    '02691156': 'plane',
    '02933112': 'cabinet',
    '02958343': 'car',
    '03001627': 'chair',
    '03636649': 'lamp',
    '04256520': 'couch',
    '04379243': 'table',
    '04530566': 'watercraft',

    '02818832':'bed',
    '02828884':'bench',
    '02871439':'bookshelf',
    '02924116':'bus',
    '03467517':'guitar',
    '03790512':'motorbike',
    '03948459':'pistol',
    '04225987':'skateboard'
}

def test(args):
    inputs = tf.placeholder(tf.float32, (1, 2048, 3))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    reconstruction = tf.placeholder(tf.float32, (1, 1024*args.step_ratio, 3))
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    model_module = importlib.import_module('.%s' % args.model_type, 'models')

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        _,features_partial = model_module.encoder(inputs)
        coarse, fine = model_module.decoder(inputs, features_partial, args.step_ratio, args.step_ratio * 1024)

    dist1_fine, dist2_fine = chamfer(reconstruction, gt)
    if args.loss_type=='cd_1':
        loss = tf.reduce_mean(dist1_fine) + tf.reduce_mean(dist2_fine)
    elif args.loss_type=='cd_2':
        loss = (tf.reduce_mean(tf.sqrt(dist1_fine)) + tf.reduce_mean(tf.sqrt(dist2_fine))) / 2

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(args.checkpoint))

    data_all=h5py.File(args.data_dir,'r')
    partial_all=data_all['incomplete_pcds'][()]
    complete_all=data_all['complete_pcds'][()]
    model_list = data_all['labels'][()].astype(int)
    data_all.close()

    cd_per_cat = {}

    total_cd = 0
    for i, model_id in enumerate(model_list):
        partial = partial_all[i]
        complete = complete_all[i]

        label=model_list[i]

        completion = sess.run(fine, feed_dict={inputs: [partial], is_training_pl: False})
        cd = sess.run(loss,feed_dict={reconstruction: completion, gt: [complete], is_training_pl: False})
        total_cd += cd

        category=objects[label]
        key_list = list(snc_synth_id_to_category.keys())
        val_list = list(snc_synth_id_to_category.values())
        synset_id=key_list[val_list.index(category)]
        if not cd_per_cat.get(synset_id):
            cd_per_cat[synset_id] = []
        cd_per_cat[synset_id].append(cd)

    print('Average Chamfer distance: %f' % (total_cd / len(model_list)))
    print('Chamfer distance per category')
    for synset_id in sorted(cd_per_cat.keys()):
        print(synset_id, '%f' % np.mean(cd_per_cat[synset_id]))

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--step_ratio', type=int, default=16)
    parser.add_argument('--loss_type', default='cd_2')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--model_type', default='model')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test(args)


