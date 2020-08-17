import argparse
import importlib
import models
import os
import tensorflow as tf
import sys
import h5py
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)  # model
sys.path.append('./utils')
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
from tf_util import chamfer,mlp
from models import mmd

parser = argparse.ArgumentParser()
parser.add_argument('--loss_type', type=str, default='cd_1')
parser.add_argument('--lr_decay_epochs', type=int, default=40)
parser.add_argument('--rec_weight',type=float, default=200.0)
parser.add_argument('--gp_weight',type=float, default=1)
parser.add_argument('--mmd_kernel',  type=str, default='mix_rq_1dot')
parser.add_argument('--gen_iter',type=int, default=1)
parser.add_argument('--dis_iter',type=int, default=1)
parser.add_argument('--log_dir', default='log/test')
parser.add_argument('--feat_weight',type=float, default=1000)
parser.add_argument('--num_gpus', type=int, default=1, help='How many gpus to use [default: 1]')
parser.add_argument('--gpu', default='1')
parser.add_argument('--pretrain_complete_decoder', default='pretrained_models/auto_encoder/2048_complete_4/model')
parser.add_argument('--step_ratio', type=int, default=4)
parser.add_argument('--num_gt_points', type=int, default=2048)
parser.add_argument('--h5_train',default='data/train_data.h5')
parser.add_argument('--h5_validate',default='data/valid_data.h5')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--restore', action='store_true')  # default=False
parser.add_argument('--testing', action='store_true')
parser.add_argument('--steps_per_eval', type=int, default=1000)
parser.add_argument('--steps_per_print', type=int, default=1000)
parser.add_argument('--model_type', default='model')
parser.add_argument('--base_lr_d', type=float, default=0.0001)
parser.add_argument('--base_lr_g', type=float, default=0.0001)
parser.add_argument('--best_loss',type=float,  default=1000)
parser.add_argument('--lr_decay', default=True)
parser.add_argument('--num_input_points', type=int, default=2048)
parser.add_argument('--lr_decay_steps', type=int, default=50000)
parser.add_argument('--fine_step', default=[10000, 20000, 50000])
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--lr_clip', type=float, default=1e-6)
args = parser.parse_args()

if args.num_gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.lr_decay_steps = int(args.lr_decay_steps / (32.0 / args.batch_size))
args.fine_step = [int(x * 32.0 / args.batch_size) for x in args.fine_step]

assert (args.batch_size % args.num_gpus == 0)
DEVICE_BATCH_SIZE = args.batch_size / args.num_gpus
NUM_GPUS = args.num_gpus

args.log_dir = args.log_dir + str(args.step_ratio)
BN_DECAY_DECAY_STEP = float(args.lr_decay_steps)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        # for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step//(args.gen_iter+args.dis_iter), args.fine_step, [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    inputs_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_input_points, 3), 'inputs')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'gt')
    complete_feature = tf.placeholder(tf.float32, (args.batch_size, 1024), 'complete_feature')
    complete_feature0 = tf.placeholder(tf.float32, (args.batch_size, 256), 'complete_feature0')
    label_pl = tf.placeholder(tf.int32, shape=(args.batch_size))

    model_module = importlib.import_module('.%s' % args.model_type, 'models')

    file_train = h5py.File(args.h5_train, 'r')
    incomplete_pcds_train = file_train['incomplete_pcds'][()]
    complete_pcds_train = file_train['complete_pcds'][()]
    labels_train = file_train['labels'][()]
    if args.num_gt_points==2048:
        if args.step_ratio==2:
            complete_features_train = file_train['complete_feature'][()]
            complete_features_train0 = file_train['complete_feature0'][()]
        elif args.step_ratio==4:
            complete_features_train = file_train['complete_feature1_4'][()]
            complete_features_train0 = file_train['complete_feature0_4'][()]
        elif args.step_ratio==8:
            complete_features_train = file_train['complete_feature1_8'][()]
            complete_features_train0 = file_train['complete_feature0_8'][()]
        elif args.step_ratio==16:
            complete_features_train = file_train['complete_feature1_16'][()]
            complete_features_train0 = file_train['complete_feature0_16'][()]
    elif args.num_gt_points==16384:
        file_train_feature = h5py.File(args.pretrain_complete_decoder + '/train_complete_feature.h5', 'r')
        complete_features_train = file_train_feature['complete_feature1'][()]
        complete_features_train0 = file_train_feature['complete_feature0'][()]
        file_train_feature.close()
    file_train.close()

    assert complete_features_train.shape[0]==complete_features_train0.shape[0]==incomplete_pcds_train.shape[0]
    assert complete_features_train.shape[1] ==1024
    assert complete_features_train0.shape[1] == 256

    train_num = incomplete_pcds_train.shape[0]
    BN_DECAY_DECAY_STEP = int(train_num / args.batch_size * args.lr_decay_epochs)

    learning_rate_d = tf.train.exponential_decay(args.base_lr_d, global_step//(args.gen_iter+args.dis_iter),
                                               BN_DECAY_DECAY_STEP, args.lr_decay_rate,
                                               staircase=True, name='lr_d')
    learning_rate_d = tf.maximum(learning_rate_d, args.lr_clip)

    learning_rate_g = tf.train.exponential_decay(args.base_lr_g, global_step//(args.gen_iter+args.dis_iter),
                                                 BN_DECAY_DECAY_STEP, args.lr_decay_rate,
                                                 staircase=True, name='lr_g')
    learning_rate_g = tf.maximum(learning_rate_g, args.lr_clip)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=0.9)
        D_optimizers = tf.train.AdamOptimizer(learning_rate_d, beta1=0.5)

    coarse_gpu=[]
    fine_gpu=[]
    tower_grads_g = []
    tower_grads_d = []
    total_dis_loss_gpu = []
    total_gen_loss_gpu = []
    total_lossReconstruction_gpu = []
    total_lossFeature_gpu=[]

    for i in range(NUM_GPUS):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.device('/gpu:%d' % (i)), tf.name_scope('gpu_%d' % (i)) as scope:
                inputs_pl_batch = tf.slice(inputs_pl, [int(i * DEVICE_BATCH_SIZE), 0, 0],[int(DEVICE_BATCH_SIZE), -1, -1])
                gt_pl_batch = tf.slice(gt_pl, [int(i * DEVICE_BATCH_SIZE), 0, 0], [int(DEVICE_BATCH_SIZE), -1, -1])
                complete_feature_batch = tf.slice(complete_feature, [int(i * DEVICE_BATCH_SIZE), 0], [int(DEVICE_BATCH_SIZE), -1])
                complete_feature_batch0 = tf.slice(complete_feature0, [int(i * DEVICE_BATCH_SIZE), 0],[int(DEVICE_BATCH_SIZE), -1])

                with tf.variable_scope('generator'):
                    features_partial_0,partial_reconstruct=model_module.encoder(inputs_pl_batch, embed_size=1024)
                    coarse_batch, fine_batch = model_module.decoder(inputs_pl_batch,
                                                                  partial_reconstruct,
                                                                  step_ratio=args.step_ratio,
                                                                  num_fine=args.step_ratio * 1024)

                with tf.variable_scope('discriminator') as dis_scope:
                    errD_fake =mlp(tf.expand_dims(partial_reconstruct, axis=1), [16], bn=None, bn_params=None)
                    dis_scope.reuse_variables()
                    errD_real = mlp(tf.expand_dims(complete_feature_batch, axis=1), [16], bn=None, bn_params=None)

                    kernel = getattr(mmd, '_%s_kernel' % args.mmd_kernel)
                    kerGI = kernel(errD_fake[:,0,:], errD_real[:,0,:])
                    errG=mmd.mmd2(kerGI)
                    errD=-errG
                    epsilon = tf.random_uniform([], 0.0, 1.0)
                    x_hat = complete_feature_batch * (1 - epsilon) + epsilon * partial_reconstruct
                    d_hat = mlp(tf.expand_dims(x_hat, axis=1), [16], bn=None, bn_params=None)
                    Ekx = lambda yy: tf.reduce_mean(kernel(d_hat[:,0,:], yy, K_XY_only=True), axis=1)
                    Ekxr, Ekxf = Ekx(errD_real[:,0,:]), Ekx(errD_fake[:,0,:])
                    witness = Ekxr - Ekxf
                    gradients = tf.gradients(witness, [x_hat])[0]
                    penalty = tf.reduce_mean(tf.square(mmd.safer_norm(gradients, axis=1) - 1.0))
                    errD_loss_batch=penalty*args.gp_weight+errD
                    errG_loss_batch = errG

                feature_loss = tf.reduce_mean(tf.squared_difference(partial_reconstruct,complete_feature_batch))+ \
                               tf.reduce_mean(tf.squared_difference(features_partial_0[:,0,:], complete_feature_batch0))

                dist1_fine, dist2_fine = chamfer(fine_batch, gt_pl_batch)
                dist1_coarse, dist2_coarse = chamfer(coarse_batch, gt_pl_batch)
                total_loss_fine = (tf.reduce_mean(tf.sqrt(dist1_fine)) + tf.reduce_mean(tf.sqrt(dist2_fine))) / 2
                total_loss_coarse = (tf.reduce_mean(tf.sqrt(dist1_coarse)) + tf.reduce_mean(tf.sqrt(dist2_coarse))) / 2
                total_loss_rec_batch = alpha * total_loss_fine + total_loss_coarse

                t_vars = tf.global_variables()
                gen_tvars = [var for var in t_vars if var.name.startswith("generator")]
                dis_tvars = [var for var in t_vars if var.name.startswith("discriminator")]
                total_gen_loss_batch = args.feat_weight*feature_loss+errG_loss_batch+args.rec_weight*total_loss_rec_batch
                total_dis_loss_batch = errD_loss_batch

                # Calculate the gradients for the batch of data on this tower.
                grads_g = G_optimizers.compute_gradients(total_gen_loss_batch, var_list=gen_tvars)
                grads_d = D_optimizers.compute_gradients(total_dis_loss_batch, var_list=dis_tvars)

                # Keep track of the gradients across all towers.
                tower_grads_g.append(grads_g)
                tower_grads_d.append(grads_d)

                coarse_gpu.append(coarse_batch)
                fine_gpu.append(fine_batch)

                total_dis_loss_gpu.append(total_dis_loss_batch)
                total_gen_loss_gpu.append(errG_loss_batch)
                total_lossReconstruction_gpu.append(args.rec_weight*total_loss_rec_batch)
                total_lossFeature_gpu.append(args.feat_weight*feature_loss)

    fine = tf.concat(fine_gpu, 0)

    grads_g = average_gradients(tower_grads_g)
    grads_d = average_gradients(tower_grads_d)

    # apply the gradients with our optimizers
    train_G = G_optimizers.apply_gradients(grads_g, global_step=global_step)
    train_D = D_optimizers.apply_gradients(grads_d, global_step=global_step)

    total_dis_loss = tf.reduce_mean(tf.stack(total_dis_loss_gpu, 0))
    total_gen_loss = tf.reduce_mean(tf.stack(total_gen_loss_gpu, 0))
    total_loss_rec = tf.reduce_mean(tf.stack(total_lossReconstruction_gpu, 0))
    total_loss_fea = tf.reduce_mean(tf.stack(total_lossFeature_gpu, 0))

    dist1_eval, dist2_eval = chamfer(fine, gt_pl)

    file_validate = h5py.File(args.h5_validate, 'r')
    incomplete_pcds_validate = file_validate['incomplete_pcds'][()]
    complete_pcds_validate = file_validate['complete_pcds'][()]
    labels_validate = file_validate['labels'][()]
    file_validate.close()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=3)
    sess.run(tf.global_variables_initializer())

    saver_decoder = tf.train.Saver(var_list=[var for var in tf.global_variables() if (var.name.startswith("generator/decoder") \
                                                                                      or var.name.startswith("generator/folding"))])
    saver_decoder.restore(sess, args.pretrain_complete_decoder)

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))

    init_step = sess.run(global_step)//(args.gen_iter+args.dis_iter)
    epoch = init_step * args.batch_size // train_num + 1
    print('init_step:%d,' % init_step, 'epoch:%d' % epoch,'training data number:%d'%train_num)
    train_idxs = np.arange(0, train_num)

    for ep_cnt in range(epoch, args.max_epoch + 1):
        num_batches = train_num // args.batch_size
        np.random.shuffle(train_idxs)

        for batch_idx in range(num_batches):
            init_step += 1
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, train_num)
            ids_train = list(np.sort(train_idxs[start_idx:end_idx]))
            batch_data = incomplete_pcds_train[ids_train]
            batch_gt = complete_pcds_train[ids_train]
            labels = labels_train[ids_train]
            # partial_feature_input=incomplete_features_train[ids_train]
            complete_feature_input=complete_features_train[ids_train]
            complete_feature_input0 = complete_features_train0[ids_train]

            feed_dict = {inputs_pl: batch_data, gt_pl: batch_gt, is_training_pl: True, label_pl: labels, complete_feature:complete_feature_input,
                         complete_feature0:complete_feature_input0}

            for i in range(args.dis_iter):
                _, loss_dis = sess.run([train_D, total_dis_loss], feed_dict=feed_dict)
            for i in range(args.gen_iter):
                _, loss_gen, rec_loss,fea_loss = sess.run([train_G, total_gen_loss, total_loss_rec, total_loss_fea], feed_dict=feed_dict)

            if init_step % args.steps_per_print == 0:
                print('epoch %d step %d gen_loss %.8f rec_loss %.8f fea_loss %.8f dis_loss %.8f' % \
                    (ep_cnt, init_step, loss_gen,rec_loss,fea_loss,loss_dis))

            if init_step % args.steps_per_eval == 0:
                total_loss = 0
                sess.run(tf.local_variables_initializer())
                batch_data = np.zeros((args.batch_size, incomplete_pcds_validate[0].shape[0], 3), 'f')
                batch_gt = np.zeros((args.batch_size, args.num_gt_points, 3), 'f')
                # partial_feature_input = np.zeros((args.batch_size, 1024), 'f')
                labels = np.zeros((args.batch_size,), dtype=np.int32)
                feature_complete_input = np.zeros((args.batch_size, 1024)).astype(np.float32)
                feature_complete_input0 = np.zeros((args.batch_size, 256)).astype(np.float32)
                for batch_idx_eval in range(0, incomplete_pcds_validate.shape[0], args.batch_size):
                    # start = time.time()
                    start_idx = batch_idx_eval
                    end_idx = min(start_idx + args.batch_size, incomplete_pcds_validate.shape[0])

                    batch_data[0:end_idx - start_idx] = incomplete_pcds_validate[start_idx:end_idx]
                    batch_gt[0:end_idx - start_idx] = complete_pcds_validate[start_idx:end_idx]
                    labels[0:end_idx - start_idx]=labels_validate[start_idx:end_idx]
                    feature_complete_input[0:end_idx - start_idx] = complete_features_train[start_idx:end_idx]
                    feature_complete_input0[0:end_idx - start_idx] = complete_features_train0[start_idx:end_idx]

                    feed_dict = {inputs_pl: batch_data, gt_pl: batch_gt, is_training_pl: False,label_pl: labels,complete_feature:feature_complete_input,complete_feature0:feature_complete_input0}
                    dist1_out, dist2_out = sess.run([dist1_eval, dist2_eval], feed_dict=feed_dict)
                    if args.loss_type == 'cd_1':
                        total_loss += np.mean(dist1_out[0:end_idx - start_idx]) * (end_idx - start_idx) \
                                      + np.mean(dist2_out[0:end_idx - start_idx]) * (end_idx - start_idx)
                    elif args.loss_type == 'cd_2':
                        total_loss += (np.mean(np.sqrt(dist1_out[0:end_idx - start_idx])) * (end_idx - start_idx) \
                                       + np.mean(np.sqrt(dist2_out[0:end_idx - start_idx])) * (end_idx - start_idx)) / 2

                if total_loss / incomplete_pcds_validate.shape[0] < args.best_loss:
                    args.best_loss = total_loss / incomplete_pcds_validate.shape[0]
                    saver.save(sess, os.path.join(args.log_dir, 'model'), init_step)

                print('epoch %d  step %d  loss %.8f best_loss %.8f' % (ep_cnt, init_step, total_loss / incomplete_pcds_validate.shape[0], args.best_loss))
    sess.close()

if __name__ == '__main__':
    train(args)
