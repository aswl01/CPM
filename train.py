import argparse
import codecs
import os
import sys
import time
import tensorflow as tf
from data_utils import *
import numpy as np
from model.cpm_network import CPM_NETWORK
import scipy.io as sio


def main(args):
    if args.model_path:
        pose_net_path = os.path.join(args.model_path, 'pose_net.ckpt')
        print('pretrained model for pose detector' + pose_net_path)

    matfn = args.label_file
    data = sio.loadmat(matfn)
    # 3 * 14 * 2000
    data = data['joints']

    # TODO: implement to get batch training data
    # batch_x, batch_c, batch_y, batch_x_orig = tf_utils.read_batch_cpm(FLAGS.tfr_data_files, FLAGS.input_size,
    #                                                                   FLAGS.heatmap_size, FLAGS.num_of_joints,
    #                                                                   FLAGS.center_radius, FLAGS.batch_size)


    # Building Graphs
    pose_image_in = tf.placeholder(tf.float32, shape=(None, 16, 376, 376, 15), name='pose_image_in')
    pose_centermap_in = tf.placeholder(tf.float32, shape=(None, 16, 376, 376, 1), name='pose_centermap_in')
    labels_placeholder = tf.placeholder(tf.float32, shape=(None, 46, 46, 15), name='gt_heatmap_placeholder')

    cpm = CPM_NETWORK(pose_image_in, pose_centermap_in, labels_placeholder, batch_size=args.batch_size,
                      weight_decay=args.l2_reg_lambda, learning_rate=args.learning_rate)
    cpm.build_loss()
    print('Finish building the model')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Train Summaries
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.num_checkpoints)
            sess.run(tf.global_variables_initializer())

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if args.model_path:
                print('Restoring the latest pretrained model: %s' % pose_net_path)
                saver.restore(sess, pose_net_path)

            while global_step != args.max_iteration:
                batch_x_np, batch_y_np, batch_c_np = sess.run([batch_x,
                                                               batch_y,
                                                               batch_c])

                gt_heatmap_np = make_gaussian_batch(batch_y_np)

                feed_dict = {pose_image_in: batch_x_np,
                             pose_centermap_in: batch_c_np,
                             labels_placeholder: gt_heatmap_np}

                stage_losses_np, total_loss_np, _, summary, current_lr, \
                stage_heatmap_np, global_step = sess.run([cpm.stage_loss,
                                                          cpm.total_loss,
                                                          cpm.train_op,
                                                          cpm.merged_summary,
                                                          cpm.learning_rate,
                                                          cpm.stage_heatmap,
                                                          cpm.global_step
                                                          ],
                                                         feed_dict=feed_dict)

                train_summary_writer.add_summary(summary, global_step)

                print('##========Iter {:>6d}========##'.format(global_step))
                for stage_num in range(args.stages):
                    print('Stage {} loss: {:>.3f}'.format(stage_num + 1, stage_losses_np[stage_num]))

                print('Total loss: {:>.3f}\n\n'.format(total_loss_np))

                if global_step % args.checkpoint_every == 0:
                    path = saver.save(sess, save_path=checkpoint_prefix, global_step=global_step)
                    print("Saved model checkpoint to {}\n".format(path))
    coord.request_stop()
    coord.join(threads)

    print('Training done.')


def make_gaussian_batch(labels):
    batch_datum = np.zeros(shape=(labels.shape[0], 46, 46, 15))
    stride_x = 376 // 46
    stride_y = 656 // 46

    for data_num in range(labels.shape[0]):
        for joint_num in range(labels.shape[2]):
            position = labels[data_num, :, joint_num]
            x = np.arange(0, 46, 1, float)
            y = x[:, np.newaxis]

            x0 = position[0]
            y0 = position[1]

            batch_datum[data_num, :, :, joint_num] = np.exp(
                -((x * stride_x - x0) ** 2 + (y * stride_y - y0) ** 2) / 2.0 / 3 / 3)
        batch_datum[data_num, :, :, 14] = np.ones((46, 46)) - np.amax(batch_datum[data_num, :, :, 0:14], axis=2)

    return batch_datum


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument('--model_path', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--label_file', type=str, help='Path to the .mat file')
    parser.add_argument('--log_dir', type=str, help='log directory to save the summary')

    # Model Hyperparameters
    parser.add_argument('--l2_reg_lambda', type=float,
                        help='L2 regularization lambda (default: 0.0)', default=0.0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--stages', type=int, help='Number of stages in our model', default=6)

    # Training parameters
    parser.add_argument('--batch_size', type=int,
                        help='Batch Size (default: 64)', default=32)
    parser.add_argument('--max_iteration', type=int,
                        help='Number of training iteration (default: 300000)', default=300000)
    parser.add_argument('--checkpoint_every', type=int,
                        help='Save model after this many steps (default: 100)', default=5000)
    parser.add_argument('--num_checkpoints', type=int,
                        help='Number of checkpoints to store (default: 3)', default=3)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
