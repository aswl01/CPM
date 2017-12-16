import argparse
import codecs
import os
import sys
import time
from datetime import datetime
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import learn
from model import cpm
from data_utils import *
import numpy as np
from model.cpm_network import CPM_NETWORK
import scipy.io as sio


def main(args):
    if args.pretrained_model:
        pretrained_model = tf.train.latest_checkpoint(args.pretrained_model)
        print('Pre-trained model: %s' % os.path.expanduser(pretrained_model))

    matfn = args.label_file
    data = sio.loadmat(matfn)

    data = data['joints']
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            network = CPM_NETWORK()
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-5)
            grads_and_vars = optimizer.compute_gradients(network.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument('--dev_sample_percentage', type=float,
                        help='Percentage of the training data to use for validation', default=0.01)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--label_file', type=str, help='Path to the .mat file')

    # Model Hyperparameters
    parser.add_argument('--embedding_dim', type=int,
                        help='Dimensionality of character embedding (default: 128)', default=128)
    parser.add_argument('--filter_sizes', type=str,
                        help="Comma-separated filter sizes (default: 2,3,4,5')", default="2, 3, 4, 5, 6")
    parser.add_argument('--num_filters', type=int,
                        help='Number of filters per filter size (default: 128)', default=128)
    parser.add_argument('--dropout_keep_prob', type=float,
                        help='Dropout keep probability (default: 0.5)', default=0.5)
    parser.add_argument('--l2_reg_lambda', type=float,
                        help='L2 regularization lambda (default: 0.0)', default=0.0)

    # Training parameters
    parser.add_argument('--batch_size', type=int,
                        help='Batch Size (default: 64)', default=64)
    parser.add_argument('--num_epochs', type=int,
                        help='Number of training epochs (default: 200)', default=200)
    parser.add_argument('--evaluate_every', type=int,
                        help='Evaluate model on dev set after this many steps (default: 100)', default=100)
    parser.add_argument('--checkpoint_every', type=int,
                        help='Save model after this many steps (default: 100)', default=100)
    parser.add_argument('--num_checkpoints', type=int,
                        help='Number of checkpoints to store (default: 3)', default=3)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
