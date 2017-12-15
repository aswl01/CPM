import numpy as np
import tensorflow as tf
from cpm import trained_LEEDS_PC


class CPM_NETWORK(object):
    def __init__(self, imageN, imageH, imageW, num_stage=6, epoch_size=1, batch_size=1, weight_decay=0.05,
                 random_crop=True, random_flip=True,
                 random_contrast=True, random_rotate=True):
        # define placeholder for the input image
        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        self.pose_image_in = tf.placeholder(tf.float32, shape=(None, imageN, imageH, imageW, 3), name='pose_image_in')
        self.pose_centermap_in = tf.placeholder(tf.float32, shape=(None, imageN, imageH, imageW, 1),
                                                name='pose_centermap_in')
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, num_stage, imageH, imageW, 3), name='labels')
        # inference
        self.pose_image_out, endpoint = trained_LEEDS_PC(self.pose_image_in, self.pose_centermap_in, weight_decay=weight_decay)
        # calculate the loss
        self.stage_loss = loss(endpoint)
        tf.summary.scalar('stage_loss', self.stage_loss)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n([self.stage_loss] + regularization_losses, name='total_loss')
        tf.summary.scalar('total_loss', self.total_loss)

def loss(endpoint):
