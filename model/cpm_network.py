import numpy as np
import tensorflow as tf
from cpm import trained_LEEDS_PC
import numpy as np


class CPM_NETWORK(object):
    def __init__(self, pose_image_in, pose_centermap_in, labels_placeholder, batch_size, num_stage=6, weight_decay=0.05, learning_rate=10e-5):
        # define placeholder for the input image
        self.stage_loss = 0
        self.total_loss = 0
        self.batch_size = batch_size
        self.stage_loss = [0] * num_stage
        self.learning_rate = learning_rate
        self.pose_image_in = pose_image_in
        self.pose_centermap_in = pose_centermap_in
        self.labels_placeholder = labels_placeholder
        self.stages = num_stage
        # inference
        self.pose_image_out, self.stage_heatmap = trained_LEEDS_PC(pose_image_in, pose_centermap_in,
                                                                   weight_decay=weight_decay)

    def build_loss(self):
        self.total_loss = 0

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.labels_placeholder,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_creat_global_step()

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.learning_rate,
                                                            optimizer='Adam')
        self.merged_summary = tf.summary.merge_all()
