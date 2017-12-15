import tensorflow as tf
import tensorflow.contrib.slim as slim


def trained_person_MPI(images, scope='PersonNet', weight_decay=0.05):
    with tf.variable_scope(scope, 'PersonNet'):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='VALID', kernel_size=2):
                net = images
                for i in range(2):
                    net = slim.conv2d(net, 64, 3, scope='conv1_{}'.format(i + 1))
                net = slim.max_pool2d(net, scope='maxpool1')
                for i in range(2):
                    net = slim.conv2d(net, 128, 3, scope='conv2_{}'.format(i + 1))
                net = slim.max_pool2d(net, scope='maxpool2')
                for i in range(4):
                    net = slim.conv2d(net, 256, 3, scope='conv3_{}'.format(i + 1))
                net = slim.max_pool2d(net, scope='maxpool3')
                for i in range(4):
                    net = slim.conv2d(net, 512, 3, scope='conv4_{}'.format(i + 1))
                net = slim.max_pool2d(net, scope='maxpool4')
                conv5_1 = slim.conv2d(net, 512, 3, scope='conv5_1')
                conv5_2_CPM = slim.conv2d(conv5_1, 128, 3, scope='conv5_2_CPM')
                conv6_1_CPM = slim.conv2d(conv5_2_CPM, 512, 1, scope='conv6_1_CPM')
                conv6_2_CPM = slim.conv2d(conv6_1_CPM, 1, 1, scope='conv6_2_CPM')
                concat_stage2 = tf.concat([conv6_2_CPM, conv5_2_CPM], 3)
                for i in range(5):
                    concat_stage2 = slim.conv2d(concat_stage2, 128, 7, scope='Mconv{}_stage2'.format(i + 1))
                Mconv6_stage2 = slim.conv2d(concat_stage2, 128, 1, scope='Mconv6_stage2')
                Mconv7_stage2 = slim.conv2d(Mconv6_stage2, 1, 1, scope='Mconv7_stage2')
                concat_stage3 = tf.concat([Mconv7_stage2, conv5_2_CPM], 3)
                for i in range(5):
                    concat_stage3 = slim.conv2d(concat_stage3, 128, 7, scope='Mconv{}_stage3'.format(i + 1))
                Mconv6_stage3 = slim.conv2d(concat_stage3, 128, 1, scope='Mconv6_stage3')
                Mconv7_stage3 = slim.conv2d(Mconv6_stage3, 1, 1, scope='Mconv7_stage3')
                concat_stage4 = tf.concat([Mconv7_stage3, conv5_2_CPM], 3)
                for i in range(5):
                    concat_stage4 = slim.conv2d(concat_stage4, 128, 7, scope='Mconv{}_stage4'.format(i + 1))
                Mconv6_stage4 = slim.conv2d(concat_stage4, 128, 1, scope='Mconv6_stage4')
                Mconv7_stage4 = slim.conv2d(Mconv6_stage4, 1, 1, scope='Mconv7_stage4')
    return Mconv7_stage4


def trained_LEEDS_PC(images, center_map, scope='PoseNet', weight_decay=0.05):
    with tf.variable_scope(scope, 'PoseNet'):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='VALID', kernel_size=3):
                pool_center_lower = slim.avg_pool2d(center_map, 9, 8, scope='pool_center_lower')
                conv4_stage1, _ = _stage_x(images, 1)
                conv5_stage1 = slim.conv2d(conv4_stage1, 512, 9, scope='conv5_stage1')
                conv6_stage1 = slim.conv2d(conv5_stage1, 512, 1, scope='conv6_stage1')
                conv7_stage1 = slim.conv2d(conv6_stage1, 15, 1, scope='conv7_stage1')
                print(conv7_stage1.shape)
                conv4_stage2, pool3_stage2 = _stage_x(images, 2)
                concat_stage2 = tf.concat([conv4_stage2, conv7_stage1, pool_center_lower], 3)
                Mconv5_stage2 = _LEEDS_substage(concat_stage2, 2)
                print(Mconv5_stage2.shape)
                conv1_stage3 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage3')
                concat_stage3 = tf.concat([conv1_stage3, Mconv5_stage2, pool_center_lower], 3)
                Mconv5_stage3 = _LEEDS_substage(concat_stage3, 3)
                print(Mconv5_stage3.shape)
                conv1_stage4 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage4')
                concat_stage4 = tf.concat([conv1_stage4, Mconv5_stage3, pool_center_lower], 3)
                Mconv5_stage4 = _LEEDS_substage(concat_stage4, 4)
                print(Mconv5_stage4.shape)
                conv1_stage5 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage5')
                concat_stage5 = tf.concat([conv1_stage5, Mconv5_stage4, pool_center_lower], 3)
                Mconv5_stage5 = _LEEDS_substage(concat_stage5, 5)
                print(Mconv5_stage5.shape)
                conv1_stage6 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage6')
                concat_stage6 = tf.concat([conv1_stage6, Mconv5_stage5, pool_center_lower], 3)
                Mconv5_stage6 = _LEEDS_substage(concat_stage6, 6)
                print(Mconv5_stage6.shape)
                return Mconv5_stage6


def trained_MPI(images, center_map, scope='PoseNet', weight_decay=0.05):
    with tf.variable_scope(scope, 'PoseNet'):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='VALID', kernel_size=3):
                pool_center_lower = slim.avg_pool2d(center_map, 9, 8, scope='pool_center_lower')
                net = images
                for i in range(2):
                    net = slim.conv2d(net, 64, 3, scope='conv1_{}'.format(i + 1))
                net = slim.max_pool2d(net, scope='maxpool1')
                for i in range(2):
                    net = slim.conv2d(net, 128, 3, scope='conv2_{}'.format(i + 1))
                net = slim.max_pool2d(net, scope='maxpool2')
                for i in range(4):
                    net = slim.conv2d(net, 256, 3, scope='conv3_{}'.format(i + 1))
                net = slim.max_pool2d(net, scope='maxpool3')
                for i in range(2):
                    net = slim.conv2d(net, 512, 3, scope='conv4_{}'.format(i + 1))
                for i in range(3, 7):
                    net = slim.conv2d(net, 256, 3, scope='conv4_{}_CPM'.format(i))
                conv4_7_CPM = slim.conv2d(net, 128, 3, scope='conv4_7_CPM')
                conv5_1_CPM = slim.conv2d(conv4_7_CPM, 512, 1, scope='conv5_1_CPM')
                conv5_2_CPM = slim.conv2d(conv5_1_CPM, 15, 1, scope='conv5_2_CPM')
                concat_stage2 = tf.concat([conv5_2_CPM, conv4_7_CPM, pool_center_lower], 3)
                Mconv7_stage2 = _MPI_substage(concat_stage2, 2)
                concat_stage3 = tf.concat([Mconv7_stage2, conv4_7_CPM, pool_center_lower], 3)
                Mconv7_stage3 = _MPI_substage(concat_stage3, 3)
                concat_stage4 = tf.concat([Mconv7_stage3, conv4_7_CPM, pool_center_lower], 3)
                Mconv7_stage4 = _MPI_substage(concat_stage4, 4)
                concat_stage5 = tf.concat([Mconv7_stage4, conv4_7_CPM, pool_center_lower], 3)
                Mconv7_stage5 = _MPI_substage(concat_stage5, 5)
                concat_stage6 = tf.concat([Mconv7_stage5, conv4_7_CPM, pool_center_lower], 3)
                Mconv7_stage6 = _MPI_substage(concat_stage6, 6)
                return Mconv7_stage6


def _stage_x(net, stage):
    for i in range(3):
        net = slim.conv2d(net, 128, 9, scope='conv{}_stage{}'.format(i + 1, stage))
        net = slim.max_pool2d(net, scope='pool{}_stage{}'.format(i + 1, stage))
    conv4_stage = slim.conv2d(net, 32, 5, scope='conv4_stage{}'.format(stage))
    return conv4_stage, net


def _LEEDS_substage(net, stage):
    for i in range(3):
        net = slim.conv2d(net, 128, 11, scope='Mconv{}_stage{}'.format(i + 1, stage))
    Mconv4_stage = slim.conv2d(net, 128, 1, scope='Mconv4_stage{}'.format(stage))
    Mconv5_stage = slim.conv2d(Mconv4_stage, 15, 1, scope='Mconv5_stage{}'.format(stage), activation_fn=None)
    return Mconv5_stage


def _MPI_substage(net, stage):
    for i in range(5):
        net = slim.conv2d(net, 128, 7, scope='Mconv{}_stage{}'.format(i + 1, stage))
    Mconv6_stage = slim.conv2d(net, 128, 1, scope='Mconv4_stage{}'.format(stage))
    Mconv7_stage = slim.conv2d(Mconv6_stage, 15, 1, scope='Mconv5_stage{}'.format(stage), activation_fn=None)
    return Mconv7_stage
