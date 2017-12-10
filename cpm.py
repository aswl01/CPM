import tensorflow as tf
import tensorflow.contrib.slim as slim


def trained_person_MPI(images, scope='PersonNet', weight_decay=0.05):
    with tf.variable_scope(scope, 'PersonNet'):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=tf.nn.relu,
                            initizalizer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='VALID', kernel_size=2):
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, scope='maxpool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, scope='maxpool2')
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, scope='maxpool3')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
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
                            initizalizer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='VALID', kernel_size=3):
                pool_center_lower = slim.avg_pool2d(center_map, 9, 8, scope='pool_center_lower')
                conv4_stage1, _ = stage_x(images, 1)
                conv5_stage1 = slim.conv2d(conv4_stage1, 512, 9, scope='conv5_stage1')
                conv6_stage1 = slim.conv2d(conv5_stage1, 512, 1, scope='conv6_stage1')
                conv7_stage1 = slim.conv2d(conv6_stage1, 15, 1, scope='conv7_stage1')
                conv4_stage2, pool3_stage2 = stage_x(images, 2)
                concat_stage2 = tf.concat([conv4_stage2, conv7_stage1, pool_center_lower], 3)
                Mconv5_stage2 = substage(concat_stage2, 2)
                conv1_stage3 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage3')
                concat_stage3 = tf.concat([conv1_stage3, Mconv5_stage2, pool_center_lower], 3)
                Mconv5_stage3 = substage(concat_stage3, 3)
                conv1_stage4 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage4')
                concat_stage4 = tf.concat([conv1_stage4, Mconv5_stage3, pool_center_lower], 3)
                Mconv5_stage4 = substage(concat_stage4, 4)
                conv1_stage5 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage5')
                concat_stage5 = tf.concat([conv1_stage5, Mconv5_stage4, pool_center_lower], 3)
                Mconv5_stage5 = substage(concat_stage5, 5)
                conv1_stage6 = slim.conv2d(pool3_stage2, 32, 5, scope='conv1_stage6')
                concat_stage6 = tf.concat([conv1_stage6, Mconv5_stage5, pool_center_lower], 3)
                Mconv5_stage6 = substage(concat_stage6, 6)
                return Mconv5_stage6


def trained_MPI(images, center_map, scope='PoseNet', weight_decay=0.05):
    with tf.variable_scope(scope, 'PoseNet'):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=tf.nn.relu,
                            initizalizer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='VALID', kernel_size=3):
                pool_center_lower = slim.avg_pool2d(center_map, 9, 8, scope='pool_center_lower')
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, scope='maxpool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, scope='maxpool2')
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, scope='maxpool3')
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                for i in range(3, 7):
                    net = slim.conv2d(net, 256, 3, scope='conv4_{}_CPM'.format(i))
                conv4_7_CPM = slim.conv2d(net, 128, 3, scope='conv4_7_CPM')
                conv5_1_CPM = slim.conv2d(conv4_7_CPM, 512, 1, scope='conv5_1_CPM')


def stage_x(net, stage):
    for i in range(3):
        net = slim.conv2d(net, 128, 9, scope='conv{}_stage{}'.format(i + 1, stage))
        net = slim.max_pool2d(net, scope='pool{}_stage{}'.format(i + 1, stage))
    conv4_stage = slim.conv2d(net, 32, 5, scope='conv4_stage{}'.format(stage))
    return conv4_stage, net


def substage(net, stage):
    for i in range(3):
        net = slim.conv2d(net, 128, 11, scope='Mconv{}_stage_{}'.format(i + 1, stage))
    Mconv4_stage = slim.conv2d(net, 128, 1, scope='Mconv4_stage{}'.format(stage))
    Mconv5_stage = slim.conv2d(Mconv4_stage, 15, 1, scope='Mconv5_stage{}'.format(stage), activation_fn=None)
    return Mconv5_stage
