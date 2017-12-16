import numpy as np
import tensorflow as tf
from cpm import trained_LEEDS_PC
import numpy as np
import scipy.io as sio


class CPM_NETWORK(object):
    def __init__(self, imageN, imageH, imageW, num_stage=6, epoch_size=1, batch_size=1, weight_decay=0.05,
                 random_crop=True, random_flip=True,
                 random_contrast=True, random_rotate=True):
        # define placeholder for the input image
        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        self.pose_image_in = tf.placeholder(tf.float32, shape=(None, imageN, imageH, imageW, 3), name='pose_image_in')
        self.pose_centermap_in = tf.placeholder(tf.float32, shape=(None, imageN, imageH, imageW, 1),
                                                name='pose_centermap_in')
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 3, 14), name='labels')
        # inference
        self.pose_image_out, endpoint = trained_LEEDS_PC(self.pose_image_in, self.pose_centermap_in,
                                                         weight_decay=weight_decay)
        # calculate the loss
        self.stage_loss = loss_func(endpoint, self.labels_placeholder)
        tf.summary.scalar('stage_loss', self.stage_loss)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n([self.stage_loss] + regularization_losses, name='total_loss')


def ideal_addGaussian(x, y):
    sigma = 21.0

    xx = np.linspace(1.0, float(376), 376)
    yy = np.linspace(1.0, float(656), 656)

    X, Y = np.meshgrid(xx, yy)
    X = X - x
    Y = Y - y

    D2 = np.power(X, 2) + np.power(Y, 2)

    Exponent = D2 * (1 / sigma) * (1 / sigma) * 0.5 * (-1)
    label_matrix = np.exp(Exponent)
    return label_matrix


'''
Args:
    stageOut: list of stage output matrix
    index: which picture you used for this train iteration in 2000 dataset pictures
    height: trainpic height pixels
    width: trainpic width pixels

Returns:
    loss value
'''


# height width
# stageOut batchsize X stage X 46 X 46 X 15
# index batchsize X 3 X 14
def loss_func(stageOut, label):
    # Path for dataset
    res = 0.0
    for j in range(len(stageOut)):

        coordinate_list = label[j, :, :]
        matrix_list = []
        for i in range(14):
            matrix_list.append(ideal_addGaussian(coordinate_list[0][i], coordinate_list[1][i]))

        ideal_matrix = matrix_list[0]
        for i in range(13):
            ideal_matrix = np.dstack([ideal_matrix, matrix_list[i + 1]])

        ideal_matrix = np.dstack([ideal_matrix, np.zeros((376, 656))])

        # reshape to 46 X 46
        ideal_matrix = np.reshape(ideal_matrix, (46, 46, 15))

        sum2 = 0.0

        for i in range(6):
            stageOutMatrix = stageOut[j, i, :, :, :]
            tmpMatrix = stageOutMatrix - ideal_matrix
            afterNormMatrix = np.linalg.norm(tmpMatrix, axis=2)
            afterNormMatrix = np.power(afterNormMatrix, 2)
            sum1 = np.sum(afterNormMatrix, axis=1)
            sum2 += np.sum(sum1, axis=0)

        res += sum2

    return res
