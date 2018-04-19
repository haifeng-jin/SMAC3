import datetime
import logging

from examples.cnn.cnn import cnn, select_gpu
from smac.facade.func_facade import fmin_smac

"""
Example for the use of fmin_smac, a basic SMAC-wrapper to optimize a
function.
We optimize the branin-function, which has two parameters: x1 and x2.
The fmin_smac needs neither a scenario-file, nor a configuration space.
All relevant information is directly passed to the function.
"""

import keras
import numpy as np
from keras.datasets import cifar10


def get_data():
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    # Convert class vectors to binary class matrices.
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def run(params):
    print('Start time: ', datetime.datetime.now())
    result = cnn(params, get_data())
    print('Result: ', params, result)
    print('End time: ', datetime.datetime.now())
    return result


if __name__ == '__main__':
    select_gpu()
    logging.basicConfig(level=20)  # 10: debug; 20: info
    x, cost, _ = fmin_smac(func=run,  # function
                           x0=[50, 50, 50, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01],  # default configuration
                           bounds=[(10, 500), (10, 500), (10, 500),
                                   (0, 1), (0, 1), (0, 1),
                                   (0.0001, 0.1), (0.0001, 0.1), (0.0001, 0.1)],  # limits
                           maxfun=500,  # maximum number of evaluations
                           rng=3)  # random seed
    print("Optimum at {} with cost of {}".format(x, cost))
