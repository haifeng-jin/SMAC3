import datetime
import logging

from keras.layers import MaxPooling1D, Conv1D
from keras.preprocessing import sequence

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
from keras.datasets import cifar10, imdb


def get_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=400)
    x_test = sequence.pad_sequences(x_test, maxlen=400)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test


def run(params):
    print('Start time: ', datetime.datetime.now())
    result = cnn(params, get_data(), Conv1D, MaxPooling1D, True, True)
    print('Result: ', params, result)
    print('End time: ', datetime.datetime.now())
    return result


if __name__ == '__main__':
    select_gpu()
    logging.basicConfig(level=20)  # 10: debug; 20: info
    x, cost, _ = fmin_smac(func=run,  # function
                           x0=[50, 50, 50, 50, 50, 50, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01],  # default configuration
                           bounds=[(10, 500), (10, 500), (10, 500), (10, 500), (10, 500), (10, 500),
                                   (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
                                   (0.0001, 0.1), (0.0001, 0.1), (0.0001, 0.1), (0.0001, 0.1)],  # limits
                           maxfun=500,  # maximum number of evaluations
                           rng=3)  # random seed
    print("Optimum at {} with cost of {}".format(x, cost))
