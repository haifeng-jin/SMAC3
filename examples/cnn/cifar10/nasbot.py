import os
import GPUtil
import tensorflow as tf
import numpy as np
from keras import Input, Model, backend
from keras.layers import Conv2D, MaxPooling2D, Dense, Concatenate, Flatten, AveragePooling2D
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold

from examples.cnn.cifar10.cifar10 import get_data
from examples.cnn.cnn import ModelTrainer


def get_model(x_train):
    input_shape = x_train.shape[1:]
    input_tensor = output_tensor = Input(shape=input_shape)

    output_tensor = Conv2D(64, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(64, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = MaxPooling2D(padding='same')(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)

    output5 = output_tensor

    output_tensor = MaxPooling2D(padding='same')(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = MaxPooling2D(padding='same')(output_tensor)

    output14 = output_tensor

    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(128, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)

    output_tensor = Concatenate()([output_tensor, output14])
    output_tensor = MaxPooling2D(padding='same')(output_tensor)
    output_tensor = Conv2D(448, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(512, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(512, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Conv2D(512, 3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)

    output23 = output_tensor

    output_tensor = AveragePooling2D(padding='same')(output5)
    output_tensor = AveragePooling2D(padding='same')(output_tensor)
    output_tensor = AveragePooling2D(padding='same')(output_tensor)

    output_tensor = Concatenate()([output_tensor, output23])
    output_tensor = Conv2D(512, 7,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = MaxPooling2D(padding='same')(output_tensor)
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(128,
                          activation='relu')(output_tensor)
    output_tensor = Dense(256,
                          activation='relu')(output_tensor)
    output_tensor = Dense(448,
                          activation='relu')(output_tensor)
    output_tensor = Dense(10,
                          activation='softmax')(output_tensor)

    return Model(input_tensor, output_tensor)


def reset_weights(model):
    """Reset weights with a new model"""
    session = backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def train_model(model, x_train, y_train, x_test, y_test):
    # reset_weights(model)
    ModelTrainer(model, x_train, y_train, x_test, y_test, True).train_model()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=True)

    print(accuracy)

    return 1 - accuracy


def select_gpu():
    try:
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list

        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except EnvironmentError:
        print("GPU not found")


select_gpu()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
backend.set_session(sess)

x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
cnn_model = get_model(X)
weights = cnn_model.get_weights()
for train, test in k_fold.split(X, y_stub):
    cnn_model.set_weights(weights)
    print("Training started")
    ret.append(train_model(cnn_model, X[train], Y[train], X[test], Y[test]))
print(np.array(ret))
