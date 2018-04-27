import numpy as np
from keras import Input, Model
from keras.layers import SeparableConv2D, Conv2D, MaxPooling2D, Flatten, Dense, Add
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold, train_test_split

from examples.cnn.cifar10.cifar10 import get_data
from examples.cnn.cnn import ModelTrainer


def cnn(x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    input_tensor = output_tensor = Input(shape=input_shape)
    output_tensor = SeparableConv2D(256,
                                    kernel_size=5,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output1 = output_tensor
    output_tensor = Conv2D(256,
                           kernel_size=5,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output2 = output_tensor
    output_tensor = Conv2D(256,
                           kernel_size=5,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output3 = output_tensor
    output_tensor = Add()([output_tensor, output1])
    output_tensor = SeparableConv2D(256,
                                    kernel_size=5,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output4 = output_tensor
    output_tensor = Add()([output_tensor, output1])
    output_tensor = MaxPooling2D()(output_tensor)

    output1 = MaxPooling2D()(output1)
    output2 = MaxPooling2D()(output2)
    output3 = MaxPooling2D()(output3)
    output4 = MaxPooling2D()(output4)

    output_tensor = SeparableConv2D(256,
                                    kernel_size=3,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output5 = output_tensor
    output_tensor = Add()([output_tensor, output1, output4])
    output_tensor = Conv2D(256,
                           kernel_size=5,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output6 = output_tensor
    output_tensor = Add()([output_tensor, output3, output4, output5])
    output_tensor = SeparableConv2D(256,
                                    kernel_size=3,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output7 = output_tensor
    output_tensor = Add()([output_tensor, output1, output3, output4, output5, output6])
    output_tensor = SeparableConv2D(256,
                                    kernel_size=5,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Add()([output_tensor, output1, output4, output5, output7])
    output_tensor = MaxPooling2D()(output_tensor)

    output1 = MaxPooling2D()(output1)
    output2 = MaxPooling2D()(output2)
    output3 = MaxPooling2D()(output3)
    output4 = MaxPooling2D()(output4)
    output5 = MaxPooling2D()(output5)
    output6 = MaxPooling2D()(output6)
    output7 = MaxPooling2D()(output7)

    output_tensor = Conv2D(256,
                           kernel_size=5,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output9 = output_tensor
    output_tensor = Add()([output_tensor, output6])
    output_tensor = SeparableConv2D(256,
                                    kernel_size=5,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output10 = output_tensor
    output_tensor = Add()([output_tensor, output2, output4, output7])
    output_tensor = SeparableConv2D(256,
                                    kernel_size=5,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Add()([output_tensor, output2, output3, output5, output9])
    output_tensor = Conv2D(256,
                           kernel_size=3,
                           kernel_initializer='he_normal',
                           activation='relu',
                           padding='same',
                           kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Add()([output_tensor, output1, output3, output4, output5, output7, output10])
    output_tensor = SeparableConv2D(256,
                                    kernel_size=5,
                                    kernel_initializer='he_normal',
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=l2(1e-4))(output_tensor)
    output_tensor = Add()([output_tensor, output4, output7, output10])
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(10, activation='softmax')(output_tensor)

    model = Model(input_tensor, output_tensor)

    ModelTrainer(model, x_train, y_train, x_test, y_test, False).train_model()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

    print(accuracy)

    return 1 - accuracy


x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
for train, test in k_fold.split(X, y_stub):
    ret.append(cnn(X[train], Y[train], X[test], Y[test]))
print(np.array(ret))
