import tensorflow as tf
from keras import backend as K, Input, Model
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, BatchNormalization
import numpy as np


X = np.random.rand(1000, 8, 8, 3)
Y = np.random.rand(1000, 10)
# create model
input_tensor = output_tensor = Input(shape=(8, 8, 3))
output_tensor = Conv2D(filters=30,
                       kernel_size=3,
                       kernel_initializer='he_normal')(output_tensor)
output_tensor = BatchNormalization()(output_tensor)
output_tensor = Activation('relu')(output_tensor)
output_tensor = Dropout(0.5)(output_tensor)
output_tensor = MaxPooling2D()(output_tensor)

output_tensor = Flatten()(output_tensor)
output_tensor = Dense(10, kernel_initializer='he_normal', activation='softmax')(output_tensor)

model = Model(input_tensor, output_tensor)
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
K.set_session(sess)
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)