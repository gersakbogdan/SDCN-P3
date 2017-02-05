import helpers
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


tf.python.control_flow_ops = tf

#
# Based on Nvidia - End to End Learning for Self-Driving Cars network architecture:
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
#
model = Sequential()

# Normalization
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(64, 64, 3)))
# Layer 1
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Dropout(0.5))
# Layer 2
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Dropout(0.5))
# Layer 3
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Dropout(0.5))
# Layer 4
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu'))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Dropout(0.5))
# Layer 5
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu'))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Dropout(0.5))

model.add(Flatten())

# Fully-connected layer 6
model.add(Dense(1164, activation='relu'))
# Fully-connected layer 7
model.add(Dense(100, activation='relu'))
# Fully-connected layer 8
model.add(Dense(50, activation='relu'))
# Fully-connected layer 9
model.add(Dense(10, activation='relu'))
# Output
model.add(Dense(1))


# Compile using Adam optimizer and mse loss
model.compile('adam', 'mse')
# Display model details
model.summary()

# Train model
history = model.fit_generator(
    helpers.train_generator(),
    samples_per_epoch=49152,
    nb_epoch=5,
    validation_data=helpers.validation_generator(),
    nb_val_samples=15360,
    verbose=1)

# Save model and weights
model.save('model.h5')
