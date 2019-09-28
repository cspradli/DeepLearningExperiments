from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
farenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees celsius = {} degrees ferenheit".format(c, farenheit_a[i]))

# input shape refers to the single input value. That is a 1-d array with one member
# units specifies the number of neurons in the layer
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# once the layers are defined, they need to assembled into a model. Seq model takes
# a list of layers as args, specifying the calculation order from in to out
# this model has just l0
model = tf.keras.Sequential([l0])

# Loss function is the way of handling errors
# Opt function is a way of adjusting interal vals in order to reduce loss
# the params for optimization is the learning rate, i.e. alpha
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, farenheit_a, epochs=500, verbose=False)
print("Finished training nnet")

import matplotlib.pyplot as plt
plt.xlabel('Epoch num')
plt.ylabel("Loss mag")
plt.plot(history.history['loss'])

print(model.predict([100.0]))

print("These are the layer vars: {}".format(l0.get_weights()))