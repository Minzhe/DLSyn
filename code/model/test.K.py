import numpy as np
import keras.backend as K
import tensorflow as tf
from neural_net import weighted_mse

def weighted_loss(t, p):
    loss = np.square(t - p)
    weight = 2 * abs(t) + 1
    weight[abs(t) > 0.5] = 1
    return np.mean(loss * weight)

t = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape((-1,1))
p = np.array([0.05, 0.12, 0.15, 0.28, 0.4, 0.55, 0.58, 0.8, 0.89, 0.88, 1.2]).reshape((-1,1))
print(weighted_loss(t, p))
exit()

t = K.constant(t)
p = K.constant(p)


with tf.Session() as sess:
    loss = weighted_mse(t, p)
    print(sess.run(loss))