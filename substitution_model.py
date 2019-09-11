import tensorflow as tf
import numpy as np

jc_eigendecomposition = [np.array(x) for x in [
    [
        [1.0, 2.0, 0.0, 0.5],
        [1.0, -2.0, 0.5, 0.0],
        [1.0, 2.0, 0.0, -0.5],
        [1.0, -2.0, -0.5, 0.0]
    ],
    [0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333],
    [
        [0.25, 0.25, 0.25, 0.25],
        [0.125, -0.125, 0.125, -0.125],
        [0.0, 1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0, 0.0]
    ]
]]

jc_frequencies = np.array([0.25, 0.25, 0.25, 0.25])

def transition_probs(eigendecomposition, t):
    U, lambd, Vt = [np.array(x) for x in eigendecomposition]
    diag = tf.linalg.diag(tf.exp(tf.expand_dims(t, 1) * tf.expand_dims(lambd, 0)))
    return tf.reduce_sum(tf.reshape(U, [1, 4, 4, 1, 1]) * tf.reshape(diag, [-1, 1, 4, 4, 1]) * tf.reshape(Vt, [1, 1, 1, 4, 4]), axis=[2, 3])
