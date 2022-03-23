import tensorflow as tf


def pack_matrix(mat):
    return tf.stack([tf.stack(row, axis=-1) for row in mat], axis=-2)


def pack_matrix_transposed(mat):
    return tf.stack([tf.stack(col, axis=-1) for col in mat], axis=-1)
