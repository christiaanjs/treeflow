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

A, C, G, T = list(range(4))

def hky_eigendecomposition(pi, kappa):
	piY = pi[T] + pi[C]
	piR = pi[A] + pi[G]

	beta = -1.0 / (2.0 * (piR*piY + kappa * (pi[A]*pi[G] + pi[C]*pi[T])))
	A_R = 1.0 + piR * (kappa - 1)
	A_Y = 1.0 + piY * (kappa - 1)
	lambd = tf.stack([ # Eigenvalues 
		0.0,
		beta,
		beta * A_Y,
		beta * A_R
	])
	U = tf.transpose(tf.stack([ # Right eigenvectors as columns (rows of transpose)
		[1.0, 1.0, 1.0, 1.0],
		[1.0/piR, -1.0/piY, 1.0/piR, -1.0/piY],
		[0.0, pi[T]/piY, 0.0, -pi[C]/piY],
		[pi[G]/piR, 0.0, -pi[A]/piR, 0.0]
	]))

	Vt = tf.stack([ # Left eigenvectors as rows
		[pi[A], pi[C], pi[G], pi[T]],
		[pi[A]*piY, -pi[C]*piR, pi[G]*piY, -pi[T]*piR],
		[0.0, 1.0, 0.0, -1.0],
		[1.0, 0.0, -1.0, 0.0]
	])

	return [tf.dtypes.cast(x, tf.dtypes.float64) for x in [U, lambd, Vt]]

def hky_q_matrix(pi, kappa):
    return tf.stack([
        [-(pi[C] + kappa*pi[G] + pi[T]), pi[C], kappa*pi[G], pi[T]],
        [pi[A], -(pi[A] + pi[G] + kappa*pi[T]) , pi[G], kappa*pi[T]],
        [kappa*pi[A], pi[C], -(kappa*pi[A] + pi[C] + pi[T]), pi[T]],
        [pi[A], kappa*pi[C], pi[G], -(pi[A] + kappa*pi[C] + pi[G])]
    ])

def gtr_q_matrix(pi, rates):
    return tf.stack([
        [-(rates[0]*pi[1] + rates[1]*pi[2] + rates[2]*pi[3]), rates[0]*pi[1], rates[1]*pi[2], rates[2]*pi[3]],
        [rates[0]*pi[0], -(rates[0]*pi[0] + rates[3]*pi[2] + rates[4]*pi[3]), rates[3]*pi[2], rates[4]*pi[3]],
        [rates[1]*pi[0], rates[3]*pi[1], -(rates[1]*pi[0] + rates[3]*pi[1] + rates[5]*pi[3]), rates[5]*pi[3]],
        [rates[2]*pi[0], rates[4]*pi[1], rates[5]*pi[2], -(rates[2]*pi[0] + rates[4]*pi[1] + rates[5]*pi[2])]     
    ])

def normalising_constant(q, pi):
    return -tf.reduce_sum(tf.linalg.diag_part(q) * pi)

def normalise(q, pi):
    return q / normalising_constant(q, pi)

def transition_probs(eigendecomposition, t):
    U, lambd, Vt = eigendecomposition
    diag = tf.linalg.diag(tf.exp(tf.expand_dims(t, 1) * tf.expand_dims(lambd, 0)))
    return tf.reduce_sum(tf.reshape(U, [1, 4, 4, 1, 1]) * tf.reshape(diag, [-1, 1, 4, 4, 1]) * tf.reshape(Vt, [1, 1, 1, 4, 4]), axis=[2, 3])
