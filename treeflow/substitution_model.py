import tensorflow as tf
import numpy as np

A, C, G, T = list(range(4))

class SubstitutionModel:
    def q(self, frequencies, **kwargs):
        raise NotImplementedError()

    def q_norm(self, frequencies, **kwargs):
        return normalise(self.q(frequencies, **kwargs), frequencies)

    def eigen(self, frequencies, **kwargs):
        raise NotImplementedError()

    def q_frequency_differentials(self, frequencies, **kwargs):
        raise NotImplementedError()

    def q_norm_frequency_differentials(self, frequencies, **kwargs): 
        q = self.q(frequencies, **kwargs)
        q_norm, norm_const = normalise_and_constant(q, frequencies)
        diffs = self.q_frequency_differentials(frequencies, **kwargs)
        return tf.stack([normalised_differential(diffs[i], q_norm, norm_const, frequencies, frequency_index=i, q=q) for i in range(4)])                

    def q_param_differentials(self, frequencies, **kwargs):
        raise NotImplementedError()

    def q_norm_param_differentials(self, frequencies, **kwargs):
        q = self.q(frequencies, **kwargs)
        q_norm, norm_const = normalise_and_constant(q, frequencies)
        diffs = self.q_param_differentials(frequencies, **kwargs) 
        return { key: normalised_differential(diff, q_norm, norm_const, frequencies) for key, diff in diffs.items() }

    def param_keys(self):
        raise NotImplementedError()

class JC(SubstitutionModel):
    def frequencies(self, *args, **kwargs):
        return tf.convert_to_tensor(np.array([0.25, 0.25, 0.25, 0.25]))

    def eigen(self, *args, **kwargs):
        return [tf.convert_to_tensor(np.array(x)) for x in [
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

    def q(self, *args, **kwargs):
        return tf.convert_to_tensor(np.array([
            [-1, 1/3, 1/3, 1/3],
            [1/3, -1, 1/3, 1/3],
            [1/3, 1/3, -1, 1/3],
            [1/3, 1/3, 1/3, -1]
        ]))


class HKY(SubstitutionModel):
    def eigen(self, frequencies, kappa):
        pi = frequencies
        piY = pi[T] + pi[C]
        piR = pi[A] + pi[G]

        beta = -1.0 / (2.0 * (piR*piY + kappa * (pi[A]*pi[G] + pi[C]*pi[T])))
        A_R = 1.0 + piR * (kappa - 1)
        A_Y = 1.0 + piY * (kappa - 1)
        eval = tf.stack([ # Eigenvalues 
            0.0,
            beta,
            beta * A_Y,
            beta * A_R
        ])
        evec = tf.transpose(tf.stack([ # Right eigenvectors as columns (rows of transpose)
            [1.0, 1.0, 1.0, 1.0],
            [1.0/piR, -1.0/piY, 1.0/piR, -1.0/piY],
            [0.0, pi[T]/piY, 0.0, -pi[C]/piY],
            [pi[G]/piR, 0.0, -pi[A]/piR, 0.0]
        ]))

        ivec = tf.stack([ # Left eigenvectors as rows
            [pi[A], pi[C], pi[G], pi[T]],
            [pi[A]*piY, -pi[C]*piR, pi[G]*piY, -pi[T]*piR],
            [0.0, 1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0, 0.0]
        ])

        return [tf.dtypes.cast(x, tf.dtypes.float64) for x in [evec, eval, ivec]]

    def q(self, frequencies, kappa):
        pi = frequencies
        return tf.stack([
            [-(pi[C] + kappa*pi[G] + pi[T]), pi[C], kappa*pi[G], pi[T]],
            [pi[A], -(pi[A] + pi[G] + kappa*pi[T]) , pi[G], kappa*pi[T]],
            [kappa*pi[A], pi[C], -(kappa*pi[A] + pi[C] + pi[T]), pi[T]],
            [pi[A], kappa*pi[C], pi[G], -(pi[A] + kappa*pi[C] + pi[G])]
        ])

    def q_param_differentials(self, frequencies, kappa):
        pi = frequencies
        return { 'kappa': tf.stack([
            [-pi[G], 0.0, pi[G], 0.0],
            [0.0, -pi[T], 0.0, pi[T]],
            [pi[A], 0.0, -pi[A], 0.0],
            [0.0, pi[C], 0.0, -pi[C]]
        ])}

    def q_frequency_differentials(self, frequencies, kappa):
        return tf.stack([
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [kappa, 0.0, -kappa, 0.0],
                [1.0, 0.0, 0.0, -1.0]
            ],
            [
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, kappa, 0.0, -kappa]
            ],
            [
                [-kappa, 0.0, kappa, 0.0],
                [0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0]
            ],
            [
                [-1.0, 0.0, 0.0, 1.0],
                [0.0, -kappa, 0.0, kappa],
                [0.0, 0.0, -1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0]
            ]
        ])

    def param_keys(self):
        return ['kappa']

class GTR(SubstitutionModel):
    def q(self, frequencies, rates):
        pi = frequencies
        return tf.stack([
            [-(rates[0]*pi[1] + rates[1]*pi[2] + rates[2]*pi[3]), rates[0]*pi[1], rates[1]*pi[2], rates[2]*pi[3]],
            [rates[0]*pi[0], -(rates[0]*pi[0] + rates[3]*pi[2] + rates[4]*pi[3]), rates[3]*pi[2], rates[4]*pi[3]],
            [rates[1]*pi[0], rates[3]*pi[1], -(rates[1]*pi[0] + rates[3]*pi[1] + rates[5]*pi[3]), rates[5]*pi[3]],
            [rates[2]*pi[0], rates[4]*pi[1], rates[5]*pi[2], -(rates[2]*pi[0] + rates[4]*pi[1] + rates[5]*pi[2])]     
        ])

    def eigen(self, frequencies, rates):
        q = q(frequencies, rates)
        eval, evec = tf.linalg.eigh(q)
        ivec = tf.linalg.inv(evec)
        return (evec, eval, ivec)

def normalising_constant(q, pi):
    return -tf.reduce_sum(tf.linalg.diag_part(q) * pi)

def normalise(q, pi):
    return q / normalising_constant(q, pi)

def normalise_and_constant(q, pi):
    norm_const = normalising_constant(q, pi)
    return q / norm_const, norm_const

def normalised_differential(q_diff, q_norm, norm_const, pi, frequency_index=None, q=None):
    norm_grad = normalising_constant(q_diff, pi)
    if frequency_index is not None:
        norm_grad = norm_grad - q[frequency_index, frequency_index]
    return (q_diff - q_norm * norm_grad) / norm_const

def transition_probs(eigendecomposition, t):
    U, lambd, Vt = eigendecomposition
    diag = tf.linalg.diag(tf.exp(tf.expand_dims(t, 1) * tf.expand_dims(lambd, 0)))
    return tf.reduce_sum(tf.reshape(U, [1, 4, 4, 1, 1]) * tf.reshape(diag, [-1, 1, 4, 4, 1]) * tf.reshape(Vt, [1, 1, 1, 4, 4]), axis=[2, 3])

def transition_probs_differential(q_diff, eigendecomposition, t, inv_mult=True):
    evec, eval, ivec = eigendecomposition
    g = tf.linalg.matmul(tf.linalg.matmul(ivec, q_diff), evec)
    eval_i = tf.reshape(eval, [-1, 1, 1])
    eval_j = tf.reshape(eval, [1, -1, 1])
    t_b = tf.reshape(t, [1, 1, -1])
    if inv_mult:
        G_diag = tf.expand_dims(tf.linalg.diag_part(g), 1) * tf.expand_dims(t, 0)
        G_non_diag = tf.expand_dims(g, 2) * (1.0 - tf.math.exp((eval_j - eval_i)*t_b)) / (eval_i - eval_j)
    else:
        t_d = tf.expand_dims(t, 0)
        G_diag = tf.expand_dims(tf.linalg.diag_part(g), 1) * t_d * tf.math.exp(tf.expand_dims(eval, 1) * t_d)
        G_non_diag = tf.expand_dims(g, 2) * (tf.math.exp(eval_i * t_b) - tf.math.exp(eval_j * t_b)) / (eval_i - eval_j)
    indices = tf.range(4)
    diag_indices = tf.stack([indices, indices], axis=1)
    G = tf.transpose(tf.tensor_scatter_nd_update(G_non_diag, diag_indices, G_diag), perm=[2, 0, 1])
    return tf.linalg.matmul(tf.linalg.matmul(evec, G), ivec)
    
     

