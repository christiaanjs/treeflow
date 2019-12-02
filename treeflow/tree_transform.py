import tensorflow as tf
import tensorflow_probability as tfp

class BranchBreaking(tfp.bijectors.Bijector):
    def __init__(self, parent_indices, preorder_node_indices, validate_args=False, name='tree_transform'):
        super(BranchBreaking, self).__init__(validate_args=validate_args, forward_min_event_ndims=1,name=name)
        self.parent_indices = parent_indices
        self.preorder_node_indices = preorder_node_indices

    def _forward(self, x):
        node_indices = tf.reshape(self.preorder_node_indices, [-1, 1, 1])
        parent_indices = tf.reshape(tf.gather(self.parent_indices, self.preorder_node_indices), [-1, 1, 1])
        def f(out, elems):
            node_index, parent_index = elems
            return tf.tensor_scatter_nd_update(out, node_index, tf.expand_dims(out[parent_index]*out[node_index], 0))
        return tf.scan(f, (node_indices, parent_indices), x)[-1]

    def _inverse(self, y):
        return y / tf.gather(y, self.parent_indices)

    def _log_det_jacobian(self, y):
        return -tf.reduce_sum(tf.math.log(tf.gather(y, self.parent_indices)))