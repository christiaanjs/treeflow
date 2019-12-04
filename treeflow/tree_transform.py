import tensorflow as tf
import tensorflow_probability as tfp

class BranchBreaking(tfp.bijectors.Bijector):
    def __init__(self, parent_indices, preorder_node_indices, name='branch_breaking'):
        super(BranchBreaking, self).__init__(forward_min_event_ndims=1,name=name)
        self.parent_indices = parent_indices
        self.preorder_node_indices = preorder_node_indices

    def _forward(self, x):
        def f(out, elems):
            node_index, parent_index = elems
            return tf.tensor_scatter_nd_update(out, tf.reshape(node_index, [1, 1]), tf.expand_dims(out[parent_index]*out[node_index], 0))
        return tf.scan(f, (self.preorder_node_indices, tf.gather(self.parent_indices, self.preorder_node_indices)), x)[-1]

    def _inverse(self, y):
        return y / tf.concat([tf.gather(y, self.parent_indices), tf.ones((1,), dtype=tf.dtypes.float64)], 0)

    def _inverse_log_det_jacobian(self, y):
        return -tf.reduce_sum(tf.math.log(tf.gather(y, self.parent_indices)))


class TreeChain(tfp.bijectors.Chain):
    def __init__(self, parent_indices, preorder_node_indices, name='tree_chain'):
        branch_breaking = BranchBreaking(parent_indices, preorder_node_indices)
        blockwise = tfp.bijectors.Blockwise(
            [tfp.bijectors.Sigmoid(), tfp.bijectors.Exp()],
            block_sizes=tf.concat([parent_indices.shape, [1]])
        )
        super(TreeChain, self).__init__([branch_breaking, blockwise], name=name)