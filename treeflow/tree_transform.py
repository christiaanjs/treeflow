import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.tf_util

class BranchBreaking(tfp.bijectors.Bijector): # TODO: Broadcast over batch_dims
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='BranchBreaking'):
        super(BranchBreaking, self).__init__(forward_min_event_ndims=1,name=name)
        self.parent_indices = parent_indices
        self.preorder_node_indices = preorder_node_indices # Don't include root
        self.anchor_heights = tf.zeros(len(preorder_node_indices) + 1, dtype=tf.dtypes.float32) if anchor_heights is None else anchor_heights

    def _forward_1d(self, x):
        length = x.shape[-1]
        init = tf.scatter_nd([[length - 1]], tf.expand_dims(x[-1] + self.anchor_heights[-1], 0), self.anchor_heights.shape)
        def f(out, elems):
            node_index, parent_index, proportion, anchor_height = elems
            node_height = (out[parent_index] - anchor_height) * proportion + anchor_height
            return tf.tensor_scatter_nd_update(out, tf.reshape(node_index, [1, 1]), tf.expand_dims(node_height, 0))
        return tf.scan(
            f,
            (
                self.preorder_node_indices,
                tf.gather(self.parent_indices, self.preorder_node_indices),
                tf.gather(x, self.preorder_node_indices), tf.gather(self.anchor_heights, self.preorder_node_indices)
            ),
            init)[-1]

    def _forward(self, x):
        return treeflow.tf_util.vectorize_1d_if_needed(self._forward_1d, x)

    def _inverse_1d(self, y):
        return (y - self.anchor_heights) / tf.concat([(tf.gather(y, self.parent_indices) - self.anchor_heights[:-1]), tf.ones((1,), dtype=tf.dtypes.float32)], 0)

    def _inverse(self, y):
        return treeflow.tf_util.vectorize_1d_if_needed(self._inverse_1d, y)

    def _inverse_log_det_jacobian_1d(self, y):
        return -tf.reduce_sum(tf.math.log(tf.gather(y, self.parent_indices) - self.anchor_heights[:-1]))

    def _inverse_log_det_jacobian(self, y):
        return treeflow.tf_util.vectorize_1d_if_needed(self._inverse_log_det_jacobian_1d, y)


class TreeChain(tfp.bijectors.Chain):
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='TreeChain'):
        branch_breaking = BranchBreaking(parent_indices, preorder_node_indices, anchor_heights=anchor_heights)
        blockwise = tfp.bijectors.Blockwise(
            [tfp.bijectors.Sigmoid(), tfp.bijectors.Exp()],
            block_sizes=tf.concat([parent_indices.shape, [1]], 0)
        )
        super(TreeChain, self).__init__([branch_breaking, blockwise], name=name)

class FixedTopologyDistribution(tfp.distributions.JointDistributionNamed):
    def __init__(self, height_distribution, topology, name='FixedTopologyDistribution'):
        super(FixedTopologyDistribution, self).__init__(dict(
            topology=tfp.distributions.JointDistributionNamed({
                key: tfp.distributions.Independent(
                    tfp.distributions.Deterministic(loc=value),
                    reinterpreted_batch_ndims=height_distribution.event_shape.ndims
                ) for key, value in topology.items()
            }),
            heights=height_distribution
        ))
        self.topology_keys = topology.keys()
        self.heights_reparam = height_distribution.reparameterization_type

    @property
    def reparameterization_type(self): # Hack to allow VI
        return dict(heights=self.heights_reparam, topology={ key: tfp.distributions.FULLY_REPARAMETERIZED for key in self.topology_keys })

    
