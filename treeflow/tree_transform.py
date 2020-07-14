import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.tf_util
import numpy as np
import libsbn
from treeflow import DEFAULT_FLOAT_DTYPE_TF, DEFAULT_FLOAT_DTYPE_NP

class ParentCorrelation(tfp.bijectors.ScaleMatvecLU):
    def __init__(self, parent_indices, beta, name='ParentAffine'):
        non_root_count = parent_indices.shape[-1]
        node_count = non_root_count + 1
        perm = tf.range(node_count)
        indices = tf.concat([tf.expand_dims(tf.range(non_root_count), 1), tf.expand_dims(tf.convert_to_tensor(parent_indices, tf.int32), 1)], axis=1)
        build_triu = lambda beta: tf.eye(node_count) + tf.scatter_nd(indices, beta, [node_count, node_count])
        super(ParentCorrelation, self).__init__(tfp.util.DeferredTensor(beta, build_triu, shape=[node_count, node_count]), perm, name=name)

class BranchBreaking(tfp.bijectors.Bijector): # TODO: Broadcast over batch_dims
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='BranchBreaking'):
        super(BranchBreaking, self).__init__(forward_min_event_ndims=1,name=name)
        self.parent_indices = parent_indices
        self.preorder_node_indices = preorder_node_indices # Don't include root
        self.anchor_heights = tf.zeros(len(preorder_node_indices) + 1, dtype=tf.dtypes.float64) if anchor_heights is None else anchor_heights

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
        return treeflow.tf_util.vectorize_1d_if_needed(self._forward_1d, x, x.shape.rank - 1)

    def _inverse_1d(self, y):
        return (y - self.anchor_heights) / tf.concat([(tf.gather(y, self.parent_indices) - self.anchor_heights[:-1]), tf.ones((1,), dtype=tf.dtypes.float64)], 0)

    def _inverse(self, y):
        return treeflow.tf_util.vectorize_1d_if_needed(self._inverse_1d, y, y.shape.rank - 1)

    def _inverse_log_det_jacobian_1d(self, y):
        return -tf.reduce_sum(tf.math.log(tf.gather(y, self.parent_indices) - self.anchor_heights[:-1]))

    def _inverse_log_det_jacobian(self, y):
        return treeflow.tf_util.vectorize_1d_if_needed(self._inverse_log_det_jacobian_1d, y, y.shape.rank - 1)

class Ratio(BranchBreaking):
    def __init__(self, inst, *args, name='BranchBreaking', **kwargs):
        super(Ratio, self).__init__(*args, **kwargs)
        self.inst = inst
        self.tree = inst.tree_collection.trees[0]
        self.node_height_state = np.array(self.tree.node_heights, copy=False)

    def _forward_1d_numpy(self, x): # TODO: Should we do vectorization in Numpy or TF land?
        self.tree.set_node_heights_via_height_ratios(x)
        return self.node_height_state[-x.shape[-1]:].astype(x.dtype)

    def _ratio_gradient_numpy(self, heights, dheights):
        self.node_height_state[-heights.shape[-1]:] = heights
        return np.array(libsbn.ratio_gradient_of_height_gradient(self.tree, dheights), dtype=heights.dtype)

    def _forward_1d(self, x):
        @tf.custom_gradient
        def libsbn_tf_func(x):
            heights = tf.numpy_function(self._forward_1d_numpy, [x], DEFAULT_FLOAT_DTYPE_TF)
            def grad(dheights):
                return tf.numpy_function(self._ratio_gradient_numpy, [heights, dheights], DEFAULT_FLOAT_DTYPE_TF)
            return heights, grad
        return libsbn_tf_func(x)

class TreeChain(tfp.bijectors.Chain):
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='TreeChain', inst=None):
        ratio_bijector = (
            BranchBreaking(parent_indices, preorder_node_indices, anchor_heights=anchor_heights)
            if inst is None
            else Ratio(inst, parent_indices, preorder_node_indices, anchor_heights=anchor_heights)
        )
        blockwise = tfp.bijectors.Blockwise(
            [tfp.bijectors.Sigmoid(), tfp.bijectors.Exp()],
            block_sizes=tf.concat([parent_indices.shape, [1]], 0)
        )
        self.ratio_bijector = ratio_bijector
        self.to_ratio_bijector = blockwise
        super(TreeChain, self).__init__([ratio_bijector, blockwise], name=name)

class FixedLeafHeightDistribution(tfp.distributions.Blockwise):
    def __init__(self, node_height_distribution, leaf_heights):
        self.node_height_distribution = node_height_distribution
        self.leaf_heights = leaf_heights
        super(FixedLeafHeightDistribution, self).__init__([
            tfp.distributions.Independent(tfp.distributions.Deterministic(leaf_heights), reinterpreted_batch_ndims=1),
            node_height_distribution
        ])

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
        self.height_distribution = height_distribution

    @property
    def reparameterization_type(self): # Hack to allow VI
        return dict(heights=self.heights_reparam, topology={ key: tfp.distributions.FULLY_REPARAMETERIZED for key in self.topology_keys })

    
