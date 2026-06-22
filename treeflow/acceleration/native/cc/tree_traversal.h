// Shared host-side primitives for treeflow's native tree-traversal ops.
//
// Both native ops -- the Felsenstein pruning likelihood (postorder) and the
// node-height ratio transform (preorder) -- describe the tree topology to the
// kernel as integer index tensors and then walk the nodes in a fixed order on
// the host. The helpers here capture the parts of that machinery that are the
// same regardless of the traversal direction so the two op kernels can share
// them rather than each carrying a private copy.
//
// Index tensor conventions (all node ids are 0-based; a tree with L leaves has
// M = 2L-1 nodes and L-1 internal nodes):
//
//   postorder_indices [L-1]   internal node ids, children-before-parents. The
//                             last entry is the root. Used by the likelihood to
//                             visit a node only once its children's partials are
//                             ready.
//   preorder_indices  [L-1]   internal node ids, parents-before-children. The
//                             first entry is the root. Used by the ratio
//                             transform so a node's parent height is already set
//                             when the node is visited; the reverse walk
//                             (children-before-parents) is the gradient order.
//   child_indices     [L-1,C] child node ids per internal node (postorder rows),
//                             padded with negative ids in unused slots.
//   parent_indices    [.]     parent internal-node id per internal node (in the
//                             internal-id space used by the ratio transform);
//                             the root entry is never read.
//
// Both ops also flatten all leading (sample/site/category) batch dimensions to
// a single dimension B in their Python wrappers and shard the per-batch-element
// traversals across B with tensorflow::Shard.

#ifndef TREEFLOW_ACCELERATION_NATIVE_CC_TREE_TRAVERSAL_H_
#define TREEFLOW_ACCELERATION_NATIVE_CC_TREE_TRAVERSAL_H_

#include <cstdint>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace treeflow {

// Read an index tensor (int32 or int64) into an int64 vector. The op kernels
// accept either integer width (the ``Tindex`` op attribute) but the host-side
// traversal loops always index with int64, so this normalises once up front.
template <typename Tindex>
inline void ReadIndices(const ::tensorflow::Tensor& t,
                        std::vector<int64_t>* out) {
  auto flat = t.flat<Tindex>();
  out->resize(flat.size());
  for (int64_t i = 0; i < flat.size(); ++i) (*out)[i] = flat(i);
}

}  // namespace treeflow

#endif  // TREEFLOW_ACCELERATION_NATIVE_CC_TREE_TRAVERSAL_H_
