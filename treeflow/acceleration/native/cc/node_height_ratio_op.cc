// Native TensorFlow custom op implementing the node-height ratio transform and
// its analytic reverse-mode gradient.
//
// The transform maps the per-internal-node height ratios used by variational /
// HMC inference to the actual node heights of a time tree. It is a preorder
// (root-to-leaves) tree traversal: the root height is read directly, and every
// other internal node's height is placed a fraction (its ratio) of the way
// between its anchor height and its parent's height. This is the compiled
// counterpart of the reference ``tf.TensorArray`` loop in
// ``treeflow.traversal.ratio_transform.ratios_to_node_heights``.
//
// Forward recursion (root r = preorder_indices[0] = node_count - 1):
//   h[r] = ratios[r] + anchor[r]
//   h[i] = (h[parent[i]] - anchor[i]) * ratios[i] + anchor[i]   (i != r)
//
// The forward op outputs the node heights; the backward op consumes them (so it
// has the parent heights to hand without recomputing the forward traversal) and
// walks the same nodes in reverse preorder -- children before parents -- to
// accumulate the height adjoints, exactly mirroring how the likelihood op's
// backward pass reuses its saved partials.
//
// Reverse-mode (with h[] the saved forward heights and g[] = dL/dh):
//   process nodes children-before-parents, seeding g[i] = grad_heights[i]:
//     non-root i:  dL/dratios[i] = g[i] * (h[parent[i]] - anchor[i])
//                  dL/danchor[i] = g[i] * (1 - ratios[i])
//                  g[parent[i]] += g[i] * ratios[i]
//     root r:      dL/dratios[r] = g[r]
//                  dL/danchor[r] = g[r]
//
// Layout conventions (row-major):
//   ratios          [B, N]   per-internal-node height ratios
//   anchor_heights  [B, N]   per-internal-node anchor (lower-bound) heights
//   preorder_indices [N]     internal node ids, parents-before-children (root 1st)
//   parent_indices   [.]     parent internal-node id per internal node (root unused)
// where B is the flattened batch (samples * sites * ...) and N = L - 1 internal
// nodes. The Python wrapper broadcasts ratios / anchor_heights to a common batch
// and flattens it, so both arrive as [B, N] here.
//
// Outputs:
//   heights [B, N]  node heights, indexed by internal node id (same order as in).

#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tree_traversal.h"

using namespace tensorflow;
using treeflow::ReadIndices;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("NodeHeightRatio")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("ratios: T")
    .Input("anchor_heights: T")
    .Input("preorder_indices: Tindex")
    .Input("parent_indices: Tindex")
    .Output("heights: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle ratios = c->input(0);  // [B, N]
      TF_RETURN_IF_ERROR(c->WithRank(ratios, 2, &ratios));
      c->set_output(0, ratios);
      return OkStatus();
    });

REGISTER_OP("NodeHeightRatioGrad")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("grad_heights: T")
    .Input("heights: T")
    .Input("ratios: T")
    .Input("anchor_heights: T")
    .Input("preorder_indices: Tindex")
    .Input("parent_indices: Tindex")
    .Output("grad_ratios: T")
    .Output("grad_anchor_heights: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));  // same shape as ratios
      c->set_output(1, c->input(3));  // same shape as anchor_heights
      return OkStatus();
    });

template <typename T, typename Tindex>
class NodeHeightRatioOp : public OpKernel {
 public:
  explicit NodeHeightRatioOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ratios_t = ctx->input(0);
    const Tensor& anchor_t = ctx->input(1);
    const Tensor& preorder_t = ctx->input(2);
    const Tensor& parent_t = ctx->input(3);

    OP_REQUIRES(ctx, ratios_t.dims() == 2,
                errors::InvalidArgument("ratios must be rank 2 [B,N]"));
    OP_REQUIRES(ctx, anchor_t.dims() == 2,
                errors::InvalidArgument("anchor_heights must be rank 2 [B,N]"));
    OP_REQUIRES(ctx, anchor_t.dim_size(0) == ratios_t.dim_size(0) &&
                         anchor_t.dim_size(1) == ratios_t.dim_size(1),
                errors::InvalidArgument(
                    "ratios and anchor_heights must have the same shape"));

    const int64_t B = ratios_t.dim_size(0);
    const int64_t N = ratios_t.dim_size(1);  // internal node count (L - 1)

    OP_REQUIRES(ctx, preorder_t.NumElements() == N,
                errors::InvalidArgument("preorder_indices must have N elements"));

    std::vector<int64_t> preorder, parent;
    ReadIndices<Tindex>(preorder_t, &preorder);
    ReadIndices<Tindex>(parent_t, &parent);

    Tensor* heights_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {B, N}, &heights_t));

    const T* ratios = ratios_t.flat<T>().data();
    const T* anchor = anchor_t.flat<T>().data();
    T* heights = heights_t->flat<T>().data();

    const int64_t root = preorder[0];

    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const T* ratios_b = ratios + b * N;
        const T* anchor_b = anchor + b * N;
        T* heights_b = heights + b * N;

        // Root: height read directly (preorder visits it first).
        heights_b[root] = ratios_b[root] + anchor_b[root];

        // Remaining nodes: parent already placed (parents precede children).
        for (int64_t idx = 1; idx < N; ++idx) {
          const int64_t i = preorder[idx];
          const int64_t p = parent[i];
          heights_b[i] =
              (heights_b[p] - anchor_b[i]) * ratios_b[i] + anchor_b[i];
        }
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, N, work);
  }
};

template <typename T, typename Tindex>
class NodeHeightRatioGradOp : public OpKernel {
 public:
  explicit NodeHeightRatioGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_heights_t = ctx->input(0);
    const Tensor& heights_t = ctx->input(1);
    const Tensor& ratios_t = ctx->input(2);
    const Tensor& anchor_t = ctx->input(3);
    const Tensor& preorder_t = ctx->input(4);
    const Tensor& parent_t = ctx->input(5);

    const int64_t B = ratios_t.dim_size(0);
    const int64_t N = ratios_t.dim_size(1);

    OP_REQUIRES(ctx, preorder_t.NumElements() == N,
                errors::InvalidArgument("preorder_indices must have N elements"));

    std::vector<int64_t> preorder, parent;
    ReadIndices<Tindex>(preorder_t, &preorder);
    ReadIndices<Tindex>(parent_t, &parent);

    Tensor* grad_ratios_t = nullptr;
    Tensor* grad_anchor_t = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, ratios_t.shape(), &grad_ratios_t));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, anchor_t.shape(), &grad_anchor_t));

    const T* grad_heights = grad_heights_t.flat<T>().data();
    const T* heights = heights_t.flat<T>().data();
    const T* ratios = ratios_t.flat<T>().data();
    const T* anchor = anchor_t.flat<T>().data();
    T* grad_ratios = grad_ratios_t->flat<T>().data();
    T* grad_anchor = grad_anchor_t->flat<T>().data();

    const int64_t root = preorder[0];

    auto work = [&](int64_t begin, int64_t end) {
      std::vector<T> ybar(N);  // height adjoints g[i] = dL/dh[i]
      for (int64_t b = begin; b < end; ++b) {
        const T* grad_heights_b = grad_heights + b * N;
        const T* heights_b = heights + b * N;
        const T* ratios_b = ratios + b * N;
        const T* anchor_b = anchor + b * N;
        T* grad_ratios_b = grad_ratios + b * N;
        T* grad_anchor_b = grad_anchor + b * N;

        for (int64_t i = 0; i < N; ++i) ybar[i] = grad_heights_b[i];

        // Reverse preorder: children before parents, so a node's adjoint is
        // complete (its children have pushed their contributions) before it is
        // read. preorder[0] is the root, hence processed last (idx == 0).
        for (int64_t idx = N - 1; idx >= 1; --idx) {
          const int64_t i = preorder[idx];
          const int64_t p = parent[i];
          const T gi = ybar[i];
          grad_ratios_b[i] = gi * (heights_b[p] - anchor_b[i]);
          grad_anchor_b[i] = gi * (T(1) - ratios_b[i]);
          ybar[p] += gi * ratios_b[i];
        }
        grad_ratios_b[root] = ybar[root];
        grad_anchor_b[root] = ybar[root];
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, N, work);
  }
};

#define REGISTER_CPU(T, Tindex)                                  \
  REGISTER_KERNEL_BUILDER(Name("NodeHeightRatio")                \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<T>("T")            \
                              .TypeConstraint<Tindex>("Tindex"), \
                          NodeHeightRatioOp<T, Tindex>);         \
  REGISTER_KERNEL_BUILDER(Name("NodeHeightRatioGrad")            \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<T>("T")            \
                              .TypeConstraint<Tindex>("Tindex"), \
                          NodeHeightRatioGradOp<T, Tindex>);

REGISTER_CPU(float, int32)
REGISTER_CPU(float, int64_t)
REGISTER_CPU(double, int32)
REGISTER_CPU(double, int64_t)

#undef REGISTER_CPU
