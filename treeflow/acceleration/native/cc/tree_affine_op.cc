// Native TensorFlow custom ops implementing the two structured affine tree maps
// of the tree normalising flow, with analytic reverse-mode gradients.
//
// These are the compiled counterparts of the reference traversals in
// ``treeflow.traversal.tree_affine``:
//
//   preorder (root-to-tip), each node reading its parent's *output*:
//     y[r] = scale[r] * x[r] + shift[r]                            (root r)
//     y[i] = scale[i] * x[i] + shift[i] + pw[i] * y[parent[i]]     (i != r)
//
//   postorder (tip-to-root), each node reading its children's *outputs*:
//     w[i] = scale[i] * x[i] + shift[i]
//            + sum_{c : child_indices[i,c] >= 0} cw[i,c] * w[child_indices[i,c]]
//
// Both are linear and triangular in their own traversal order, with ``scale``
// on the diagonal -- the log-det-Jacobian is ``sum_i log scale[i]`` and needs no
// op, and the inverses are pure gathers done in Python. So only the forward
// sweeps are compiled here, exactly as for the node-height ratio op.
//
// Reverse-mode. Each backward op consumes the saved forward output (so parent /
// child values are to hand without recomputing the sweep) and walks the nodes in
// the order that completes an adjoint before it is read -- reverse preorder
// (children before parents) for the preorder map, reverse postorder (parents
// before children) for the postorder map:
//
//   preorder, with g[] = dL/dy seeded from grad_y, non-root i:
//     dL/dx[i] = g[i]*scale[i];  dL/dscale[i] = g[i]*x[i];  dL/dshift[i] = g[i]
//     dL/dpw[i] = g[i]*y[parent[i]];  g[parent[i]] += g[i]*pw[i]
//   the root's pw entry is unused by the forward map, so its gradient is 0.
//
//   postorder, with g[] = dL/dw seeded from grad_w, every internal i:
//     dL/dx[i] = g[i]*scale[i];  dL/dscale[i] = g[i]*x[i];  dL/dshift[i] = g[i]
//     for each internal child c: dL/dcw[i,c] = g[i]*w[c];  g[c] += g[i]*cw[i,c]
//   leaf-child slots carry no coordinate, so their cw gradient is 0.
//
// Layout conventions (row-major), B the flattened batch and N = L-1 internal
// nodes; the Python wrapper broadcasts every input to a common batch and
// flattens it, so all arrive batched here:
//   x, scale, shift, parent_weight  [B, N]
//   child_weight                    [B, N, C]
//   preorder_indices  [N]     internal node ids, parents-before-children
//   postorder_indices [N]     internal node ids, children-before-parents
//   parent_indices    [N-1]   parent internal-node id of each non-root internal
//                             node (the root's entry is never read)
//   child_indices     [N, C]  child ids in internal-node space; leaf children
//                             are negative and carry no coordinate

#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tree_traversal.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using treeflow::ReadIndices;

// ---------------------------------------------------------------------------
// Op registrations
// ---------------------------------------------------------------------------

REGISTER_OP("TreeAffinePreorder")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("x: T")
    .Input("scale: T")
    .Input("shift: T")
    .Input("parent_weight: T")
    .Input("preorder_indices: Tindex")
    .Input("parent_indices: Tindex")
    .Output("y: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x = c->input(0);  // [B, N]
      TF_RETURN_IF_ERROR(c->WithRank(x, 2, &x));
      c->set_output(0, x);
      return OkStatus();
    });

REGISTER_OP("TreeAffinePreorderGrad")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("grad_y: T")
    .Input("y: T")
    .Input("x: T")
    .Input("scale: T")
    .Input("shift: T")
    .Input("parent_weight: T")
    .Input("preorder_indices: Tindex")
    .Input("parent_indices: Tindex")
    .Output("grad_x: T")
    .Output("grad_scale: T")
    .Output("grad_shift: T")
    .Output("grad_parent_weight: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      c->set_output(3, c->input(5));
      return OkStatus();
    });

REGISTER_OP("TreeAffinePostorder")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("x: T")
    .Input("scale: T")
    .Input("shift: T")
    .Input("child_weight: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Output("w: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x = c->input(0);  // [B, N]
      TF_RETURN_IF_ERROR(c->WithRank(x, 2, &x));
      c->set_output(0, x);
      return OkStatus();
    });

REGISTER_OP("TreeAffinePostorderGrad")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("grad_w: T")
    .Input("w: T")
    .Input("x: T")
    .Input("scale: T")
    .Input("shift: T")
    .Input("child_weight: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Output("grad_x: T")
    .Output("grad_scale: T")
    .Output("grad_shift: T")
    .Output("grad_child_weight: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      c->set_output(3, c->input(5));
      return OkStatus();
    });

// ---------------------------------------------------------------------------
// Preorder (root-to-tip) affine
// ---------------------------------------------------------------------------

template <typename T, typename Tindex>
class TreeAffinePreorderOp : public OpKernel {
 public:
  explicit TreeAffinePreorderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x_t = ctx->input(0);
    const Tensor& scale_t = ctx->input(1);
    const Tensor& shift_t = ctx->input(2);
    const Tensor& weight_t = ctx->input(3);
    const Tensor& preorder_t = ctx->input(4);
    const Tensor& parent_t = ctx->input(5);

    OP_REQUIRES(ctx, x_t.dims() == 2,
                errors::InvalidArgument("x must be rank 2 [B,N]"));
    for (int i = 1; i < 4; ++i) {
      OP_REQUIRES(ctx, ctx->input(i).shape() == x_t.shape(),
                  errors::InvalidArgument(
                      "scale, shift and parent_weight must have x's shape"));
    }

    const int64_t B = x_t.dim_size(0);
    const int64_t N = x_t.dim_size(1);

    OP_REQUIRES(ctx, preorder_t.NumElements() == N,
                errors::InvalidArgument("preorder_indices must have N elements"));
    OP_REQUIRES(
        ctx, parent_t.NumElements() >= N - 1,
        errors::InvalidArgument("parent_indices must have at least N-1 elements"));

    std::vector<int64_t> preorder, parent;
    ReadIndices<Tindex>(preorder_t, &preorder);
    ReadIndices<Tindex>(parent_t, &parent);

    Tensor* y_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {B, N}, &y_t));

    const T* x = x_t.flat<T>().data();
    const T* scale = scale_t.flat<T>().data();
    const T* shift = shift_t.flat<T>().data();
    const T* weight = weight_t.flat<T>().data();
    T* y = y_t->flat<T>().data();

    const int64_t root = preorder[0];

    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const T* x_b = x + b * N;
        const T* scale_b = scale + b * N;
        const T* shift_b = shift + b * N;
        const T* weight_b = weight + b * N;
        T* y_b = y + b * N;

        // Root: no parent term (preorder visits it first).
        y_b[root] = scale_b[root] * x_b[root] + shift_b[root];

        for (int64_t idx = 1; idx < N; ++idx) {
          const int64_t i = preorder[idx];
          const int64_t p = parent[i];
          y_b[i] = scale_b[i] * x_b[i] + shift_b[i] + weight_b[i] * y_b[p];
        }
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, N, work);
  }
};

template <typename T, typename Tindex>
class TreeAffinePreorderGradOp : public OpKernel {
 public:
  explicit TreeAffinePreorderGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_y_t = ctx->input(0);
    const Tensor& y_t = ctx->input(1);
    const Tensor& x_t = ctx->input(2);
    const Tensor& scale_t = ctx->input(3);
    const Tensor& weight_t = ctx->input(5);
    const Tensor& preorder_t = ctx->input(6);
    const Tensor& parent_t = ctx->input(7);

    const int64_t B = x_t.dim_size(0);
    const int64_t N = x_t.dim_size(1);

    OP_REQUIRES(ctx, preorder_t.NumElements() == N,
                errors::InvalidArgument("preorder_indices must have N elements"));

    std::vector<int64_t> preorder, parent;
    ReadIndices<Tindex>(preorder_t, &preorder);
    ReadIndices<Tindex>(parent_t, &parent);

    Tensor* grad_x_t = nullptr;
    Tensor* grad_scale_t = nullptr;
    Tensor* grad_shift_t = nullptr;
    Tensor* grad_weight_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_t.shape(), &grad_x_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, x_t.shape(), &grad_scale_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, x_t.shape(), &grad_shift_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, x_t.shape(), &grad_weight_t));

    const T* grad_y = grad_y_t.flat<T>().data();
    const T* y = y_t.flat<T>().data();
    const T* x = x_t.flat<T>().data();
    const T* scale = scale_t.flat<T>().data();
    const T* weight = weight_t.flat<T>().data();
    T* grad_x = grad_x_t->flat<T>().data();
    T* grad_scale = grad_scale_t->flat<T>().data();
    T* grad_shift = grad_shift_t->flat<T>().data();
    T* grad_weight = grad_weight_t->flat<T>().data();

    const int64_t root = preorder[0];

    auto work = [&](int64_t begin, int64_t end) {
      std::vector<T> ybar(N);
      for (int64_t b = begin; b < end; ++b) {
        const T* grad_y_b = grad_y + b * N;
        const T* y_b = y + b * N;
        const T* x_b = x + b * N;
        const T* scale_b = scale + b * N;
        const T* weight_b = weight + b * N;
        T* grad_x_b = grad_x + b * N;
        T* grad_scale_b = grad_scale + b * N;
        T* grad_shift_b = grad_shift + b * N;
        T* grad_weight_b = grad_weight + b * N;

        for (int64_t i = 0; i < N; ++i) ybar[i] = grad_y_b[i];

        // Reverse preorder: a node's adjoint is complete (all of its children
        // have pushed their contributions) before it is read.
        for (int64_t idx = N - 1; idx >= 1; --idx) {
          const int64_t i = preorder[idx];
          const int64_t p = parent[i];
          const T gi = ybar[i];
          grad_x_b[i] = gi * scale_b[i];
          grad_scale_b[i] = gi * x_b[i];
          grad_shift_b[i] = gi;
          grad_weight_b[i] = gi * y_b[p];
          ybar[p] += gi * weight_b[i];
        }
        const T gr = ybar[root];
        grad_x_b[root] = gr * scale_b[root];
        grad_scale_b[root] = gr * x_b[root];
        grad_shift_b[root] = gr;
        grad_weight_b[root] = T(0);  // the root has no parent term
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, N, work);
  }
};

// ---------------------------------------------------------------------------
// Postorder (tip-to-root) affine
// ---------------------------------------------------------------------------

template <typename T, typename Tindex>
class TreeAffinePostorderOp : public OpKernel {
 public:
  explicit TreeAffinePostorderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x_t = ctx->input(0);
    const Tensor& scale_t = ctx->input(1);
    const Tensor& shift_t = ctx->input(2);
    const Tensor& weight_t = ctx->input(3);
    const Tensor& postorder_t = ctx->input(4);
    const Tensor& child_t = ctx->input(5);

    OP_REQUIRES(ctx, x_t.dims() == 2,
                errors::InvalidArgument("x must be rank 2 [B,N]"));
    OP_REQUIRES(ctx, scale_t.shape() == x_t.shape() &&
                         shift_t.shape() == x_t.shape(),
                errors::InvalidArgument("scale and shift must have x's shape"));
    OP_REQUIRES(ctx, weight_t.dims() == 3,
                errors::InvalidArgument("child_weight must be rank 3 [B,N,C]"));

    const int64_t B = x_t.dim_size(0);
    const int64_t N = x_t.dim_size(1);
    const int64_t C = weight_t.dim_size(2);

    OP_REQUIRES(ctx,
                weight_t.dim_size(0) == B && weight_t.dim_size(1) == N,
                errors::InvalidArgument("child_weight must be [B,N,C]"));
    OP_REQUIRES(ctx, postorder_t.NumElements() == N,
                errors::InvalidArgument("postorder_indices must have N elements"));
    OP_REQUIRES(ctx, child_t.NumElements() == N * C,
                errors::InvalidArgument("child_indices must have N*C elements"));

    std::vector<int64_t> postorder, child;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);

    Tensor* w_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {B, N}, &w_t));

    const T* x = x_t.flat<T>().data();
    const T* scale = scale_t.flat<T>().data();
    const T* shift = shift_t.flat<T>().data();
    const T* weight = weight_t.flat<T>().data();
    T* w = w_t->flat<T>().data();

    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const T* x_b = x + b * N;
        const T* scale_b = scale + b * N;
        const T* shift_b = shift + b * N;
        const T* weight_b = weight + b * N * C;
        T* w_b = w + b * N;

        // Children before parents, so every internal child is already placed.
        for (int64_t idx = 0; idx < N; ++idx) {
          const int64_t i = postorder[idx];
          T acc = scale_b[i] * x_b[i] + shift_b[i];
          for (int64_t c = 0; c < C; ++c) {
            const int64_t ci = child[i * C + c];
            if (ci >= 0) acc += weight_b[i * C + c] * w_b[ci];
          }
          w_b[i] = acc;
        }
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, N * C, work);
  }
};

template <typename T, typename Tindex>
class TreeAffinePostorderGradOp : public OpKernel {
 public:
  explicit TreeAffinePostorderGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_w_t = ctx->input(0);
    const Tensor& w_t = ctx->input(1);
    const Tensor& x_t = ctx->input(2);
    const Tensor& scale_t = ctx->input(3);
    const Tensor& weight_t = ctx->input(5);
    const Tensor& postorder_t = ctx->input(6);
    const Tensor& child_t = ctx->input(7);

    const int64_t B = x_t.dim_size(0);
    const int64_t N = x_t.dim_size(1);
    const int64_t C = weight_t.dim_size(2);

    OP_REQUIRES(ctx, postorder_t.NumElements() == N,
                errors::InvalidArgument("postorder_indices must have N elements"));
    OP_REQUIRES(ctx, child_t.NumElements() == N * C,
                errors::InvalidArgument("child_indices must have N*C elements"));

    std::vector<int64_t> postorder, child;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);

    Tensor* grad_x_t = nullptr;
    Tensor* grad_scale_t = nullptr;
    Tensor* grad_shift_t = nullptr;
    Tensor* grad_weight_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_t.shape(), &grad_x_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, x_t.shape(), &grad_scale_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, x_t.shape(), &grad_shift_t));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(3, weight_t.shape(), &grad_weight_t));

    const T* grad_w = grad_w_t.flat<T>().data();
    const T* w = w_t.flat<T>().data();
    const T* x = x_t.flat<T>().data();
    const T* scale = scale_t.flat<T>().data();
    const T* weight = weight_t.flat<T>().data();
    T* grad_x = grad_x_t->flat<T>().data();
    T* grad_scale = grad_scale_t->flat<T>().data();
    T* grad_shift = grad_shift_t->flat<T>().data();
    T* grad_weight = grad_weight_t->flat<T>().data();

    auto work = [&](int64_t begin, int64_t end) {
      std::vector<T> wbar(N);
      for (int64_t b = begin; b < end; ++b) {
        const T* grad_w_b = grad_w + b * N;
        const T* w_b = w + b * N;
        const T* x_b = x + b * N;
        const T* scale_b = scale + b * N;
        const T* weight_b = weight + b * N * C;
        T* grad_x_b = grad_x + b * N;
        T* grad_scale_b = grad_scale + b * N;
        T* grad_shift_b = grad_shift + b * N;
        T* grad_weight_b = grad_weight + b * N * C;

        for (int64_t i = 0; i < N; ++i) wbar[i] = grad_w_b[i];

        // Reverse postorder: parents before children, so a node's adjoint is
        // complete (its parent has pushed its contribution) before it is read.
        for (int64_t idx = N - 1; idx >= 0; --idx) {
          const int64_t i = postorder[idx];
          const T gi = wbar[i];
          grad_x_b[i] = gi * scale_b[i];
          grad_scale_b[i] = gi * x_b[i];
          grad_shift_b[i] = gi;
          for (int64_t c = 0; c < C; ++c) {
            const int64_t ci = child[i * C + c];
            if (ci >= 0) {
              grad_weight_b[i * C + c] = gi * w_b[ci];
              wbar[ci] += gi * weight_b[i * C + c];
            } else {
              // A leaf child carries no coordinate: its weight is unused.
              grad_weight_b[i * C + c] = T(0);
            }
          }
        }
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, N * C, work);
  }
};

#define REGISTER_CPU(T, Tindex)                                     \
  REGISTER_KERNEL_BUILDER(Name("TreeAffinePreorder")                \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tindex>("Tindex"),    \
                          TreeAffinePreorderOp<T, Tindex>);         \
  REGISTER_KERNEL_BUILDER(Name("TreeAffinePreorderGrad")            \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tindex>("Tindex"),    \
                          TreeAffinePreorderGradOp<T, Tindex>);     \
  REGISTER_KERNEL_BUILDER(Name("TreeAffinePostorder")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tindex>("Tindex"),    \
                          TreeAffinePostorderOp<T, Tindex>);        \
  REGISTER_KERNEL_BUILDER(Name("TreeAffinePostorderGrad")           \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tindex>("Tindex"),    \
                          TreeAffinePostorderGradOp<T, Tindex>);

REGISTER_CPU(float, int32)
REGISTER_CPU(float, int64_t)
REGISTER_CPU(double, int32)
REGISTER_CPU(double, int64_t)

#undef REGISTER_CPU
