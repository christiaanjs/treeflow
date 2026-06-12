// Native TensorFlow custom op implementing Felsenstein's pruning algorithm
// (the phylogenetic likelihood) and its analytic reverse-mode gradient with
// respect to the per-branch transition probability matrices.
//
// The forward op outputs both the per-site likelihoods and the partial
// likelihood vectors at every node.  The backward op consumes those saved
// partials so the gradient pass does not need to recompute the postorder
// traversal from scratch -- it reuses the forward partials exactly as a
// hand-written BEAGLE-style implementation would.
//
// Layout conventions (row-major):
//   sequences          [B,  L, S]      leaf partials (one-hot or ambiguity)
//   transition_probs   [Bt, M, S, S]   per-node transition matrices P[s, j]
//   frequencies        [Bf, S]         root state frequencies
//   postorder_indices  [I]             internal node ids in postorder
//   child_indices      [I, C]          child node ids per internal node
// where
//   B  = batch size (e.g. alignment sites)
//   L  = leaf count, M = 2L-1 total nodes, I = L-1 internal nodes
//   S  = state count, C = max children per internal node
//   Bt, Bf are either 1 (broadcast over B) or equal to B.
//
// Outputs:
//   site_likelihood [B]        per-batch likelihood (not log)
//   node_partials   [B, M, S]  partials at every node (saved for backward)

#include <algorithm>
#include <cmath>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("PhyloLikelihood")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("sequences: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Output("site_likelihood: T")
    .Output("node_partials: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle seq = c->input(0);     // [B, L, S]
      ShapeHandle probs = c->input(1);   // [Bt, M, S, S]
      TF_RETURN_IF_ERROR(c->WithRank(seq, 3, &seq));
      TF_RETURN_IF_ERROR(c->WithRank(probs, 4, &probs));
      DimensionHandle B = c->Dim(seq, 0);
      DimensionHandle L = c->Dim(seq, 1);
      DimensionHandle S = c->Dim(seq, 2);
      // Total node count Nn = 2L - 1 (transition_probs only stores the 2L-2
      // non-root branches, but partials are needed at every node).
      DimensionHandle Nn;
      TF_RETURN_IF_ERROR(c->Multiply(L, 2, &Nn));
      TF_RETURN_IF_ERROR(c->Subtract(Nn, 1, &Nn));
      c->set_output(0, c->Vector(B));
      c->set_output(1, c->MakeShape({B, Nn, S}));
      return OkStatus();
    });

REGISTER_OP("PhyloLikelihoodGrad")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("grad_site_likelihood: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("node_partials: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Output("grad_transition_probs: T")
    .Output("grad_frequencies: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));  // same shape as transition_probs
      c->set_output(1, c->input(2));  // same shape as frequencies
      return OkStatus();
    });

// Numerically stable variant: rescales the partial likelihood vector at every
// internal node by its per-site maximum, accumulating the log of the scale
// factors, and returns the per-site LOG likelihood. Also outputs the scaled
// partials and the scale factors so the backward op can reuse them.
REGISTER_OP("PhyloLikelihoodRescaled")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("sequences: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Output("site_log_likelihood: T")
    .Output("node_partials: T")
    .Output("node_scales: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle seq = c->input(0);
      ShapeHandle probs = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(seq, 3, &seq));
      TF_RETURN_IF_ERROR(c->WithRank(probs, 4, &probs));
      DimensionHandle B = c->Dim(seq, 0);
      DimensionHandle L = c->Dim(seq, 1);
      DimensionHandle S = c->Dim(seq, 2);
      DimensionHandle Nn;
      TF_RETURN_IF_ERROR(c->Multiply(L, 2, &Nn));
      TF_RETURN_IF_ERROR(c->Subtract(Nn, 1, &Nn));
      c->set_output(0, c->Vector(B));
      c->set_output(1, c->MakeShape({B, Nn, S}));
      c->set_output(2, c->MakeShape({B, Nn}));
      return OkStatus();
    });

REGISTER_OP("PhyloLikelihoodRescaledGrad")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Input("grad_site_log_likelihood: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("node_partials: T")
    .Input("node_scales: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Output("grad_transition_probs: T")
    .Output("grad_frequencies: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));  // same shape as transition_probs
      c->set_output(1, c->input(2));  // same shape as frequencies
      return OkStatus();
    });

namespace {

// Read an index tensor (int32 or int64) into an int64 vector.
template <typename Tindex>
inline void ReadIndices(const Tensor& t, std::vector<int64_t>* out) {
  auto flat = t.flat<Tindex>();
  out->resize(flat.size());
  for (int64_t i = 0; i < flat.size(); ++i) (*out)[i] = flat(i);
}

}  // namespace

template <typename T, typename Tindex>
class PhyloLikelihoodOp : public OpKernel {
 public:
  explicit PhyloLikelihoodOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sequences = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& postorder_t = ctx->input(3);
    const Tensor& child_t = ctx->input(4);

    OP_REQUIRES(ctx, sequences.dims() == 3,
                errors::InvalidArgument("sequences must be rank 3 [B,L,S]"));
    OP_REQUIRES(ctx, probs.dims() == 4,
                errors::InvalidArgument(
                    "transition_probs must be rank 4 [Bt,M,S,S]"));
    OP_REQUIRES(ctx, freqs.dims() == 2,
                errors::InvalidArgument("frequencies must be rank 2 [Bf,S]"));
    OP_REQUIRES(ctx, child_t.dims() == 2,
                errors::InvalidArgument("child_indices must be rank 2 [I,C]"));

    const int64_t B = sequences.dim_size(0);
    const int64_t L = sequences.dim_size(1);
    const int64_t S = sequences.dim_size(2);
    const int64_t Bt = probs.dim_size(0);
    const int64_t M = probs.dim_size(1);  // transition_probs node count (>= 2L-2)
    const int64_t Nn = 2 * L - 1;         // total node count (partials buffer)
    const int64_t Bf = freqs.dim_size(0);
    const int64_t I = postorder_t.dim_size(0);
    const int64_t C = child_t.dim_size(1);

    OP_REQUIRES(ctx, probs.dim_size(2) == S && probs.dim_size(3) == S,
                errors::InvalidArgument("transition_probs state mismatch"));
    OP_REQUIRES(ctx, freqs.dim_size(1) == S,
                errors::InvalidArgument("frequencies state mismatch"));
    OP_REQUIRES(ctx, Bt == 1 || Bt == B,
                errors::InvalidArgument("transition_probs batch must be 1 or B"));
    OP_REQUIRES(ctx, Bf == 1 || Bf == B,
                errors::InvalidArgument("frequencies batch must be 1 or B"));
    OP_REQUIRES(ctx, I == L - 1,
                errors::InvalidArgument("postorder length must be L-1"));
    OP_REQUIRES(ctx, M >= Nn - 1,
                errors::InvalidArgument(
                    "transition_probs must have at least 2L-2 nodes"));

    std::vector<int64_t> postorder, child;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);

    Tensor* site_ll = nullptr;
    Tensor* node_partials = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {B}, &site_ll));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {B, Nn, S}, &node_partials));

    const T* seq = sequences.flat<T>().data();
    const T* P = probs.flat<T>().data();
    const T* fr = freqs.flat<T>().data();
    T* ll = site_ll->flat<T>().data();
    T* part = node_partials->flat<T>().data();

    const int64_t root = postorder[I - 1];

    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = (Bt == 1) ? 0 : b;
        const int64_t bf = (Bf == 1) ? 0 : b;
        const T* seq_b = seq + b * L * S;
        const T* P_b = P + bt * M * S * S;
        const T* fr_b = fr + bf * S;
        T* part_b = part + b * Nn * S;

        // Leaves: copy one-hot / ambiguity partials directly.
        for (int64_t k = 0; k < L * S; ++k) part_b[k] = seq_b[k];

        // Internal nodes in postorder (children always precomputed).
        for (int64_t i = 0; i < I; ++i) {
          const int64_t v = postorder[i];
          T* pv = part_b + v * S;
          for (int64_t s = 0; s < S; ++s) pv[s] = T(1);
          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;  // padded child slot
            const T* Pc = P_b + cnode * S * S;
            const T* pc = part_b + cnode * S;
            for (int64_t s = 0; s < S; ++s) {
              T g = T(0);
              const T* Prow = Pc + s * S;
              for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j];
              pv[s] *= g;
            }
          }
        }

        // Root likelihood.
        const T* pr = part_b + root * S;
        T acc = T(0);
        for (int64_t s = 0; s < S; ++s) acc += fr_b[s] * pr[s];
        ll[b] = acc;
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    const int64_t cost = M * S * S;  // rough per-batch cost
    Shard(workers->num_threads, workers->workers, B, cost, work);
  }
};

template <typename T, typename Tindex>
class PhyloLikelihoodGradOp : public OpKernel {
 public:
  explicit PhyloLikelihoodGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_ll_t = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& node_partials = ctx->input(3);
    const Tensor& postorder_t = ctx->input(4);
    const Tensor& child_t = ctx->input(5);

    const int64_t B = grad_ll_t.dim_size(0);
    const int64_t Bt = probs.dim_size(0);
    const int64_t M = probs.dim_size(1);  // transition_probs node count
    const int64_t S = probs.dim_size(2);
    const int64_t Nn = node_partials.dim_size(1);  // total node count (2L-1)
    const int64_t Bf = freqs.dim_size(0);
    const int64_t I = postorder_t.dim_size(0);
    const int64_t C = child_t.dim_size(1);

    std::vector<int64_t> postorder, child;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);

    Tensor* grad_P_t = nullptr;
    Tensor* grad_F_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, probs.shape(), &grad_P_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, freqs.shape(), &grad_F_t));

    const T* grad_ll = grad_ll_t.flat<T>().data();
    const T* P = probs.flat<T>().data();
    const T* fr = freqs.flat<T>().data();
    const T* part = node_partials.flat<T>().data();
    T* gP = grad_P_t->flat<T>().data();
    T* gF = grad_F_t->flat<T>().data();

    const int64_t gP_size = Bt * M * S * S;
    const int64_t gF_size = Bf * S;
    for (int64_t k = 0; k < gP_size; ++k) gP[k] = T(0);
    for (int64_t k = 0; k < gF_size; ++k) gF[k] = T(0);

    const int64_t root = postorder[I - 1];
    const bool reduceP = (Bt == 1) && (B > 1);
    const bool reduceF = (Bf == 1) && (B > 1);

    mutex mu;  // guards reductions into the shared gP / gF

    auto work = [&](int64_t begin, int64_t end) {
      // Per-shard scratch.
      std::vector<T> bar(Nn * S);
      std::vector<T> gmat(C * S);
      std::vector<T> sib(C * S);
      std::vector<T> w(S);
      // Thread-local accumulators only when reducing a broadcast batch.
      std::vector<T> localP(reduceP ? M * S * S : 0, T(0));
      std::vector<T> localF(reduceF ? S : 0, T(0));

      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = (Bt == 1) ? 0 : b;
        const int64_t bf = (Bf == 1) ? 0 : b;
        const T* P_b = P + bt * M * S * S;
        const T* fr_b = fr + bf * S;
        const T* part_b = part + b * Nn * S;
        const T dll = grad_ll[b];

        T* gP_b = reduceP ? localP.data() : (gP + bt * M * S * S);
        T* gF_b = reduceF ? localF.data() : (gF + bf * S);

        std::fill(bar.begin(), bar.end(), T(0));

        // Seed: L = sum_s fr[s] * part[root, s].
        const T* pr = part_b + root * S;
        for (int64_t s = 0; s < S; ++s) {
          bar[root * S + s] = dll * fr_b[s];
          gF_b[s] += dll * pr[s];
        }

        // Reverse postorder (parents before children).
        for (int64_t i = I - 1; i >= 0; --i) {
          const int64_t v = postorder[i];
          const T* barv = &bar[v * S];

          // g_c[s] = sum_j P_c[s,j] * part_c[j]; identity for padded slots.
          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) {
              for (int64_t s = 0; s < S; ++s) gmat[ci * S + s] = T(1);
              continue;
            }
            const T* Pc = P_b + cnode * S * S;
            const T* pc = part_b + cnode * S;
            for (int64_t s = 0; s < S; ++s) {
              T g = T(0);
              const T* Prow = Pc + s * S;
              for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j];
              gmat[ci * S + s] = g;
            }
          }

          // Exclusive products across siblings via prefix/suffix (no division,
          // robust to zeros).
          for (int64_t s = 0; s < S; ++s) {
            T pre = T(1);
            for (int64_t ci = 0; ci < C; ++ci) {
              sib[ci * S + s] = pre;
              pre *= gmat[ci * S + s];
            }
            T suf = T(1);
            for (int64_t ci = C - 1; ci >= 0; --ci) {
              sib[ci * S + s] *= suf;
              suf *= gmat[ci * S + s];
            }
          }

          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;
            for (int64_t s = 0; s < S; ++s) w[s] = barv[s] * sib[ci * S + s];
            const T* Pc = P_b + cnode * S * S;
            const T* pc = part_b + cnode * S;
            T* gPc = gP_b + cnode * S * S;
            // dL/dP_c[s,j] = w[s] * part_c[j]
            for (int64_t s = 0; s < S; ++s) {
              const T ws = w[s];
              T* gRow = gPc + s * S;
              for (int64_t j = 0; j < S; ++j) gRow[j] += ws * pc[j];
            }
            // Propagate adjoint to child partials.
            T* barc = &bar[cnode * S];
            for (int64_t j = 0; j < S; ++j) {
              T acc = T(0);
              for (int64_t s = 0; s < S; ++s) acc += w[s] * Pc[s * S + j];
              barc[j] += acc;
            }
          }
        }
      }

      if (reduceP || reduceF) {
        mutex_lock l(mu);
        if (reduceP)
          for (int64_t k = 0; k < M * S * S; ++k) gP[k] += localP[k];
        if (reduceF)
          for (int64_t k = 0; k < S; ++k) gF[k] += localF[k];
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    const int64_t cost = 2 * M * S * S;
    Shard(workers->num_threads, workers->workers, B, cost, work);
  }
};

// ===========================================================================
// Rescaled (numerically stable) variants
// ===========================================================================

template <typename T, typename Tindex>
class PhyloLikelihoodRescaledOp : public OpKernel {
 public:
  explicit PhyloLikelihoodRescaledOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sequences = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& postorder_t = ctx->input(3);
    const Tensor& child_t = ctx->input(4);

    OP_REQUIRES(ctx, sequences.dims() == 3,
                errors::InvalidArgument("sequences must be rank 3 [B,L,S]"));
    OP_REQUIRES(ctx, probs.dims() == 4,
                errors::InvalidArgument(
                    "transition_probs must be rank 4 [Bt,M,S,S]"));
    OP_REQUIRES(ctx, freqs.dims() == 2,
                errors::InvalidArgument("frequencies must be rank 2 [Bf,S]"));

    const int64_t B = sequences.dim_size(0);
    const int64_t L = sequences.dim_size(1);
    const int64_t S = sequences.dim_size(2);
    const int64_t Bt = probs.dim_size(0);
    const int64_t M = probs.dim_size(1);
    const int64_t Nn = 2 * L - 1;
    const int64_t Bf = freqs.dim_size(0);
    const int64_t I = postorder_t.dim_size(0);
    const int64_t C = child_t.dim_size(1);

    OP_REQUIRES(ctx, Bt == 1 || Bt == B,
                errors::InvalidArgument("transition_probs batch must be 1 or B"));
    OP_REQUIRES(ctx, Bf == 1 || Bf == B,
                errors::InvalidArgument("frequencies batch must be 1 or B"));
    OP_REQUIRES(ctx, I == L - 1,
                errors::InvalidArgument("postorder length must be L-1"));

    std::vector<int64_t> postorder, child;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);

    Tensor* site_ll = nullptr;
    Tensor* node_partials = nullptr;
    Tensor* node_scales = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {B}, &site_ll));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {B, Nn, S}, &node_partials));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {B, Nn}, &node_scales));

    const T* seq = sequences.flat<T>().data();
    const T* P = probs.flat<T>().data();
    const T* fr = freqs.flat<T>().data();
    T* ll = site_ll->flat<T>().data();
    T* part = node_partials->flat<T>().data();
    T* scales = node_scales->flat<T>().data();

    const int64_t root = postorder[I - 1];

    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = (Bt == 1) ? 0 : b;
        const int64_t bf = (Bf == 1) ? 0 : b;
        const T* seq_b = seq + b * L * S;
        const T* P_b = P + bt * M * S * S;
        const T* fr_b = fr + bf * S;
        T* part_b = part + b * Nn * S;
        T* scales_b = scales + b * Nn;

        for (int64_t k = 0; k < L * S; ++k) part_b[k] = seq_b[k];
        for (int64_t v = 0; v < Nn; ++v) scales_b[v] = T(1);

        T log_scale_sum = T(0);
        for (int64_t i = 0; i < I; ++i) {
          const int64_t v = postorder[i];
          T* pv = part_b + v * S;
          for (int64_t s = 0; s < S; ++s) pv[s] = T(1);
          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;
            const T* Pc = P_b + cnode * S * S;
            const T* pc = part_b + cnode * S;
            for (int64_t s = 0; s < S; ++s) {
              T g = T(0);
              const T* Prow = Pc + s * S;
              for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j];
              pv[s] *= g;
            }
          }
          // Rescale by the per-site maximum partial.
          T cmax = T(0);
          for (int64_t s = 0; s < S; ++s) cmax = std::max(cmax, pv[s]);
          if (cmax > T(0)) {
            const T inv = T(1) / cmax;
            for (int64_t s = 0; s < S; ++s) pv[s] *= inv;
            scales_b[v] = cmax;
            log_scale_sum += std::log(cmax);
          }
        }

        const T* pr = part_b + root * S;
        T acc = T(0);
        for (int64_t s = 0; s < S; ++s) acc += fr_b[s] * pr[s];
        ll[b] = std::log(acc) + log_scale_sum;
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, M * S * S, work);
  }
};

template <typename T, typename Tindex>
class PhyloLikelihoodRescaledGradOp : public OpKernel {
 public:
  explicit PhyloLikelihoodRescaledGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_ll_t = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& node_partials = ctx->input(3);
    const Tensor& node_scales = ctx->input(4);
    const Tensor& postorder_t = ctx->input(5);
    const Tensor& child_t = ctx->input(6);

    const int64_t B = grad_ll_t.dim_size(0);
    const int64_t Bt = probs.dim_size(0);
    const int64_t M = probs.dim_size(1);
    const int64_t S = probs.dim_size(2);
    const int64_t Nn = node_partials.dim_size(1);
    const int64_t Bf = freqs.dim_size(0);
    const int64_t I = postorder_t.dim_size(0);
    const int64_t C = child_t.dim_size(1);

    std::vector<int64_t> postorder, child;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);

    Tensor* grad_P_t = nullptr;
    Tensor* grad_F_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, probs.shape(), &grad_P_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, freqs.shape(), &grad_F_t));

    const T* grad_ll = grad_ll_t.flat<T>().data();
    const T* P = probs.flat<T>().data();
    const T* fr = freqs.flat<T>().data();
    const T* part = node_partials.flat<T>().data();
    const T* scales = node_scales.flat<T>().data();
    T* gP = grad_P_t->flat<T>().data();
    T* gF = grad_F_t->flat<T>().data();

    for (int64_t k = 0; k < Bt * M * S * S; ++k) gP[k] = T(0);
    for (int64_t k = 0; k < Bf * S; ++k) gF[k] = T(0);

    const int64_t root = postorder[I - 1];
    const bool reduceP = (Bt == 1) && (B > 1);
    const bool reduceF = (Bf == 1) && (B > 1);

    mutex mu;

    auto work = [&](int64_t begin, int64_t end) {
      std::vector<T> bar(Nn * S);
      std::vector<T> gmat(C * S);
      std::vector<T> sib(C * S);
      std::vector<T> w(S);
      std::vector<T> localP(reduceP ? M * S * S : 0, T(0));
      std::vector<T> localF(reduceF ? S : 0, T(0));

      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = (Bt == 1) ? 0 : b;
        const int64_t bf = (Bf == 1) ? 0 : b;
        const T* P_b = P + bt * M * S * S;
        const T* fr_b = fr + bf * S;
        const T* part_b = part + b * Nn * S;
        const T* scales_b = scales + b * Nn;
        const T dll = grad_ll[b];

        T* gP_b = reduceP ? localP.data() : (gP + bt * M * S * S);
        T* gF_b = reduceF ? localF.data() : (gF + bf * S);

        std::fill(bar.begin(), bar.end(), T(0));

        // logL = log(Lhat) + sum log scale. Seed with d logL / d phat_root.
        const T* pr = part_b + root * S;
        T Lhat = T(0);
        for (int64_t s = 0; s < S; ++s) Lhat += fr_b[s] * pr[s];
        const T invLhat = T(1) / Lhat;
        for (int64_t s = 0; s < S; ++s) {
          bar[root * S + s] = dll * fr_b[s] * invLhat;
          gF_b[s] += dll * pr[s] * invLhat;
        }

        for (int64_t i = I - 1; i >= 0; --i) {
          const int64_t v = postorder[i];
          const T* barv = &bar[v * S];
          const T inv_cv = T(1) / scales_b[v];

          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) {
              for (int64_t s = 0; s < S; ++s) gmat[ci * S + s] = T(1);
              continue;
            }
            const T* Pc = P_b + cnode * S * S;
            const T* pc = part_b + cnode * S;
            for (int64_t s = 0; s < S; ++s) {
              T g = T(0);
              const T* Prow = Pc + s * S;
              for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j];
              gmat[ci * S + s] = g;
            }
          }

          for (int64_t s = 0; s < S; ++s) {
            T pre = T(1);
            for (int64_t ci = 0; ci < C; ++ci) {
              sib[ci * S + s] = pre;
              pre *= gmat[ci * S + s];
            }
            T suf = T(1);
            for (int64_t ci = C - 1; ci >= 0; --ci) {
              sib[ci * S + s] *= suf;
              suf *= gmat[ci * S + s];
            }
          }

          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;
            // The 1/c_v factor from phat_v = raw_v / c_v.
            for (int64_t s = 0; s < S; ++s)
              w[s] = barv[s] * sib[ci * S + s] * inv_cv;
            const T* Pc = P_b + cnode * S * S;
            const T* pc = part_b + cnode * S;
            T* gPc = gP_b + cnode * S * S;
            for (int64_t s = 0; s < S; ++s) {
              const T ws = w[s];
              T* gRow = gPc + s * S;
              for (int64_t j = 0; j < S; ++j) gRow[j] += ws * pc[j];
            }
            T* barc = &bar[cnode * S];
            for (int64_t j = 0; j < S; ++j) {
              T acc = T(0);
              for (int64_t s = 0; s < S; ++s) acc += w[s] * Pc[s * S + j];
              barc[j] += acc;
            }
          }
        }
      }

      if (reduceP || reduceF) {
        mutex_lock l(mu);
        if (reduceP)
          for (int64_t k = 0; k < M * S * S; ++k) gP[k] += localP[k];
        if (reduceF)
          for (int64_t k = 0; k < S; ++k) gF[k] += localF[k];
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, 2 * M * S * S, work);
  }
};

#define REGISTER_CPU(T, Tindex)                                          \
  REGISTER_KERNEL_BUILDER(Name("PhyloLikelihood")                        \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindex>("Tindex"),         \
                          PhyloLikelihoodOp<T, Tindex>);                 \
  REGISTER_KERNEL_BUILDER(Name("PhyloLikelihoodGrad")                    \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindex>("Tindex"),         \
                          PhyloLikelihoodGradOp<T, Tindex>);             \
  REGISTER_KERNEL_BUILDER(Name("PhyloLikelihoodRescaled")                \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindex>("Tindex"),         \
                          PhyloLikelihoodRescaledOp<T, Tindex>);         \
  REGISTER_KERNEL_BUILDER(Name("PhyloLikelihoodRescaledGrad")            \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindex>("Tindex"),         \
                          PhyloLikelihoodRescaledGradOp<T, Tindex>);

REGISTER_CPU(float, int32)
REGISTER_CPU(float, int64_t)
REGISTER_CPU(double, int32)
REGISTER_CPU(double, int64_t)

#undef REGISTER_CPU
