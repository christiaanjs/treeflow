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
// Performance: the per-node transition-matrix products are the hot loops.  By
// default (block_size = 1) sites are processed one at a time, writing partials
// directly into the output buffer -- the original, and on the hardware tested
// the fastest, per-site traversal.  Setting the `block_size` op attribute > 1
// instead processes that many sites together: each block is transposed into a
// contiguous, site-innermost scratch layout ([node, state, block]) so the inner
// products become AXPY loops over the (contiguous) site dimension that the
// compiler can vectorise with SIMD, and so each transition matrix is loaded
// once per node per block rather than once per node per site.  This is opt-in
// because it was performance-neutral-to-negative on a 4-core / nucleotide
// (S=4) benchmark (the kernel is memory-movement bound there), but may help on
// other hardware or larger state counts.  The forward result is bit-identical
// for any block_size (the per-site summation order is preserved).
//
// Layout conventions (row-major):
//   sequences          [B,  L, S]      leaf partials (one-hot or ambiguity)
//   transition_probs   [Bt, M, S, S]   per-node transition matrices P[s, j]
//   frequencies        [Bf, S]         root state frequencies
//   postorder_indices  [I]             internal node ids in postorder
//   child_indices      [I, C]          child node ids per internal node
//   probs_index        [B]             probs set used by each batch element
//   freqs_index        [B]             freqs row used by each batch element
// where
//   B  = batch size (e.g. alignment sites)
//   L  = leaf count, M = 2L-1 total nodes, I = L-1 internal nodes
//   S  = state count, C = max children per internal node
//   Bt, Bf are the number of (distinct) transition-matrix / frequency sets.
//
// Rather than tiling the transition matrices / frequencies up to the full batch
// B, the caller keeps them at their own size Bt / Bf and supplies a gather index
// per batch element: element b reads transition set probs_index[b] and frequency
// row freqs_index[b]. This handles any broadcast pattern (Bt == 1 broadcast,
// Bt == B per-element, or e.g. Bt == M shared across sites for a discrete
// rate-category mixture) without materialising redundant copies. The backward op
// accumulates the gradient back into the Bt / Bf sets (scattering by the same
// index, reducing across the elements that share a set).
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

#include "tree_traversal.h"

using namespace tensorflow;
using treeflow::ReadIndices;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("PhyloLikelihood")
    .Attr("T: {float, double}")
    .Attr("Tindex: {int32, int64} = DT_INT32")
    .Attr("block_size: int = 1")
    .Input("sequences: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Input("probs_index: Tindex")
    .Input("freqs_index: Tindex")
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
    .Attr("block_size: int = 1")
    .Input("grad_site_likelihood: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("node_partials: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Input("probs_index: Tindex")
    .Input("freqs_index: Tindex")
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
    .Attr("block_size: int = 1")
    .Input("sequences: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Input("probs_index: Tindex")
    .Input("freqs_index: Tindex")
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
    .Attr("block_size: int = 1")
    .Input("grad_site_log_likelihood: T")
    .Input("transition_probs: T")
    .Input("frequencies: T")
    .Input("node_partials: T")
    .Input("node_scales: T")
    .Input("postorder_indices: Tindex")
    .Input("child_indices: Tindex")
    .Input("probs_index: Tindex")
    .Input("freqs_index: Tindex")
    .Output("grad_transition_probs: T")
    .Output("grad_frequencies: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));  // same shape as transition_probs
      c->set_output(1, c->input(2));  // same shape as frequencies
      return OkStatus();
    });

// ReadIndices lives in tree_traversal.h and is shared with the node-height
// ratio op.

template <typename T, typename Tindex>
class PhyloLikelihoodOp : public OpKernel {
 public:
  explicit PhyloLikelihoodOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    if (block_size_ < 1) block_size_ = 1;
  }

  void Compute(OpKernelContext* ctx) override {
    const int64_t kBlock = block_size_;
    const Tensor& sequences = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& postorder_t = ctx->input(3);
    const Tensor& child_t = ctx->input(4);
    const Tensor& probs_index_t = ctx->input(5);
    const Tensor& freqs_index_t = ctx->input(6);

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
    OP_REQUIRES(ctx, probs_index_t.NumElements() == B,
                errors::InvalidArgument("probs_index must have B elements"));
    OP_REQUIRES(ctx, freqs_index_t.NumElements() == B,
                errors::InvalidArgument("freqs_index must have B elements"));
    OP_REQUIRES(ctx, I == L - 1,
                errors::InvalidArgument("postorder length must be L-1"));
    OP_REQUIRES(ctx, M >= Nn - 1,
                errors::InvalidArgument(
                    "transition_probs must have at least 2L-2 nodes"));

    std::vector<int64_t> postorder, child, pidx, fidx;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);
    ReadIndices<Tindex>(probs_index_t, &pidx);
    ReadIndices<Tindex>(freqs_index_t, &fidx);

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
      // Partials for the current site block in [node, state, block] layout.
      std::vector<T> sc(Nn * S * kBlock);
      std::vector<T> grow(kBlock);  // scratch for one product row g[block]

      for (int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const int64_t bw = std::min<int64_t>(kBlock, end - b0);

        // Transpose this block's leaf partials into the scratch layout.
        for (int64_t leaf = 0; leaf < L; ++leaf) {
          for (int64_t s = 0; s < S; ++s) {
            T* __restrict__ dst = &sc[(leaf * S + s) * kBlock];
            const T* col = seq + b0 * L * S + leaf * S + s;
            for (int64_t w = 0; w < bw; ++w) dst[w] = col[w * L * S];
          }
        }

        // Internal nodes in postorder (children always precomputed).
        for (int64_t i = 0; i < I; ++i) {
          const int64_t v = postorder[i];
          T* pv = &sc[v * S * kBlock];
          for (int64_t k = 0; k < S * kBlock; ++k) pv[k] = T(1);
          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;  // padded child slot
            const T* pc = &sc[cnode * S * kBlock];
            if (Bt == 1) {
              const T* Pc = P + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                T* __restrict__ g = grow.data();
                for (int64_t w = 0; w < bw; ++w) g[w] = T(0);
                const T* Prow = Pc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T p = Prow[j];
                  const T* __restrict__ pcj = pc + j * kBlock;
                  for (int64_t w = 0; w < bw; ++w) g[w] += p * pcj[w];
                }
                T* __restrict__ pvs = pv + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) pvs[w] *= g[w];
              }
            } else {  // gather a per-element transition matrix set via pidx.
              for (int64_t s = 0; s < S; ++s) {
                T* pvs = pv + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) {
                  const T* Prow =
                      P + pidx[b0 + w] * M * S * S + cnode * S * S + s * S;
                  T g = T(0);
                  for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j * kBlock + w];
                  pvs[w] *= g;
                }
              }
            }
          }
        }

        // Root likelihood per site in the block.
        for (int64_t w = 0; w < bw; ++w) {
          const int64_t b = b0 + w;
          const int64_t bf = fidx[b];
          const T* fr_b = fr + bf * S;
          T acc = T(0);
          for (int64_t s = 0; s < S; ++s)
            acc += fr_b[s] * sc[(root * S + s) * kBlock + w];
          ll[b] = acc;
        }

        // Stream the block's partials back to the [B, Nn, S] buffer (one
        // contiguous per-site row at a time).
        for (int64_t w = 0; w < bw; ++w) {
          T* __restrict__ dst = part + (b0 + w) * Nn * S;
          for (int64_t vs = 0; vs < Nn * S; ++vs) dst[vs] = sc[vs * kBlock + w];
        }
      }
    };

    // Per-site path (default): write partials directly into the output buffer.
    auto work_scalar = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = pidx[b];
        const int64_t bf = fidx[b];
        const T* seq_b = seq + b * L * S;
        const T* P_b = P + bt * M * S * S;
        const T* fr_b = fr + bf * S;
        T* part_b = part + b * Nn * S;

        for (int64_t k = 0; k < L * S; ++k) part_b[k] = seq_b[k];

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
        }

        const T* pr = part_b + root * S;
        T acc = T(0);
        for (int64_t s = 0; s < S; ++s) acc += fr_b[s] * pr[s];
        ll[b] = acc;
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    const int64_t cost = M * S * S;  // rough per-batch cost
    if (kBlock <= 1)
      Shard(workers->num_threads, workers->workers, B, cost, work_scalar);
    else
      Shard(workers->num_threads, workers->workers, B, cost, work);
  }

 private:
  int block_size_ = 1;
};

template <typename T, typename Tindex>
class PhyloLikelihoodGradOp : public OpKernel {
 public:
  explicit PhyloLikelihoodGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    if (block_size_ < 1) block_size_ = 1;
  }

  void Compute(OpKernelContext* ctx) override {
    const int64_t kBlock = block_size_;
    const Tensor& grad_ll_t = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& node_partials = ctx->input(3);
    const Tensor& postorder_t = ctx->input(4);
    const Tensor& child_t = ctx->input(5);
    const Tensor& probs_index_t = ctx->input(6);
    const Tensor& freqs_index_t = ctx->input(7);

    const int64_t B = grad_ll_t.dim_size(0);
    const int64_t Bt = probs.dim_size(0);
    const int64_t M = probs.dim_size(1);  // transition_probs node count
    const int64_t S = probs.dim_size(2);
    const int64_t Nn = node_partials.dim_size(1);  // total node count (2L-1)
    const int64_t Bf = freqs.dim_size(0);
    const int64_t I = postorder_t.dim_size(0);
    const int64_t C = child_t.dim_size(1);

    std::vector<int64_t> postorder, child, pidx, fidx;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);
    ReadIndices<Tindex>(probs_index_t, &pidx);
    ReadIndices<Tindex>(freqs_index_t, &fidx);

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
    // When the transition probs / frequencies are shared across more than one
    // batch element (Bt < B / Bf < B, e.g. broadcast or one set per rate
    // category), several elements scatter into the same gradient slot, so each
    // thread accumulates into a private copy and reduces under a lock. When
    // every batch element has its own set (Bt == B) the gather indices are a
    // bijection, so threads write disjoint slots directly.
    const bool reduceP = (Bt < B);
    const bool reduceF = (Bf < B);

    mutex mu;  // guards reductions into the shared gP / gF

    auto work = [&](int64_t begin, int64_t end) {
      // Per-shard scratch in [node/child, state, block] layout.
      std::vector<T> partsc(Nn * S * kBlock);
      std::vector<T> bar(Nn * S * kBlock);
      std::vector<T> gmat(C * S * kBlock);
      std::vector<T> sib(C * S * kBlock);
      std::vector<T> wrow(S * kBlock);
      // Thread-local accumulators only when reducing shared batch slots.
      std::vector<T> localP(reduceP ? Bt * M * S * S : 0, T(0));
      std::vector<T> localF(reduceF ? Bf * S : 0, T(0));

      for (int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const int64_t bw = std::min<int64_t>(kBlock, end - b0);

        // Transpose the block's saved partials into scratch.
        for (int64_t w = 0; w < bw; ++w) {
          const T* __restrict__ src = part + (b0 + w) * Nn * S;
          for (int64_t vs = 0; vs < Nn * S; ++vs) partsc[vs * kBlock + w] = src[vs];
        }

        std::fill(bar.begin(), bar.end(), T(0));

        // Seed: L = sum_s fr[s] * part[root, s].
        for (int64_t w = 0; w < bw; ++w) {
          const int64_t b = b0 + w;
          const int64_t bf = fidx[b];
          const T* fr_b = fr + bf * S;
          const T dll = grad_ll[b];
          T* gF_b = (reduceF ? localF.data() : gF) + bf * S;
          for (int64_t s = 0; s < S; ++s) {
            const T prs = partsc[(root * S + s) * kBlock + w];
            bar[(root * S + s) * kBlock + w] = dll * fr_b[s];
            gF_b[s] += dll * prs;
          }
        }

        // Reverse postorder (parents before children).
        for (int64_t i = I - 1; i >= 0; --i) {
          const int64_t v = postorder[i];
          const T* barv = &bar[v * S * kBlock];

          // g_c[s, w] = sum_j P_c[s,j] * part_c[j, w]; identity for padded.
          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            T* gm = &gmat[ci * S * kBlock];
            if (cnode < 0) {
              for (int64_t k = 0; k < S * kBlock; ++k) gm[k] = T(1);
              continue;
            }
            const T* pc = &partsc[cnode * S * kBlock];
            if (Bt == 1) {
              const T* Pc = P + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                T* __restrict__ gms = gm + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) gms[w] = T(0);
                const T* Prow = Pc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T p = Prow[j];
                  const T* __restrict__ pcj = pc + j * kBlock;
                  for (int64_t w = 0; w < bw; ++w) gms[w] += p * pcj[w];
                }
              }
            } else {
              for (int64_t s = 0; s < S; ++s) {
                T* gms = gm + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) {
                  const T* Prow =
                      P + pidx[b0 + w] * M * S * S + cnode * S * S + s * S;
                  T g = T(0);
                  for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j * kBlock + w];
                  gms[w] = g;
                }
              }
            }
          }

          // Exclusive products across siblings via prefix/suffix (no division,
          // robust to zeros).
          for (int64_t s = 0; s < S; ++s) {
            for (int64_t w = 0; w < bw; ++w) {
              T pre = T(1);
              for (int64_t ci = 0; ci < C; ++ci) {
                sib[(ci * S + s) * kBlock + w] = pre;
                pre *= gmat[(ci * S + s) * kBlock + w];
              }
              T suf = T(1);
              for (int64_t ci = C - 1; ci >= 0; --ci) {
                sib[(ci * S + s) * kBlock + w] *= suf;
                suf *= gmat[(ci * S + s) * kBlock + w];
              }
            }
          }

          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;
            // w[s, .] = barv[s, .] * sib[ci, s, .]
            for (int64_t s = 0; s < S; ++s) {
              const T* __restrict__ barvs = barv + s * kBlock;
              const T* __restrict__ sibs = &sib[(ci * S + s) * kBlock];
              T* __restrict__ ws = &wrow[s * kBlock];
              for (int64_t w = 0; w < bw; ++w) ws[w] = barvs[w] * sibs[w];
            }
            const T* pc = &partsc[cnode * S * kBlock];

            // dL/dP_c[s,j] = sum_w w[s, w] * part_c[j, w]  (or per-site).
            if (Bt == 1) {
              T* gPc = (reduceP ? localP.data() : gP) + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                const T* ws = &wrow[s * kBlock];
                T* gRow = gPc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T* pcj = pc + j * kBlock;
                  T acc = T(0);
                  for (int64_t w = 0; w < bw; ++w) acc += ws[w] * pcj[w];
                  gRow[j] += acc;
                }
              }
            } else {
              for (int64_t w = 0; w < bw; ++w) {
                T* gPc = (reduceP ? localP.data() : gP) +
                         pidx[b0 + w] * M * S * S + cnode * S * S;
                for (int64_t s = 0; s < S; ++s) {
                  const T wsw = wrow[s * kBlock + w];
                  T* gRow = gPc + s * S;
                  for (int64_t j = 0; j < S; ++j) gRow[j] += wsw * pc[j * kBlock + w];
                }
              }
            }

            // Propagate adjoint to child partials: barc[j] += sum_s w[s]*P[s,j].
            T* barc = &bar[cnode * S * kBlock];
            if (Bt == 1) {
              const T* Pc = P + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                const T* __restrict__ ws = &wrow[s * kBlock];
                const T* Prow = Pc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T pj = Prow[j];
                  T* __restrict__ bcj = barc + j * kBlock;
                  for (int64_t w = 0; w < bw; ++w) bcj[w] += ws[w] * pj;
                }
              }
            } else {
              for (int64_t w = 0; w < bw; ++w) {
                const T* Pc = P + pidx[b0 + w] * M * S * S + cnode * S * S;
                for (int64_t j = 0; j < S; ++j) {
                  T acc = T(0);
                  for (int64_t s = 0; s < S; ++s) acc += wrow[s * kBlock + w] * Pc[s * S + j];
                  barc[j * kBlock + w] += acc;
                }
              }
            }
          }
        }
      }

      if (reduceP || reduceF) {
        mutex_lock l(mu);
        if (reduceP)
          for (int64_t k = 0; k < Bt * M * S * S; ++k) gP[k] += localP[k];
        if (reduceF)
          for (int64_t k = 0; k < Bf * S; ++k) gF[k] += localF[k];
      }
    };

    // Per-site path (default).
    auto work_scalar = [&](int64_t begin, int64_t end) {
      std::vector<T> bar(Nn * S);
      std::vector<T> gmat(C * S);
      std::vector<T> sib(C * S);
      std::vector<T> w(S);
      std::vector<T> localP(reduceP ? Bt * M * S * S : 0, T(0));
      std::vector<T> localF(reduceF ? Bf * S : 0, T(0));

      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = pidx[b];
        const int64_t bf = fidx[b];
        const T* P_b = P + bt * M * S * S;
        const T* fr_b = fr + bf * S;
        const T* part_b = part + b * Nn * S;
        const T dll = grad_ll[b];

        T* gP_b = (reduceP ? localP.data() : gP) + bt * M * S * S;
        T* gF_b = (reduceF ? localF.data() : gF) + bf * S;

        std::fill(bar.begin(), bar.end(), T(0));

        const T* pr = part_b + root * S;
        for (int64_t s = 0; s < S; ++s) {
          bar[root * S + s] = dll * fr_b[s];
          gF_b[s] += dll * pr[s];
        }

        for (int64_t i = I - 1; i >= 0; --i) {
          const int64_t v = postorder[i];
          const T* barv = &bar[v * S];

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
            for (int64_t s = 0; s < S; ++s) w[s] = barv[s] * sib[ci * S + s];
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
          for (int64_t k = 0; k < Bt * M * S * S; ++k) gP[k] += localP[k];
        if (reduceF)
          for (int64_t k = 0; k < Bf * S; ++k) gF[k] += localF[k];
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    const int64_t cost = 2 * M * S * S;
    if (kBlock <= 1)
      Shard(workers->num_threads, workers->workers, B, cost, work_scalar);
    else
      Shard(workers->num_threads, workers->workers, B, cost, work);
  }

 private:
  int block_size_ = 1;
};

// ===========================================================================
// Rescaled (numerically stable) variants
// ===========================================================================

template <typename T, typename Tindex>
class PhyloLikelihoodRescaledOp : public OpKernel {
 public:
  explicit PhyloLikelihoodRescaledOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    if (block_size_ < 1) block_size_ = 1;
  }

  void Compute(OpKernelContext* ctx) override {
    const int64_t kBlock = block_size_;
    const Tensor& sequences = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& postorder_t = ctx->input(3);
    const Tensor& child_t = ctx->input(4);
    const Tensor& probs_index_t = ctx->input(5);
    const Tensor& freqs_index_t = ctx->input(6);

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

    OP_REQUIRES(ctx, probs_index_t.NumElements() == B,
                errors::InvalidArgument("probs_index must have B elements"));
    OP_REQUIRES(ctx, freqs_index_t.NumElements() == B,
                errors::InvalidArgument("freqs_index must have B elements"));
    OP_REQUIRES(ctx, I == L - 1,
                errors::InvalidArgument("postorder length must be L-1"));

    std::vector<int64_t> postorder, child, pidx, fidx;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);
    ReadIndices<Tindex>(probs_index_t, &pidx);
    ReadIndices<Tindex>(freqs_index_t, &fidx);

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
      std::vector<T> sc(Nn * S * kBlock);       // partials [node, state, block]
      std::vector<T> scsc(Nn * kBlock);         // scale factors [node, block]
      std::vector<T> logsum(kBlock);            // running log-scale sum
      std::vector<T> grow(kBlock);

      for (int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const int64_t bw = std::min<int64_t>(kBlock, end - b0);

        for (int64_t leaf = 0; leaf < L; ++leaf) {
          for (int64_t s = 0; s < S; ++s) {
            T* __restrict__ dst = &sc[(leaf * S + s) * kBlock];
            const T* col = seq + b0 * L * S + leaf * S + s;
            for (int64_t w = 0; w < bw; ++w) dst[w] = col[w * L * S];
          }
        }
        for (int64_t k = 0; k < Nn * kBlock; ++k) scsc[k] = T(1);
        for (int64_t w = 0; w < bw; ++w) logsum[w] = T(0);

        for (int64_t i = 0; i < I; ++i) {
          const int64_t v = postorder[i];
          T* pv = &sc[v * S * kBlock];
          for (int64_t k = 0; k < S * kBlock; ++k) pv[k] = T(1);
          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;
            const T* pc = &sc[cnode * S * kBlock];
            if (Bt == 1) {
              const T* Pc = P + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                T* __restrict__ g = grow.data();
                for (int64_t w = 0; w < bw; ++w) g[w] = T(0);
                const T* Prow = Pc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T p = Prow[j];
                  const T* __restrict__ pcj = pc + j * kBlock;
                  for (int64_t w = 0; w < bw; ++w) g[w] += p * pcj[w];
                }
                T* __restrict__ pvs = pv + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) pvs[w] *= g[w];
              }
            } else {
              for (int64_t s = 0; s < S; ++s) {
                T* pvs = pv + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) {
                  const T* Prow =
                      P + pidx[b0 + w] * M * S * S + cnode * S * S + s * S;
                  T g = T(0);
                  for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j * kBlock + w];
                  pvs[w] *= g;
                }
              }
            }
          }
          // Rescale by the per-site maximum partial.
          T* scv = &scsc[v * kBlock];
          for (int64_t w = 0; w < bw; ++w) {
            T cmax = T(0);
            for (int64_t s = 0; s < S; ++s)
              cmax = std::max(cmax, pv[s * kBlock + w]);
            if (cmax > T(0)) {
              const T inv = T(1) / cmax;
              for (int64_t s = 0; s < S; ++s) pv[s * kBlock + w] *= inv;
              scv[w] = cmax;
              logsum[w] += std::log(cmax);
            }
          }
        }

        for (int64_t w = 0; w < bw; ++w) {
          const int64_t b = b0 + w;
          const int64_t bf = fidx[b];
          const T* fr_b = fr + bf * S;
          T acc = T(0);
          for (int64_t s = 0; s < S; ++s)
            acc += fr_b[s] * sc[(root * S + s) * kBlock + w];
          ll[b] = std::log(acc) + logsum[w];
        }

        // Stream partials and scales back to the [B, Nn, S] / [B, Nn] buffers.
        for (int64_t w = 0; w < bw; ++w) {
          T* __restrict__ dst = part + (b0 + w) * Nn * S;
          for (int64_t vs = 0; vs < Nn * S; ++vs) dst[vs] = sc[vs * kBlock + w];
          T* __restrict__ dsc = scales + (b0 + w) * Nn;
          for (int64_t v = 0; v < Nn; ++v) dsc[v] = scsc[v * kBlock + w];
        }
      }
    };

    // Per-site path (default).
    auto work_scalar = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = pidx[b];
        const int64_t bf = fidx[b];
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
    if (kBlock <= 1)
      Shard(workers->num_threads, workers->workers, B, M * S * S, work_scalar);
    else
      Shard(workers->num_threads, workers->workers, B, M * S * S, work);
  }

 private:
  int block_size_ = 1;
};

template <typename T, typename Tindex>
class PhyloLikelihoodRescaledGradOp : public OpKernel {
 public:
  explicit PhyloLikelihoodRescaledGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    if (block_size_ < 1) block_size_ = 1;
  }

  void Compute(OpKernelContext* ctx) override {
    const int64_t kBlock = block_size_;
    const Tensor& grad_ll_t = ctx->input(0);
    const Tensor& probs = ctx->input(1);
    const Tensor& freqs = ctx->input(2);
    const Tensor& node_partials = ctx->input(3);
    const Tensor& node_scales = ctx->input(4);
    const Tensor& postorder_t = ctx->input(5);
    const Tensor& child_t = ctx->input(6);
    const Tensor& probs_index_t = ctx->input(7);
    const Tensor& freqs_index_t = ctx->input(8);

    const int64_t B = grad_ll_t.dim_size(0);
    const int64_t Bt = probs.dim_size(0);
    const int64_t M = probs.dim_size(1);
    const int64_t S = probs.dim_size(2);
    const int64_t Nn = node_partials.dim_size(1);
    const int64_t Bf = freqs.dim_size(0);
    const int64_t I = postorder_t.dim_size(0);
    const int64_t C = child_t.dim_size(1);

    std::vector<int64_t> postorder, child, pidx, fidx;
    ReadIndices<Tindex>(postorder_t, &postorder);
    ReadIndices<Tindex>(child_t, &child);
    ReadIndices<Tindex>(probs_index_t, &pidx);
    ReadIndices<Tindex>(freqs_index_t, &fidx);

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
    // See PhyloLikelihoodGradOp: reduce when a probs/freqs set is shared by more
    // than one batch element (Bt < B / Bf < B).
    const bool reduceP = (Bt < B);
    const bool reduceF = (Bf < B);

    mutex mu;

    auto work = [&](int64_t begin, int64_t end) {
      std::vector<T> partsc(Nn * S * kBlock);
      std::vector<T> scalesc(Nn * kBlock);
      std::vector<T> bar(Nn * S * kBlock);
      std::vector<T> gmat(C * S * kBlock);
      std::vector<T> sib(C * S * kBlock);
      std::vector<T> wrow(S * kBlock);
      std::vector<T> localP(reduceP ? Bt * M * S * S : 0, T(0));
      std::vector<T> localF(reduceF ? Bf * S : 0, T(0));

      for (int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const int64_t bw = std::min<int64_t>(kBlock, end - b0);

        for (int64_t w = 0; w < bw; ++w) {
          const T* __restrict__ src = part + (b0 + w) * Nn * S;
          for (int64_t vs = 0; vs < Nn * S; ++vs) partsc[vs * kBlock + w] = src[vs];
          const T* __restrict__ ssrc = scales + (b0 + w) * Nn;
          for (int64_t v = 0; v < Nn; ++v) scalesc[v * kBlock + w] = ssrc[v];
        }

        std::fill(bar.begin(), bar.end(), T(0));

        // logL = log(Lhat) + sum log scale. Seed with d logL / d phat_root.
        for (int64_t w = 0; w < bw; ++w) {
          const int64_t b = b0 + w;
          const int64_t bf = fidx[b];
          const T* fr_b = fr + bf * S;
          const T dll = grad_ll[b];
          T Lhat = T(0);
          for (int64_t s = 0; s < S; ++s)
            Lhat += fr_b[s] * partsc[(root * S + s) * kBlock + w];
          const T invLhat = T(1) / Lhat;
          T* gF_b = (reduceF ? localF.data() : gF) + bf * S;
          for (int64_t s = 0; s < S; ++s) {
            const T prs = partsc[(root * S + s) * kBlock + w];
            bar[(root * S + s) * kBlock + w] = dll * fr_b[s] * invLhat;
            gF_b[s] += dll * prs * invLhat;
          }
        }

        for (int64_t i = I - 1; i >= 0; --i) {
          const int64_t v = postorder[i];
          const T* barv = &bar[v * S * kBlock];
          const T* scv = &scalesc[v * kBlock];  // scale c_v (divide adjoint by it)

          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            T* gm = &gmat[ci * S * kBlock];
            if (cnode < 0) {
              for (int64_t k = 0; k < S * kBlock; ++k) gm[k] = T(1);
              continue;
            }
            const T* pc = &partsc[cnode * S * kBlock];
            if (Bt == 1) {
              const T* Pc = P + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                T* __restrict__ gms = gm + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) gms[w] = T(0);
                const T* Prow = Pc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T p = Prow[j];
                  const T* __restrict__ pcj = pc + j * kBlock;
                  for (int64_t w = 0; w < bw; ++w) gms[w] += p * pcj[w];
                }
              }
            } else {
              for (int64_t s = 0; s < S; ++s) {
                T* gms = gm + s * kBlock;
                for (int64_t w = 0; w < bw; ++w) {
                  const T* Prow =
                      P + pidx[b0 + w] * M * S * S + cnode * S * S + s * S;
                  T g = T(0);
                  for (int64_t j = 0; j < S; ++j) g += Prow[j] * pc[j * kBlock + w];
                  gms[w] = g;
                }
              }
            }
          }

          for (int64_t s = 0; s < S; ++s) {
            for (int64_t w = 0; w < bw; ++w) {
              T pre = T(1);
              for (int64_t ci = 0; ci < C; ++ci) {
                sib[(ci * S + s) * kBlock + w] = pre;
                pre *= gmat[(ci * S + s) * kBlock + w];
              }
              T suf = T(1);
              for (int64_t ci = C - 1; ci >= 0; --ci) {
                sib[(ci * S + s) * kBlock + w] *= suf;
                suf *= gmat[(ci * S + s) * kBlock + w];
              }
            }
          }

          for (int64_t ci = 0; ci < C; ++ci) {
            const int64_t cnode = child[i * C + ci];
            if (cnode < 0) continue;
            // The 1/c_v factor from phat_v = raw_v / c_v.
            for (int64_t s = 0; s < S; ++s) {
              const T* __restrict__ barvs = barv + s * kBlock;
              const T* __restrict__ sibs = &sib[(ci * S + s) * kBlock];
              T* __restrict__ ws = &wrow[s * kBlock];
              for (int64_t w = 0; w < bw; ++w)
                ws[w] = barvs[w] * sibs[w] / scv[w];
            }
            const T* pc = &partsc[cnode * S * kBlock];

            if (Bt == 1) {
              T* gPc = (reduceP ? localP.data() : gP) + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                const T* ws = &wrow[s * kBlock];
                T* gRow = gPc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T* pcj = pc + j * kBlock;
                  T acc = T(0);
                  for (int64_t w = 0; w < bw; ++w) acc += ws[w] * pcj[w];
                  gRow[j] += acc;
                }
              }
            } else {
              for (int64_t w = 0; w < bw; ++w) {
                T* gPc = (reduceP ? localP.data() : gP) +
                         pidx[b0 + w] * M * S * S + cnode * S * S;
                for (int64_t s = 0; s < S; ++s) {
                  const T wsw = wrow[s * kBlock + w];
                  T* gRow = gPc + s * S;
                  for (int64_t j = 0; j < S; ++j) gRow[j] += wsw * pc[j * kBlock + w];
                }
              }
            }

            T* barc = &bar[cnode * S * kBlock];
            if (Bt == 1) {
              const T* Pc = P + cnode * S * S;
              for (int64_t s = 0; s < S; ++s) {
                const T* __restrict__ ws = &wrow[s * kBlock];
                const T* Prow = Pc + s * S;
                for (int64_t j = 0; j < S; ++j) {
                  const T pj = Prow[j];
                  T* __restrict__ bcj = barc + j * kBlock;
                  for (int64_t w = 0; w < bw; ++w) bcj[w] += ws[w] * pj;
                }
              }
            } else {
              for (int64_t w = 0; w < bw; ++w) {
                const T* Pc = P + pidx[b0 + w] * M * S * S + cnode * S * S;
                for (int64_t j = 0; j < S; ++j) {
                  T acc = T(0);
                  for (int64_t s = 0; s < S; ++s) acc += wrow[s * kBlock + w] * Pc[s * S + j];
                  barc[j * kBlock + w] += acc;
                }
              }
            }
          }
        }
      }

      if (reduceP || reduceF) {
        mutex_lock l(mu);
        if (reduceP)
          for (int64_t k = 0; k < Bt * M * S * S; ++k) gP[k] += localP[k];
        if (reduceF)
          for (int64_t k = 0; k < Bf * S; ++k) gF[k] += localF[k];
      }
    };

    // Per-site path (default).
    auto work_scalar = [&](int64_t begin, int64_t end) {
      std::vector<T> bar(Nn * S);
      std::vector<T> gmat(C * S);
      std::vector<T> sib(C * S);
      std::vector<T> w(S);
      std::vector<T> localP(reduceP ? Bt * M * S * S : 0, T(0));
      std::vector<T> localF(reduceF ? Bf * S : 0, T(0));

      for (int64_t b = begin; b < end; ++b) {
        const int64_t bt = pidx[b];
        const int64_t bf = fidx[b];
        const T* P_b = P + bt * M * S * S;
        const T* fr_b = fr + bf * S;
        const T* part_b = part + b * Nn * S;
        const T* scales_b = scales + b * Nn;
        const T dll = grad_ll[b];

        T* gP_b = (reduceP ? localP.data() : gP) + bt * M * S * S;
        T* gF_b = (reduceF ? localF.data() : gF) + bf * S;

        std::fill(bar.begin(), bar.end(), T(0));

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
          for (int64_t k = 0; k < Bt * M * S * S; ++k) gP[k] += localP[k];
        if (reduceF)
          for (int64_t k = 0; k < Bf * S; ++k) gF[k] += localF[k];
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    if (kBlock <= 1)
      Shard(workers->num_threads, workers->workers, B, 2 * M * S * S,
            work_scalar);
    else
      Shard(workers->num_threads, workers->workers, B, 2 * M * S * S, work);
  }

 private:
  int block_size_ = 1;
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
