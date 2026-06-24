// Native TensorFlow custom ops for conditional clade (subsplit Bayesian network)
// topology distributions.
//
// These are the compiled counterparts of the pure-TensorFlow ``tf.while_loop``
// implementations in ``treeflow/conditional_clade/tensor_ops.py``. The recursive,
// data-dependent structure of a topology -- which clades get expanded depends on
// which subsplits were sampled -- is awkward to express with ``tf.while_loop`` but
// natural as ordinary C++ recursion/loops, so the kernels here are both simpler
// and faster than the graph-mode reference.
//
// Clades are integer bitsets over the taxa (bit ``i`` set iff taxon ``i`` is a
// member). A tree with ``L`` leaves has ``M = 2L-1`` nodes and ``L-1`` internal
// nodes; the labelling convention is leaves ``0..L-1``, internal nodes ``L..2L-2``
// in post-order (every parent index larger than its children), root last.
//
// Ops:
//   ConditionalCladeSample          -- sample parent_indices from per-subsplit
//                                      logits (one independent topology per seed).
//   ConditionalCladeLogProb         -- log-probability of a topology, plus the
//                                      chosen flat subsplit indices (saved for the
//                                      gradient).
//   ConditionalCladeLogProbGrad     -- scatter-add gradient w.r.t. the conditional
//                                      log-probs.
//   ParentIndicesToChildIndices     -- derive child_indices from parent_indices.
//   ChildIndicesToPreorder          -- pre-order traversal from child_indices.
//
// Support layout (constants derived from a ConditionalCladeSupport, passed in as
// tensors):
//   clade_offset [2^L]  start offset of each clade's subsplit segment in the flat
//                       logit vector (indexed by clade bitset).
//   clade_count  [2^L]  number of subsplits of each clade.
//   flat_child1  [M_s]  child1 bitset of each flat subsplit (canonical: contains
//                       the parent's smallest taxon).
//   flat_child2  [M_s]  child2 bitset of each flat subsplit.
//   flat_parent  [M_s]  parent clade bitset of each flat subsplit.
// where M_s is the total number of subsplits (the length of the logit vector).

#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

inline int Popcount(uint32_t clade) { return __builtin_popcount(clade); }
inline bool IsSingleton(uint32_t clade) {
  return clade != 0 && (clade & (clade - 1)) == 0;
}
inline int LowestTaxon(uint32_t clade) { return __builtin_ctz(clade); }

}  // namespace

// ---------------------------------------------------------------------------
// Op registrations
// ---------------------------------------------------------------------------
REGISTER_OP("ConditionalCladeSample")
    .Attr("T: {float, double}")
    .Attr("taxon_count: int")
    .Input("logits: T")
    .Input("seeds: int32")
    .Input("clade_offset: int32")
    .Input("clade_count: int32")
    .Input("flat_child1: int32")
    .Input("flat_child2: int32")
    .Output("parent_indices: int32")
    .Output("flat_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      int taxon_count;
      TF_RETURN_IF_ERROR(c->GetAttr("taxon_count", &taxon_count));
      ShapeHandle seeds = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(seeds, 2, &seeds));
      c->set_output(0, c->MakeShape({c->Dim(seeds, 0), 2 * taxon_count - 2}));
      c->set_output(1, c->MakeShape({c->Dim(seeds, 0), taxon_count - 1}));
      return OkStatus();
    });

REGISTER_OP("ConditionalCladeLogProb")
    .Attr("T: {float, double}")
    .Attr("taxon_count: int")
    .Input("conditional_log_probs: T")
    .Input("parent_indices: int32")
    .Input("flat_parent: int32")
    .Input("flat_child1: int32")
    .Output("log_prob: T")
    .Output("flat_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      int taxon_count;
      TF_RETURN_IF_ERROR(c->GetAttr("taxon_count", &taxon_count));
      ShapeHandle parents = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(parents, 2, &parents));
      c->set_output(0, c->MakeShape({c->Dim(parents, 0)}));
      c->set_output(1, c->MakeShape({c->Dim(parents, 0), taxon_count - 1}));
      return OkStatus();
    });

REGISTER_OP("ConditionalCladeLogProbGrad")
    .Attr("T: {float, double}")
    .Input("grad_log_prob: T")
    .Input("flat_indices: int32")
    .Input("conditional_log_probs: T")  // value unused; provides output shape
    .Output("grad_conditional_log_probs: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      return OkStatus();
    });

REGISTER_OP("ParentIndicesToChildIndices")
    .Attr("taxon_count: int")
    .Input("parent_indices: int32")
    .Output("child_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      int taxon_count;
      TF_RETURN_IF_ERROR(c->GetAttr("taxon_count", &taxon_count));
      ShapeHandle parents = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(parents, 2, &parents));
      c->set_output(
          0, c->MakeShape({c->Dim(parents, 0), 2 * taxon_count - 1, 2}));
      return OkStatus();
    });

REGISTER_OP("ChildIndicesToPreorder")
    .Attr("taxon_count: int")
    .Input("child_indices: int32")
    .Output("preorder_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      int taxon_count;
      TF_RETURN_IF_ERROR(c->GetAttr("taxon_count", &taxon_count));
      ShapeHandle child = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(child, 3, &child));
      c->set_output(0, c->MakeShape({c->Dim(child, 0), 2 * taxon_count - 1}));
      return OkStatus();
    });

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
template <typename T>
class ConditionalCladeSampleOp : public OpKernel {
 public:
  explicit ConditionalCladeSampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("taxon_count", &taxon_count_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& logits_t = ctx->input(0);
    const Tensor& seeds_t = ctx->input(1);
    const Tensor& clade_offset_t = ctx->input(2);
    const Tensor& clade_count_t = ctx->input(3);
    const Tensor& flat_child1_t = ctx->input(4);
    const Tensor& flat_child2_t = ctx->input(5);

    OP_REQUIRES(ctx, seeds_t.dims() == 2 && seeds_t.dim_size(1) == 2,
                errors::InvalidArgument("seeds must be [B, 2]"));

    const int n = taxon_count_;
    const int node_count = 2 * n - 1;
    const int64_t B = seeds_t.dim_size(0);
    const uint32_t root = (1u << n) - 1;

    const T* logits = logits_t.flat<T>().data();
    const int32* clade_offset = clade_offset_t.flat<int32>().data();
    const int32* clade_count = clade_count_t.flat<int32>().data();
    const int32* flat_child1 = flat_child1_t.flat<int32>().data();
    const int32* flat_child2 = flat_child2_t.flat<int32>().data();
    const int32* seeds = seeds_t.flat<int32>().data();

    Tensor* parent_t = nullptr;
    Tensor* flat_t = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {B, node_count - 1}, &parent_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {B, n - 1}, &flat_t));
    int32* parent = parent_t->flat<int32>().data();
    int32* flat_indices = flat_t->flat<int32>().data();

    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        std::seed_seq seq{seeds[2 * b], seeds[2 * b + 1]};
        std::mt19937_64 rng(seq);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        int32* parent_b = parent + b * (node_count - 1);
        int32* flat_b = flat_indices + b * (n - 1);
        int next_internal = n;
        int decision = 0;  // counts internal nodes; flat order is irrelevant
                           // downstream (the estimators sum over the n-1 indices)

        // Recursively expand a clade, returning its assigned node id. Internal
        // ids are handed out on the way back up (post-order).
        std::function<int(uint32_t)> expand = [&](uint32_t clade) -> int {
          if (IsSingleton(clade)) return LowestTaxon(clade);
          const int offset = clade_offset[clade];
          const int count = clade_count[clade];
          // Categorical sample over the clade's subsplit segment.
          double max_logit = -std::numeric_limits<double>::infinity();
          for (int k = 0; k < count; ++k)
            max_logit = std::max<double>(max_logit, logits[offset + k]);
          double total = 0.0;
          for (int k = 0; k < count; ++k)
            total += std::exp(static_cast<double>(logits[offset + k]) - max_logit);
          double u = uniform(rng) * total;
          double acc = 0.0;
          int choice = count - 1;
          for (int k = 0; k < count; ++k) {
            acc += std::exp(static_cast<double>(logits[offset + k]) - max_logit);
            if (u <= acc) {
              choice = k;
              break;
            }
          }
          const int flat = offset + choice;
          flat_b[decision++] = flat;  // record the chosen flat subsplit index
          const int id1 = expand(static_cast<uint32_t>(flat_child1[flat]));
          const int id2 = expand(static_cast<uint32_t>(flat_child2[flat]));
          const int nid = next_internal++;
          parent_b[id1] = nid;
          parent_b[id2] = nid;
          return nid;
        };
        expand(root);
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, node_count, work);
  }

 private:
  int taxon_count_;
};

// ---------------------------------------------------------------------------
// Log-probability
// ---------------------------------------------------------------------------
template <typename T>
class ConditionalCladeLogProbOp : public OpKernel {
 public:
  explicit ConditionalCladeLogProbOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("taxon_count", &taxon_count_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& cond_t = ctx->input(0);
    const Tensor& parent_t = ctx->input(1);
    const Tensor& flat_parent_t = ctx->input(2);
    const Tensor& flat_child1_t = ctx->input(3);

    OP_REQUIRES(ctx, parent_t.dims() == 2,
                errors::InvalidArgument("parent_indices must be [B, 2n-2]"));

    const int n = taxon_count_;
    const int node_count = 2 * n - 1;
    const int64_t B = parent_t.dim_size(0);
    const int64_t M = flat_parent_t.NumElements();
    const int64_t pow2n = 1LL << n;

    const T* cond = cond_t.flat<T>().data();
    const int32* parent = parent_t.flat<int32>().data();
    const int32* flat_parent = flat_parent_t.flat<int32>().data();
    const int32* flat_child1 = flat_child1_t.flat<int32>().data();

    // (parent clade, canonical child1) -> flat subsplit index.
    std::unordered_map<int64_t, int32> table;
    table.reserve(M * 2);
    for (int64_t i = 0; i < M; ++i) {
      const int64_t key =
          static_cast<int64_t>(flat_parent[i]) * pow2n + flat_child1[i];
      table[key] = static_cast<int32>(i);
    }

    Tensor* log_prob_t = nullptr;
    Tensor* flat_indices_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {B}, &log_prob_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {B, n - 1}, &flat_indices_t));
    T* log_prob = log_prob_t->flat<T>().data();
    int32* flat_indices = flat_indices_t->flat<int32>().data();

    auto work = [&](int64_t begin, int64_t end) {
      std::vector<uint32_t> clade_of(node_count);
      std::vector<int> child0(node_count), child1(node_count);
      for (int64_t b = begin; b < end; ++b) {
        const int32* parent_b = parent + b * (node_count - 1);
        int32* flat_indices_b = flat_indices + b * (n - 1);

        for (int i = 0; i < n; ++i) clade_of[i] = 1u << i;
        for (int i = n; i < node_count; ++i) {
          child0[i] = -1;
          child1[i] = -1;
        }
        // Children of each node (ascending child id => column 0 < column 1).
        for (int child = 0; child < node_count - 1; ++child) {
          const int p = parent_b[child];
          if (child0[p] < 0)
            child0[p] = child;
          else
            child1[p] = child;
        }
        // Post-order clade union (children have smaller ids than parents).
        for (int i = n; i < node_count; ++i)
          clade_of[i] = clade_of[child0[i]] | clade_of[child1[i]];

        double total = 0.0;
        for (int i = n; i < node_count; ++i) {
          const uint32_t cl0 = clade_of[child0[i]];
          const uint32_t cl1 = clade_of[child1[i]];
          const uint32_t low0 = cl0 & (~cl0 + 1);
          const uint32_t low1 = cl1 & (~cl1 + 1);
          const uint32_t canonical_child1 = (low0 < low1) ? cl0 : cl1;
          const uint32_t parent_clade = cl0 | cl1;
          const int64_t key =
              static_cast<int64_t>(parent_clade) * pow2n + canonical_child1;
          const int32 flat = table.at(key);
          flat_indices_b[i - n] = flat;
          total += static_cast<double>(cond[flat]);
        }
        log_prob[b] = static_cast<T>(total);
      }
    };

    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, node_count, work);
  }

 private:
  int taxon_count_;
};

template <typename T>
class ConditionalCladeLogProbGradOp : public OpKernel {
 public:
  explicit ConditionalCladeLogProbGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_t = ctx->input(0);
    const Tensor& flat_indices_t = ctx->input(1);
    const Tensor& cond_t = ctx->input(2);

    const int64_t B = flat_indices_t.dim_size(0);
    const int64_t per = flat_indices_t.dim_size(1);
    const int64_t M = cond_t.NumElements();

    const T* grad = grad_t.flat<T>().data();
    const int32* flat_indices = flat_indices_t.flat<int32>().data();

    Tensor* out_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, cond_t.shape(), &out_t));
    T* out = out_t->flat<T>().data();
    for (int64_t i = 0; i < M; ++i) out[i] = T(0);

    // log_prob[b] = sum_k cond[flat_indices[b,k]], so the adjoint of cond is a
    // scatter-add of the per-sample upstream gradient onto the chosen indices.
    for (int64_t b = 0; b < B; ++b) {
      const T g = grad[b];
      const int32* row = flat_indices + b * per;
      for (int64_t k = 0; k < per; ++k) out[row[k]] += g;
    }
  }
};

// ---------------------------------------------------------------------------
// Topology index transforms
// ---------------------------------------------------------------------------
class ParentIndicesToChildIndicesOp : public OpKernel {
 public:
  explicit ParentIndicesToChildIndicesOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("taxon_count", &taxon_count_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& parent_t = ctx->input(0);
    OP_REQUIRES(ctx, parent_t.dims() == 2,
                errors::InvalidArgument("parent_indices must be [B, 2n-2]"));
    const int node_count = 2 * taxon_count_ - 1;
    const int64_t B = parent_t.dim_size(0);
    const int32* parent = parent_t.flat<int32>().data();

    Tensor* child_t = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {B, node_count, 2}, &child_t));
    int32* child = child_t->flat<int32>().data();

    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const int32* parent_b = parent + b * (node_count - 1);
        int32* child_b = child + b * node_count * 2;
        for (int i = 0; i < node_count * 2; ++i) child_b[i] = -1;
        for (int c = 0; c < node_count - 1; ++c) {
          const int p = parent_b[c];
          if (child_b[p * 2] < 0)
            child_b[p * 2] = c;
          else
            child_b[p * 2 + 1] = c;
        }
      }
    };
    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, node_count, work);
  }

 private:
  int taxon_count_;
};

class ChildIndicesToPreorderOp : public OpKernel {
 public:
  explicit ChildIndicesToPreorderOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("taxon_count", &taxon_count_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& child_t = ctx->input(0);
    OP_REQUIRES(ctx, child_t.dims() == 3,
                errors::InvalidArgument("child_indices must be [B, M, 2]"));
    const int node_count = 2 * taxon_count_ - 1;
    const int64_t B = child_t.dim_size(0);
    const int32* child = child_t.flat<int32>().data();

    Tensor* pre_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {B, node_count}, &pre_t));
    int32* pre = pre_t->flat<int32>().data();

    auto work = [&](int64_t begin, int64_t end) {
      std::vector<int> stack(node_count);
      for (int64_t b = begin; b < end; ++b) {
        const int32* child_b = child + b * node_count * 2;
        int32* pre_b = pre + b * node_count;
        int top = 0;
        stack[top++] = node_count - 1;  // root
        int counter = 0;
        while (top > 0) {
          const int node = stack[--top];
          pre_b[counter++] = node;
          const int c0 = child_b[node * 2];
          if (c0 >= 0) {
            stack[top++] = child_b[node * 2 + 1];  // c1 first (deeper)
            stack[top++] = c0;                     // c0 visited next
          }
        }
      }
    };
    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, B, node_count, work);
  }

 private:
  int taxon_count_;
};

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------
#define REGISTER_FLOAT_KERNELS(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ConditionalCladeSample").Device(DEVICE_CPU).TypeConstraint<T>(  \
          "T"),                                                             \
      ConditionalCladeSampleOp<T>);                                         \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ConditionalCladeLogProb").Device(DEVICE_CPU).TypeConstraint<T>( \
          "T"),                                                             \
      ConditionalCladeLogProbOp<T>);                                        \
  REGISTER_KERNEL_BUILDER(Name("ConditionalCladeLogProbGrad")               \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<T>("T"),                      \
                          ConditionalCladeLogProbGradOp<T>);

REGISTER_FLOAT_KERNELS(float)
REGISTER_FLOAT_KERNELS(double)
#undef REGISTER_FLOAT_KERNELS

REGISTER_KERNEL_BUILDER(
    Name("ParentIndicesToChildIndices").Device(DEVICE_CPU),
    ParentIndicesToChildIndicesOp);
REGISTER_KERNEL_BUILDER(Name("ChildIndicesToPreorder").Device(DEVICE_CPU),
                        ChildIndicesToPreorderOp);
