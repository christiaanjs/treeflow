// Native TensorFlow custom op: ancestral sampling of rooted tree topologies
// from a subsplit Bayesian network (SBN / conditional clade distribution).
//
// This is the compiled counterpart of the NumPy reference walk in
// ``treeflow.vbpi.sbn.SubsplitBayesianNetwork._sample_numpy``. Sampling a
// topology is an inherently *sequential* traversal of the SBN's pointer-array
// (CSR) support -- start at the root clade, draw one of its candidate child
// subsplits from the (pre-normalised) conditional probabilities, recurse into
// each non-leaf child clade -- so it is a poor fit for vectorised TensorFlow but
// a natural fit for a compiled kernel, exactly like the other native traversal
// ops in this directory.
//
// There is no gradient: topologies are discrete and the SBN's parameter
// gradients flow through the differentiable ``log_prob`` (pure TensorFlow), not
// through sampling. The op therefore only implements the forward draw.
//
// Support layout (all clade/candidate ids are 0-based; a tree has n taxa, so
// 2n-1 nodes and n-1 internal nodes):
//   child_offsets        [num_clades+1]  CSR offsets: clade c's candidate child
//                                        subsplits are [offsets[c], offsets[c+1])
//   candidate_left_clade  [num_cand]     left child clade id of each candidate
//   candidate_right_clade [num_cand]     right child clade id of each candidate
//   candidate_probs       [num_cand]     conditional prob of each candidate,
//                                        already normalised within its parent
//   clade_leaf_taxon      [num_clades]   taxon id if the clade is a leaf, else -1
// plus scalar attrs: root_clade_id, taxon_count, n_samples, seed.
//
// Outputs (treeflow index convention: leaves 0..n-1, internal nodes assigned in
// postorder, root last = 2n-2):
//   parent_indices    [n_samples, 2n-2]  parent node id of every non-root node
//   candidate_indices [n_samples, n-1]   candidate/param index chosen at each
//                                        internal node (aligned with node id:
//                                        entry k is internal node n+k)
//   node_clade_ids    [n_samples, 2n-1]  support clade id of every node

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;

REGISTER_OP("SbnSample")
    .Attr("T: {float, double}")
    .Attr("root_clade_id: int")
    .Attr("taxon_count: int")
    .Attr("n_samples: int")
    .Attr("seed: int")
    .Input("child_offsets: int32")
    .Input("candidate_left_clade: int32")
    .Input("candidate_right_clade: int32")
    .Input("candidate_probs: T")
    .Input("clade_leaf_taxon: int32")
    .Output("parent_indices: int32")
    .Output("candidate_indices: int32")
    .Output("node_clade_ids: int32")
    .SetShapeFn([](InferenceContext* c) {
      int64_t taxon_count, n_samples;
      TF_RETURN_IF_ERROR(c->GetAttr("taxon_count", &taxon_count));
      TF_RETURN_IF_ERROR(c->GetAttr("n_samples", &n_samples));
      const int64_t node_count = 2 * taxon_count - 1;
      c->set_output(0, c->Matrix(n_samples, node_count - 1));
      c->set_output(1, c->Matrix(n_samples, taxon_count - 1));
      c->set_output(2, c->Matrix(n_samples, node_count));
      return OkStatus();
    });

template <typename T>
class SbnSampleOp : public OpKernel {
 public:
  explicit SbnSampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("root_clade_id", &root_clade_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("taxon_count", &taxon_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("n_samples", &n_samples_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& offsets_t = ctx->input(0);
    const Tensor& left_t = ctx->input(1);
    const Tensor& right_t = ctx->input(2);
    const Tensor& probs_t = ctx->input(3);
    const Tensor& leaf_taxon_t = ctx->input(4);

    const int32* offsets = offsets_t.flat<int32>().data();
    const int32* left = left_t.flat<int32>().data();
    const int32* right = right_t.flat<int32>().data();
    const T* probs = probs_t.flat<T>().data();
    const int32* leaf_taxon = leaf_taxon_t.flat<int32>().data();

    const int64_t n = taxon_count_;
    const int64_t node_count = 2 * n - 1;

    OP_REQUIRES(ctx, n >= 1,
                errors::InvalidArgument("taxon_count must be >= 1"));

    Tensor* parent_t = nullptr;
    Tensor* candidate_t = nullptr;
    Tensor* node_clade_t = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {n_samples_, node_count - 1}, &parent_t));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {n_samples_, n - 1}, &candidate_t));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, {n_samples_, node_count}, &node_clade_t));

    int32* parent_out = parent_t->flat<int32>().data();
    int32* candidate_out = candidate_t->flat<int32>().data();
    int32* node_clade_out = node_clade_t->flat<int32>().data();

    auto work = [&](int64_t begin, int64_t end) {
      // Temp tree built top-down: one entry per sampled node.
      std::vector<int32> t_clade, t_left, t_right, t_cand;
      std::vector<int32> tf_id;
      // Explicit stacks (avoid deep recursion on unbalanced trees).
      std::vector<int32> expand_stack;
      std::vector<std::pair<int32, int32>> post_stack;  // (temp idx, phase)
      for (int64_t s = begin; s < end; ++s) {
        // Independent, reproducible RNG per sample.
        std::seed_seq seq{static_cast<int64_t>(seed_), s};
        std::mt19937_64 rng(seq);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        t_clade.clear();
        t_left.clear();
        t_right.clear();
        t_cand.clear();

        auto make_node = [&](int32 clade_id) -> int32 {
          int32 idx = static_cast<int32>(t_clade.size());
          t_clade.push_back(clade_id);
          t_left.push_back(-1);
          t_right.push_back(-1);
          t_cand.push_back(-1);
          return idx;
        };

        int32 root_idx = make_node(static_cast<int32>(root_clade_id_));
        expand_stack.clear();
        expand_stack.push_back(root_idx);
        while (!expand_stack.empty()) {
          int32 idx = expand_stack.back();
          expand_stack.pop_back();
          int32 clade = t_clade[idx];
          if (leaf_taxon[clade] >= 0) continue;  // leaf clade: no subsplit
          int32 start = offsets[clade];
          int32 stop = offsets[clade + 1];
          // Draw a candidate proportionally to (already normalised) probs; scale
          // the uniform by the group total to absorb any tiny drift.
          double total = 0.0;
          for (int32 j = start; j < stop; ++j) total += probs[j];
          double u = uniform(rng) * total;
          double acc = 0.0;
          int32 chosen = stop - 1;
          for (int32 j = start; j < stop; ++j) {
            acc += probs[j];
            if (u <= acc) {
              chosen = j;
              break;
            }
          }
          int32 l = make_node(left[chosen]);
          int32 r = make_node(right[chosen]);
          t_left[idx] = l;
          t_right[idx] = r;
          t_cand[idx] = chosen;
          expand_stack.push_back(l);
          expand_stack.push_back(r);
        }

        // Assign treeflow ids by postorder (children before parents), so ids
        // satisfy the children<parent convention with the root last.
        int64_t num_nodes = static_cast<int64_t>(t_clade.size());
        tf_id.assign(num_nodes, -1);
        int32 next_internal = static_cast<int32>(n);

        int32* parent_s = parent_out + s * (node_count - 1);
        int32* candidate_s = candidate_out + s * (n - 1);
        int32* node_clade_s = node_clade_out + s * node_count;

        post_stack.clear();
        post_stack.emplace_back(root_idx, 0);
        while (!post_stack.empty()) {
          auto& top = post_stack.back();
          int32 idx = top.first;
          int32 clade = t_clade[idx];
          if (t_left[idx] < 0) {  // leaf
            int32 id = leaf_taxon[clade];
            tf_id[idx] = id;
            node_clade_s[id] = clade;
            post_stack.pop_back();
            continue;
          }
          if (top.second == 0) {
            top.second = 1;
            post_stack.emplace_back(t_left[idx], 0);
            post_stack.emplace_back(t_right[idx], 0);
          } else {
            int32 id = next_internal++;
            tf_id[idx] = id;
            parent_s[tf_id[t_left[idx]]] = id;
            parent_s[tf_id[t_right[idx]]] = id;
            candidate_s[id - static_cast<int32>(n)] = t_cand[idx];
            node_clade_s[id] = clade;
            post_stack.pop_back();
          }
        }
      }
    };

    // Shard across samples; each sample's traversal is independent.
    auto* workers = ctx->device()->tensorflow_cpu_worker_threads();
    const int64_t cost_per_sample = 8 * node_count;
    Shard(workers->num_threads, workers->workers, n_samples_, cost_per_sample,
          work);
  }

 private:
  int64_t root_clade_id_;
  int64_t taxon_count_;
  int64_t n_samples_;
  int64_t seed_;
};

#define REGISTER_CPU(T)                                              \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SbnSample").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SbnSampleOp<T>);

REGISTER_CPU(float)
REGISTER_CPU(double)

#undef REGISTER_CPU
