# HMC Sampling of Time Trees: Geometry, Pathologies, and Preconditioning

## Overview

This report documents findings from validating `fit_fixed_topology_hmc` against
direct simulation from `BirthDeathContemporarySampling`, and analyses the
geometric properties of the ratio parameterisation that cause NUTS to produce
incorrect marginal distributions for shallow internal nodes. It then surveys
sampler alternatives and proposes a tractable preconditioning strategy.

---

## 1. Validation experiment

`experiments/validate_hmc_birth_death.ipynb` compares two ways of sampling from
`BirthDeathContemporarySampling` on a fixed 8-taxon topology:

- **Direct simulation** (2000 samples) — the CPP backward algorithm draws
  speciation times iid from the BD prior and assigns them to the topology via
  a uniform random ranking.
- **NUTS HMC** (1000 samples, 500 burn-in) — `fit_fixed_topology_hmc` with
  dual-averaging step-size adaptation, operating in the unconstrained space
  defined by the ratio bijector.

The two-sample Kolmogorov–Smirnov test on each of the 7 internal node heights
reveals that nodes adjacent to leaves (shallow nodes) fail at p < 0.05, with
HMC samples showing excess mass near zero compared to direct simulation. Deeper
nodes (closer to the root) pass. The discrepancy is not a bug in the log-prob
or log-det-Jacobian — both are analytically correct — but a consequence of
poor posterior geometry in the unconstrained parameterisation.

---

## 2. The ratio transform and its geometry

### 2.1 Bijector structure

`BirthDeathContemporarySampling._default_event_space_bijector` returns a
`TreeRatioBijector`, which chains:

```
NodeHeightRatioChainBijector = Chain([
    NodeHeightRatioBijector,          # ratios → heights (tree traversal)
    Blockwise([Sigmoid(), Exp()])     # unconstrained → ratios/root-scale
])
```

The forward map from unconstrained `x ∈ Rⁿ⁻¹` to node heights `h ∈ Rⁿ⁻¹` is:

1. **Blockwise**: first `n−2` elements through `Sigmoid` → ratios `r ∈ (0,1)`;
   last element through `Exp` → root scale `s > 0`.
2. **NodeHeightRatioBijector**: preorder traversal setting  
   `h_root = s`  
   `h_i = h_{parent(i)} × r_i`  (for contemporary trees with zero anchor heights)

### 2.2 The Jacobian is sparse and tree-structured

The Jacobian `J = ∂h/∂x` satisfies `J_{ij} ≠ 0` only when parameter `j`
corresponds to node `i` or an ancestor of `i`. In preorder ordering this gives
a **sparse upper-triangular matrix**. Each row has at most `depth(i)` non-zeros.
Analytically, for a non-root node `i` with ancestors `a₁, …, aₖ` (root = `aₖ`):

```
∂h_i/∂x_i    = h_{parent(i)} · sigmoid′(x_i)
∂h_i/∂x_{aⱼ} = h_i · sigmoid′(x_{aⱼ}) / sigmoid(x_{aⱼ})   [for aⱼ a non-root ancestor]
∂h_i/∂x_root = h_i / s                                        [for the root parameter]
```

The log-det-Jacobian of `NodeHeightRatioBijector` is also analytically available:

```
ildj = −Σᵢ log(h_{parent(i)})     [sum over non-root internal nodes]
```

This has been verified to be correct by analytical differentiation of the
inverse map `r_i = h_i / h_{parent(i)}`.

### 2.3 BD log-prob Hessian w.r.t. heights is diagonal

The BD log-prob decomposes as a sum over individual heights with no cross terms:

```
∂² log p / ∂hᵢ ∂hⱼ = 0   for i ≠ j
```

The diagonal entries are closed-form. Let `qᵢ = ρ + (1−ρ−a) exp(−r hᵢ)`:

```
∂² log p / ∂hᵢ² = −2r²ρ(1−ρ−a) exp(−r hᵢ) / qᵢ²
```

This is analytically available without autodiff and strictly negative (the
prior is log-concave in heights), so `D = −∇²_h log p` is a positive diagonal
matrix.

---

## 3. Why NUTS fails for shallow nodes

### 3.1 Funnel geometry

The ratio parameterisation creates a **hierarchical funnel** analogous to
Neal's funnel in hierarchical models. For a leaf-adjacent node `i`:

```
h_i = h_{parent(i)} × sigmoid(x_i)
```

The effective scale of `x_i` in unconstrained space depends on
`h_{parent(i)}`: when the parent height is small, a unit step in `x_i`
produces a tiny change in `h_i`, while a large parent height amplifies the
same step. NUTS adapts a global (or diagonal) step size and cannot account for
this position-dependent scaling.

### 3.2 Tail behaviour near zero

The BD prior has finite density at `h_i = 0`. In unconstrained space this
corresponds to `sigmoid(x_i) → 0`, i.e. `x_i → −∞`. The target density
acquires an exponential tail in `x_i`, which NUTS may not traverse efficiently
— particularly when the optimal step size in this tail is very different from
the step size in the bulk.

### 3.3 Depth dependence of the pathology

Nodes at greater depth (further from the root) are most affected because:

1. Their heights are a product of multiple ratios, compounding the funnel.
2. Their parent heights can be small, creating tight funnels.
3. NUTS's diagonal mass matrix cannot capture the induced
   ancestor-descendant correlations in unconstrained space.

Nodes close to the root are less affected: the root height is parameterised by
`Exp(x_root)`, which has log-linear tails and no parent-dependent scaling.

---

## 4. Alternative samplers

| Sampler | Handles funnel? | Exploits BD structure? | Complexity |
|---|---|---|---|
| NUTS (current) | Poor | No | Low |
| NUTS + fixed metric preconditioner | Partial | Yes (analytically) | Low |
| NUTS + dense adapted mass matrix | Partial | No | Medium |
| Gibbs with local uniform moves | Yes (avoids it) | Yes | Medium |
| Operator MCMC (BEAST-style) | Yes (avoids it) | Yes | Medium |
| Sequential Monte Carlo (SMC) | Yes | Yes (BD prior ≡ proposal) | Medium |
| Riemannian Manifold HMC | Yes (fully) | No | High |

### 4.1 Gibbs with local uniform moves

For fixed topology, each internal node `i` has a valid height range
`(max(child heights), parent height)`. A Gibbs sampler that resamples each
height uniformly within this interval — optionally with a log-uniform or
slice-sampling proposal to account for the BD prior — completely avoids the
ratio parameterisation and has no funnel. This is the approach used by BEAST2's
`NodeUniform` and `TreeScaler` operators and is straightforward to implement.

### 4.2 SMC from the BD prior

Since `BirthDeathContemporarySampling` can be sampled exactly and efficiently,
SMC that anneals from the BD prior to the posterior is particularly natural.
With no likelihood (pure prior validation), SMC reduces to direct simulation.
With data, the particle weights absorb the likelihood, and resampling/MCMC
moves maintain diversity. SMC is robust to funnel geometry because it starts
from exact prior samples.

### 4.3 Full Riemannian Manifold HMC

RMHMC uses the position-dependent metric

```
G(x) = J(x)ᵀ D(x) J(x) + H_ldj(x)
```

and a modified leapfrog integrator (implicit, requiring fixed-point iteration
at each step, or the SoftAbs approximation). This is the theoretically correct
solution but requires a custom `TransitionKernel` in TFP, is expensive per
step, and gains most over NUTS only in the deep funnel region that accounts for
a small fraction of posterior mass.

---

## 5. Fixed metric preconditioning (recommended near-term approach)

### 5.1 Concept

Rather than a full position-dependent Riemannian metric, evaluate the metric
analytically at the prior mean `h*` and use it as a **fixed mass matrix** in
standard NUTS. This corrects for the average geometric distortion of the ratio
transform without requiring implicit integration or a custom kernel.

The preconditioner is:

```
G* = J(h*)ᵀ D(h*) J(h*)
```

where:
- `h* = E_BD[heights]` — prior mean, computed from ~100–200 direct samples or
  analytically from order-statistics of the BD prior
- `J(h*)` — ratio-transform Jacobian at `h*`, computed in a single preorder
  traversal; sparse, tree-structured, O(n · depth) non-zeros
- `D(h*)` — diagonal BD Hessian at `h*`, closed-form, O(n) computation

`G*` has **ancestor-descendant sparsity**: entry `(j, k)` is non-zero iff `j`
and `k` are in an ancestor–descendant relationship in the topology. For a
balanced binary tree with `n` taxa this gives O(n log n) non-zeros vs O(n²)
for a dense matrix. For the 8-taxon case (7×7) the matrix is trivially small.

### 5.2 Implementation via TransformedTransitionKernel

No custom kernel is needed. Apply `G*` as a linear change of variables
wrapping NUTS:

```python
# 1. Estimate prior mean heights from direct simulation
h_star = direct_dist.sample(200, seed=0).node_heights.numpy().mean(axis=0)

# 2. Compute J(h*) analytically [preorder traversal, O(n·depth)]
J_star = compute_ratio_jacobian(topology, h_star)   # (n-1) × (n-1)

# 3. Compute D(h*) = diagonal BD Hessian             [O(n), closed form]
D_star = bd_diagonal_hessian(h_star, r, a, rho)     # (n-1,)

# 4. Form G* and Cholesky factorize                  [O(n·depth²)]
G_star = J_star.T @ np.diag(D_star) @ J_star
L_star = np.linalg.cholesky(G_star)

# 5. Wrap NUTS — HMC operates in z = L*⁻¹ x space
#    (equivalent to x-space with mass matrix G*)
precond = tfp.bijectors.ScaleMatvecTriL(
    scale_tril=tf.constant(L_star, dtype=tf.float64)
)
kernel = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=0.1),
    bijector=precond,
)
```

`TransformedTransitionKernel` handles the log-det-Jacobian correction
automatically; `target_log_prob_fn` is unchanged.

### 5.3 What this corrects and what it does not

**Corrected:**
- Global scale mismatch between nodes at different depths
- Ancestor–descendant correlations in unconstrained space (off-diagonal G*)
- Average funnel width at the prior mean

**Not corrected:**
- Position-dependent funnel narrowing away from `h*` — residual error is
  O(‖h − h*‖) in the metric approximation
- Deep funnel behaviour when parent heights are much smaller than `h*`

For the BD prior validation the posterior concentrates near `h*` by
definition, so the approximation should be accurate. For posteriors
with strong likelihood constraints the approximation degrades but remains
a useful preconditioner.

### 5.4 Pieces requiring implementation

Only one genuinely new function is needed:

```
compute_ratio_jacobian(topology, h_star) → ndarray (n-1, n-1)
```

A preorder traversal computing `∂h_i/∂x_j` for all ancestor-descendant
pairs `(i, j)`. The existing `ratios_to_node_heights` in
`treeflow/traversal/ratio_transform.py` performs the same traversal for
the forward pass; the Jacobian version accumulates partial derivatives
along the same paths.

`bd_diagonal_hessian` is a vectorised application of the closed-form
second derivative formula in §2.3.

---

## 6. Summary

The KS test failures in the validation notebook are explained by funnel
geometry arising from the ratio parameterisation, not by any error in the
log-prob or bijector. The funnel is worst for nodes adjacent to leaves and
is a known pathology of hierarchical tree parameterisations.

The most tractable fix within the existing HMC framework is **fixed metric
preconditioning**: compute `G* = J(h*)ᵀ D(h*) J(h*)` analytically once
before sampling and apply it as a constant mass matrix via
`TransformedTransitionKernel`. This requires no new kernel, no implicit
integration, and exploits the fact that both the ratio-transform Jacobian
and the BD log-prob Hessian are analytically available in closed form.

Longer-term, a Gibbs sampler with local uniform node-height moves or SMC
annealing from the BD prior would entirely bypass the funnel, at the cost
of not being a drop-in replacement for `fit_fixed_topology_hmc`.
