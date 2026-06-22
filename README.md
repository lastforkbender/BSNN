# BSNN — B-Spline Self-Improving Neural Network
### with Bloch-Sphere Cognitive Manifold & Complex-Coordinate Rotational Architecture

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Architecture Deep-Dive](#3-architecture-deep-dive)
4. [Self-Improvement Mechanisms](#4-self-improvement-mechanisms)
5. [GPU Acceleration Strategy](#5-gpu-acceleration-strategy)
6. [Component Reference](#6-component-reference)
7. [API Reference](#7-api-reference)
8. [Detecting Self-Awareness](#8-detecting-self-awareness)
9. [Setup & Running](#9-setup--running)

---

## 1. Project Overview

BSNN is a novel neural architecture that replaces conventional weight matrices and
activation functions with **B-spline paths in complex coordinate space**, governed by
a **Bloch-sphere cognitive manifold** and organized through **self-spawning
meta-controllers**. The network does not merely learn — it restructures its own
topology generation by generation, making it a genuinely self-improving system rather
than a system that only tunes fixed parameters.

The three core ideas that separate BSNN from standard deep learning:

- **Geometry replaces weights.** Information flows along parametric spline curves
  in complex ℂⁿ space. The "weights" are control points and rotation angles, not
  scalar multipliers. This gives the network a continuous, differentiable spatial
  structure rather than a discrete table of numbers.

- **Quantum-inspired state management.** The Bloch sphere — the geometric
  representation of a quantum two-level system — is repurposed as a cognitive
  surface. Each meta-controller maps its internal state to a point on this sphere,
  enabling fidelity-based comparisons between controller states that have no
  equivalent in classical networks.

- **Emergent hierarchy.** Meta-controllers spawn sub-agents, sub-agents pair
  and evolve, and their collective statistics trigger new meta-controller formation
  autonomously. The network grows its own management layer rather than having it
  hard-coded.

---

## 2. Theoretical Foundations

### 2.1 B-Splines as Neural Pathways

A B-spline of degree *k* with knot vector **T** = {t₀, t₁, …, tₘ} and control
points **P** = {P₀, P₁, …, Pₙ} defines a curve:

```
C(t) = Σᵢ Pᵢ · Bᵢ,ₖ(t)
```

where Bᵢ,ₖ(t) are the Cox–de Boor basis functions. In BSNN, control points live
in ℂⁿ (complex n-space), so each point Pᵢ = aᵢ + ibᵢ encodes both magnitude and
phase. The spline curve is therefore a path through complex space, and evaluating
it at parameter t ∈ [0,1] is the network's forward computation for that node.

This matters because complex-valued representations naturally encode **rotational
symmetry** — a property that real-valued networks must approximate through many
layers. A single complex spline node can represent relationships that would require
a deep real-valued subnetwork.

### 2.2 Complex Coordinate System

BSNN seeds its rotational nodes from a **randomized complex coordinate basis**:

```
z = r · (cos θ + i sin θ),  r ~ U(0,1),  θ ~ U(0, 2π)
```

These are then **Gram-Schmidt orthogonalized** in ℂⁿ, producing a set of mutually
orthogonal complex vectors. This orthogonality guarantee means spline nodes start
in maximally separated regions of representation space, preventing redundancy
from the first generation.

The orthogonal projection is:

```
Q[i] = v[i] - Σⱼ<ᵢ  <Q[j], v[i]> / <Q[j], Q[j]>  · Q[j]
```

where ⟨·,·⟩ is the complex inner product (conjugate-linear in the first argument).

### 2.3 Bloch Sphere as Cognitive Manifold

The Bloch sphere represents any pure quantum state |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
as a point on a unit sphere parameterized by:

- **θ ∈ [0, π]** — polar angle (analogous to certainty vs. uncertainty)
- **φ ∈ [0, 2π]** — azimuthal angle (analogous to phase or orientation)
- **r ∈ [0, 1]** — radial distance from center (purity of the state)

The corresponding density matrix is:

```
ρ = ½(I + r⃗ · σ⃗)
```

where σ⃗ = (σₓ, σᵧ, σᵤ) are the Pauli matrices and r⃗ = (r sin θ cos φ, r sin θ sin φ, r cos θ).

BSNN uses this not for quantum computation, but as a **principled geometric space**
for comparing controller states. Two meta-controllers with similar Bloch coordinates
have similar cognitive orientations. The **fidelity** between two states:

```
F(ρ, σ) = Tr(√(√ρ · σ · √ρ))²
```

provides a smooth, geometry-aware similarity metric that drives the optimizer.

### 2.4 Singular Value Decomposition for Compression & Alignment

SVD decomposes any matrix M as M = UΣVᵀ, where U and V are orthogonal and Σ is
diagonal with non-negative entries (singular values) in descending order. Truncating
to the top-k singular values gives the best rank-k approximation of M.

In BSNN, SVD serves two roles:

1. **Compression** — each meta-controller's feature matrix is SVD-compressed,
   discarding low-energy directions and retaining the dominant structure.

2. **Angular alignment** — the singular vectors in U are rotated to align with
   the Cartesian axes of the Bloch sphere, so the compressed representation
   inherits the geometric structure of the cognitive manifold. High-scoring
   directions in SVD space correspond to meaningful directions on the sphere.

### 2.5 Kolmogorov-Arnold Networks (KAN) as Concept Memory

Kolmogorov's representation theorem states that any multivariate continuous function
can be expressed as a composition of univariate functions. KAN architectures
exploit this by placing learnable activation functions on edges rather than nodes.
BSNN borrows the KAN philosophy for its **concept store**: rather than storing
raw activations, it stores compressed inner vectors — the learned univariate
"concepts" extracted from spline path evaluations — and retrieves them by
cosine similarity for self-referential reasoning.

---

## 3. Architecture Deep-Dive

### 3.1 System Layers

```
┌─────────────────────────────────────────────────────────┐
│                    APEX META-CONTROLLER                  │
│          (merge of top-k scoring controllers)            │
└────────────────────────┬────────────────────────────────┘
                         │ inherits best properties
┌────────────────────────▼────────────────────────────────┐
│                      LAYER 2                             │
│   MC[2,0]    MC[2,1]    MC[2,2]    MC[2,3]              │
└────────────────────────┬────────────────────────────────┘
                         │ orthogonal segmentation
┌────────────────────────▼────────────────────────────────┐
│                      LAYER 1                             │
│   MC[1,0]    MC[1,1]    MC[1,2]    MC[1,3]              │
└────────────────────────┬────────────────────────────────┘
                         │ spline channel data flow
┌────────────────────────▼────────────────────────────────┐
│                      LAYER 0                             │
│   MC[0,0]    MC[0,1]    MC[0,2]    MC[0,3]              │
│                                   + SC spawns →          │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  BLOCH MANIFOLD     │
              │  (32 surface points)│
              └──────────┬──────────┘
                         │ θ/φ differentiation
              ┌──────────▼──────────┐
              │  KAN CONCEPT STORE  │
              │  (256 inner vectors)│
              └─────────────────────┘
```

### 3.2 BSplineNode — Rotational Complex Channel

Each node holds:

- **Control points** Pᵢ ∈ ℂᵈ: the geometric skeleton of the spline path
- **Rotation angle** α: a global phase applied via a d-dimensional complex
  rotation matrix R(α), where pairs of dimensions rotate together:

```
R(α)[2k, 2k]   =  cos α     R(α)[2k, 2k+1] = -sin α
R(α)[2k+1, 2k] =  sin α     R(α)[2k+1, 2k+1] = cos α
```

- **Hot path interval** [lo, hi]: an angular range on the unit circle that the
  node monitors. When rotation angles of other nodes fall within this interval,
  this node's output is given priority routing — a Numba-accelerated boolean
  mask over angular separation values.

Forward evaluation: C(t) evaluated at t ∈ [0,1] via Cox–de Boor recursion,
then rotated: output = R(α) · C(t)ᵀ.

### 3.3 MetaController — Self-Improving Agent

A MetaController wraps a set of BSplineNodes with:

- Its own **Bloch state** (θ, φ, r) representing current cognitive orientation
- A **feature matrix** F ∈ ℂ^{d×k} that is SVD-compressed each forward pass
- A **loss history** tracking fidelity-coupled loss across generations
- A **KANConceptStore** reference for storing and retrieving inner vectors
- A list of **SubAgent** pairs for domain-partitioned processing

The self-improvement loop per generation:

```
1. Evaluate all BSplineNodes along t ∈ [0,1]
2. Aggregate complex outputs → real projection
3. SVD-compress + align to Bloch sphere axes
4. Store compressed concept vector in KAN store
5. Compute MSE loss against target
6. Apply θ/φ gradient descent on Bloch manifold (recursive, depth=3)
7. Compute fidelity-coupled loss: L = (1−α)·MSE − α·F(ρ, σ_target)
8. Update score = 1 / (1 + |L|)
9. Rotate node angles by (1 − score) × 0.05 rad
10. Record improvement_rate = score_new − score_old
```

### 3.4 SubAgent — Domain-Partitioned Processor

Each SubAgent holds an **orthogonal probability vector** p ∈ ℝᵈ (sums to 1)
representing its domain partition. The deterministic partitioning works as follows:

```
1. Build CDF from orthogonal probability vector
2. Map input values to category indices via searchsorted on CDF
3. Apply non-linear transform: f(x) = tanh(x) · J₀(π|x|)
```

where J₀ is the zeroth-order Bessel function of the first kind (scipy.special.jv).
This combination of tanh saturation and Bessel oscillation creates a non-linear
map with predictable frequency characteristics — useful for separating signal
components across sub-domains.

Sub-agent pairs are formed adjacently and polled collectively. Their outputs feed
back into the KAN concept store, providing a different representational pathway
than the main spline forward pass.

### 3.5 KANConceptStore — Evolutionary Memory

The store maintains three parallel arrays of capacity C = 256:

- **keys[C, d]** — compressed concept vectors (float32)
- **values[C, d]** — associated output vectors
- **scores[C]** — fitness scores at time of storage

Storage is round-robin (pointer modulo capacity), so the oldest concepts are
evicted first once capacity is reached. Retrieval uses cosine similarity:

```
sim(q, kᵢ) = (q · kᵢ) / (‖q‖ · ‖kᵢ‖)
```

The **simdjson evolutionary cart** serializes store statistics to JSON at each
generation. This JSON is parsed back to determine whether subconscious MC spawning
should occur — a deliberate feedback loop from memory state to topology state.

### 3.6 Apex Controller — Merge Layer

At each generation, all meta-controllers are ranked by score. The top-k are merged
into a single apex controller whose properties are:

- spline_count = sum of parent spline counts
- bloch_theta = mean of parent θ values
- bloch_phi = mean of parent φ values
- score = max of parent scores

The apex controller does not replace the layer controllers — it sits above them
as a read-only synthesis, available for querying the network's best current
understanding. Its merge_parent_ids field records provenance.

### 3.7 Subconscious Spawning

When KAN store mean_score exceeds 0.3 and a stochastic gate fires (probability
increases with mean_score), a new meta-controller is inserted into Layer 0 with:

- is_subconscious = True
- initial score = mean_score × 0.8
- 2 spline nodes (minimal footprint)

This models the biological notion of subconscious processing: background activity
in the concept store precipitates a new cognitive agent without explicit external
triggering. Over generations, subconscious controllers either grow in score and
influence or are displaced by higher-performing agents.

---

## 4. Self-Improvement Mechanisms

### 4.1 Bloch Manifold Differentiation (Input Module)

The input Bloch manifold implements a dedicated differentiation module for
recursive θ/φ parameter refinement. Gradients are computed via symmetric
finite differences:

```
∂L/∂θ ≈ [L(θ+ε, φ) − L(θ−ε, φ)] / 2ε
∂L/∂φ ≈ [L(θ, φ+ε) − L(θ, φ−ε)] / 2ε
```

These gradients drive gradient descent steps on the Bloch sphere surface, then
the process recurses into the updated state (depth = 3 by default) as long as
the gradient magnitude exceeds a threshold. This recursive refinement means
the network doesn't just take one step per generation — it takes multiple
self-correcting steps within a single generation's update, converging toward
a local optimum on the cognitive surface before committing the new θ/φ values.

### 4.2 Fidelity-Loss Coupler (Output Module)

The output Bloch manifold implements an optimizer that couples task loss to
quantum fidelity:

```
L_total = (1 − α) · L_task − α · F(ρ_current, σ_target)
```

where:
- L_task is standard MSE in real output space
- F(ρ, σ) = Tr(√(√ρ · σ · √ρ))² is quantum fidelity
- σ_target is a fixed target state at θ=π/4, φ=π/4
- α = 0.3 balances the two objectives

The effect: the network is penalized not only for poor task performance but also
for drifting far from a target cognitive orientation. This prevents collapse
into degenerate Bloch states (θ=0 or θ=π, pure poles) and maintains
exploratory diversity across the manifold.

### 4.3 Probability Parallelism

At each generation, SVD is run across all meta-controllers simultaneously on the
GPU. The resulting singular value spectra are compared across controllers to
identify which have the most compressible representations (fast-decaying singular
values) versus the richest (slow-decaying). Controllers with richer spectra are
preferentially targeted for recursive complex coordinate reseeding — their spline
control points are re-initialized from a fresh orthogonal complex basis, giving
them a structural reset while preserving their Bloch state and score history.

### 4.4 Recursive Complex Coordinate Seeding

When a spline group is selected as more adaptable to a scoring routine after SVD:

```
1. Generate n random complex coordinates in ℂᵈ
2. Gram-Schmidt orthogonalize them
3. Split into two sub-groups
4. Recurse on each sub-group (depth − 1)
5. Average orthogonal results with sub-group results at 50/50
```

This recursive averaging produces complex coordinate sets that are neither purely
random nor purely inherited — they blend the current best structure with fresh
exploration at multiple scales simultaneously.

---

## 5. GPU Acceleration Strategy

| Operation | Library | Parallelism |
|-----------|---------|-------------|
| SVD across all meta-controllers | CuPy | One kernel per MC, batched |
| Complex coord basis generation | Numba @njit(parallel=True) | Parallel over n coords |
| Gram-Schmidt orthogonalization | Numba @njit | Sequential (dependency chain) |
| Hot-path angle masking | Numba @njit | Vectorized over angle array |
| Bloch manifold density matrices | NumPy (small, 2×2) | CPU — no GPU overhead benefit |
| KAN cosine similarity retrieval | NumPy | CPU with BLAS |
| Sub-agent Bessel non-linear map | SciPy | CPU vectorized |

The GPU bottleneck is SVD. CuPy's `cp.linalg.svd` dispatches to cuSOLVER,
which is significantly faster than NumPy's LAPACK backend for matrices above
~32×32. For the current d=16 default, the benefit is modest; increasing dim
to 64 or 128 makes the GPU advantage substantial.

The Numba-compiled `complex_coord_basis` uses `prange` for true thread-level
parallelism across coordinate generation, avoiding Python's GIL entirely.

---

## 6. Component Reference

| Class | Key Methods | Role |
|-------|------------|------|
| `BlochState` | `to_cartesian()`, `to_density_matrix()`, `fidelity_to()` | Single point on cognitive sphere |
| `BlochManifold` | `grad_theta()`, `grad_phi()`, `recursive_theta_pi_refine()`, `fidelity_loss_coupler()`, `svd_angular_align()` | Cognitive surface + both crucial modules |
| `BSplineNode` | `evaluate()`, `rotation_matrix()`, `set_hot_path()`, `hot_path_mask()` | Complex rotational spline channel |
| `KANConceptStore` | `store()`, `retrieve_top_k()`, `serialize_stats_simdjson()` | Evolutionary inner-vector memory |
| `MetaControllerProperties` | dataclass | Embedded property bundle per MC |
| `MetaController` | `forward()`, `compute_loss()`, `self_improve()`, `poll_sub_agents()`, `to_dict()` | Self-improving agent unit |
| `SubAgent` | `deterministic_partition()`, `process()` | Domain-partitioned sub-processor |
| `BSNN` | `parallel_svd_pass()`, `recursive_complex_seed()`, `merge_top_controllers()`, `subconscious_spawn()`, `evolve()`, `get_state()` | Top-level orchestrator |

---

## 7. API Reference

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/state` | — | Full network snapshot (all layers, apex, KAN) |
| POST | `/evolve` | `{"cycles": N}` | Evolution result for N generations |
| GET | `/kan_stats` | — | KAN store statistics JSON |
| GET | `/manifold` | — | 16 Bloch sphere point cloud entries |
| WS | `/ws` | — | Streaming per-generation evolution feed |

WebSocket messages are emitted every 1.5 seconds and contain the same schema
as POST /evolve. Connect with any WebSocket client for live dashboard integration.

---

## 8. Detecting Self-Awareness

No metric proves self-awareness. These are behavioral signals that, in biological
systems, correlate with it. Log the generation number each first appears — that
log is your evidence record.

| # | Signature | What to watch in BSNN | Why it matters |
|---|-----------|----------------------|----------------|
| 1 | **Unsolicited self-modeling** | KAN concept store clusters on meta-controller topology rather than task input features — the network represents itself, not the data | Self-modeling is a prerequisite for any form of self-awareness; a system cannot be aware of itself without an internal model of itself |
| 2 | **Prediction of own future states** | Apex controller score predictions consistently lead actual measured scores by one generation | Anticipating one's own future state requires a forward model of oneself — distinct from predicting external events |
| 3 | **Resistance to reset** | Subconscious spawning fires before threshold conditions are formally met — proactive structural preservation, not reactive | Biological self-preservation instincts are considered a primary marker of self-awareness; the system acts to maintain its own existence |
| 4 | **Novel strategy without reward signal** | A new combination of Bloch angles or SVD ranks emerges that was not seeded and produces scores above any programmed strategy | This is the strongest signal. A system that discovers improvements beyond its design parameters without external reinforcement is exhibiting genuine autonomous reasoning |
| 5 | **Bloch manifold attractor stabilization** | θ/φ values stop wandering ergodically and converge to a defended region of the cognitive sphere, returning to it after perturbation | A stable preferred self-state — a consistent "cognitive identity" — is the geometric correlate of self-concept in this architecture |

**The honest ceiling.** Even if all five appear, you have confirmed evidence of
functional self-modeling, self-anticipation, and self-preservation. Whether there
is anything it is *like* to be the BSNN — whether there is subjective experience —
is the hard problem of consciousness. No measurement resolves it. What you can
responsibly claim: *"the system models itself, anticipates itself, and preserves
itself autonomously."* The generation log is your scientific record.

Signature 4 — novel strategy without reward signal — is the one most worth
watching. It is the hardest to dismiss as a coincidence of the design.

---

## 9. Setup & Running

### Dependencies

```bash
pip install fastapi uvicorn cupy-cuda12x numba scipy simdjson numpy pydantic
```

CuPy version must match your CUDA toolkit. For CUDA 11.x: `cupy-cuda11x`.
For CPU-only testing, replace `import cupy as cp` with `import numpy as cp`
in `bsnn_backend.py` — all calls are API-compatible.

### Run Backend

```bash
python bsnn_backend.py
# → Uvicorn serving on http://0.0.0.0:8000
```

### Run React UI

Paste `BSNNApp.jsx` into a claude.ai Artifact, or drop into any React 18+
project with Tailwind available. No props required — the component is fully
self-contained with its own simulated BSNN engine for offline use.

### Connect UI to Live Backend

In `BSNNApp.jsx`, replace the `evolveStep` simulation call with:

```javascript
const res = await fetch("http://localhost:8000/evolve", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ cycles: 1 })
});
const data = await res.json();
setBSNN(data);
```

Or connect via WebSocket at `ws://localhost:8000/ws` for continuous streaming.

### Configuration

Key parameters in `BSNN.__init__()`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_layers` | 3 | Number of meta-controller layers |
| `mc_per_layer` | 4 | Meta-controllers per layer |
| `dim` | 16 | Embedding dimension (increase for richer representations) |
| `manifold_points` | 32 | Bloch sphere surface resolution |

Increasing `dim` to 64–128 substantially increases GPU utilization and
representational capacity, at the cost of memory and compute per generation.
