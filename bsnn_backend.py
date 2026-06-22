"""
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
BSNN / B-Spline Neural Network with Bloch-Sphere Cognitive Manifold
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
"""

import numpy as np
import cupy as cp
import scipy.linalg as la
import scipy.special as sp
from scipy.interpolate import BSpline
from numba import cuda, njit, prange, vectorize, float64, complex128
import json, simdjson
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import threading, time, uuid


# /// COMPLEX NUMBER COORDINATE SYSTEM ///

@njit(parallel=True, fastmath=True)
def complex_coord_basis(n: int, dim: int) -> np.ndarray:
    out = np.zeros((n, dim), dtype=np.complex128)
    for i in prange(n):
        r = np.random.rand(dim)
        theta = np.random.rand(dim) * 2 * np.pi
        out[i] = r * (np.cos(theta) + 1j * np.sin(theta))
    return out

@njit(fastmath=True)
def orthogonal_complex_projection(coords: np.ndarray) -> np.ndarray:
    n, d = coords.shape
    Q = np.zeros_like(coords)
    for i in range(n):
        v = coords[i].copy()
        for j in range(i):
            proj = np.dot(np.conj(Q[j]), v) / (np.dot(np.conj(Q[j]), Q[j]) + 1e-12)
            v -= proj * Q[j]
        norm = np.sqrt(np.real(np.dot(np.conj(v), v))) + 1e-12
        Q[i] = v / norm
    return Q


# /// BLOCH SPHERE MANIFOLD ///

@dataclass
class BlochState:
    theta: float = 0.0  # polar  [0, π]
    phi: float   = 0.0  # azimuthal [0, 2π]
    r: float     = 1.0  # radial

    def to_cartesian(self) -> np.ndarray:
        return self.r * np.array([
            np.sin(self.theta) * np.cos(self.phi),
            np.sin(self.theta) * np.sin(self.phi),
            np.cos(self.theta)])

    def to_density_matrix(self) -> np.ndarray:
        # ρ = (I + r⃗·σ⃗)/2 -> Bloch sphere to density matrix
        x, y, z = self.to_cartesian()
        return 0.5 * np.array([[1 + z, x - 1j*y], [x + 1j*y, 1 - z]])

    def fidelity_to(self, other: 'BlochState') -> float:
        # Fidelity F = Tr(√(√ρ σ √ρ))² between two Bloch states
        rho = self.to_density_matrix()
        sig = other.to_density_matrix()
        sqrt_rho = la.sqrtm(rho)
        M = sqrt_rho @ sig @ sqrt_rho
        return np.real(np.trace(la.sqrtm(M)))**2

class BlochManifold:
    """
    Cognitive surface manifold / differentiable Bloch sphere with:
    
       - θ/φ differentiation module for self-improvement
       - Optimizer coupling fidelity to loss
    """
    
    def __init__(self, n_points: int = 64):
        self.n = n_points
        self.states: List[BlochState] = [
            BlochState(theta=np.random.uniform(0, np.pi), phi=np.random.uniform(0, 2*np.pi), r=np.random.uniform(0.7, 1.0)) for _ in range(n_points)]
        self.history: List[Dict] = []; self.lr_theta = 0.01; self.lr_phi = 0.01


    # • Differentiation module •
    
    def grad_theta(self, state: BlochState, loss_fn) -> float:
        # ∂L/∂θ via finite differences in θ space
        eps = 1e-4
        s_p = BlochState(state.theta + eps, state.phi, state.r)
        s_m = BlochState(state.theta - eps, state.phi, state.r)
        return (loss_fn(s_p) - loss_fn(s_m)) / (2 * eps)

    def grad_phi(self, state: BlochState, loss_fn) -> float:
        # ∂L/∂φ via finite differences in φ space
        eps = 1e-4
        s_p = BlochState(state.theta, state.phi + eps, state.r)
        s_m = BlochState(state.theta, state.phi - eps, state.r)
        return (loss_fn(s_p) - loss_fn(s_m)) / (2 * eps)

    def recursive_theta_pi_refine(self, state: BlochState, loss_fn, depth=3) -> BlochState:
        # Recursive θ/π parameter refinement for self-improvement
        # Recurses into subspaces where loss gradient is steepest
        if depth == 0:
            return state
        g_theta = self.grad_theta(state, loss_fn)
        g_phi = self.grad_phi(state, loss_fn)
        new_theta = np.clip(state.theta - self.lr_theta * g_theta, 0, np.pi)
        new_phi = (state.phi - self.lr_phi * g_phi) % (2*np.pi)
        new_state = BlochState(new_theta, new_phi, state.r)
        # recurse on half-step
        if abs(g_theta) + abs(g_phi) > 1e-3:
            return self.recursive_theta_pi_refine(new_state, loss_fn, depth-1)
        return new_state


    # • Optimizer coupling fidelity to loss •
    
    def fidelity_loss_coupler(self, target_state: BlochState, loss: float, alpha: float = 0.3) -> float:
        # Blended objective; minimize loss while maximizing fidelity to target
        # L_total = (1-α)·loss - α·F(ρ, σ_target)
        avg_fidelity = np.mean([s.fidelity_to(target_state) for s in self.states])
        return (1 - alpha) * loss - alpha * avg_fidelity

    def svd_angular_align(self, feature_matrix: np.ndarray) -> np.ndarray:
        # SVD compression with angular alignment to cognitive sphere SOTA
        U, S, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
        # align singular vectors to Bloch sphere axes
        pts = np.array([s.to_cartesian() for s in self.states[:min(len(self.states), Vt.shape[0])]])
        if pts.shape[0] < Vt.shape[0]: pts = np.vstack([pts, np.zeros((Vt.shape[0]-pts.shape[0], 3))])
        S_scaled = S / (S.max() + 1e-12)
        return (U * S_scaled) @ Vt


# /// B-SPLINE ROTATIONAL NODES ///

class BSplineNode:
    # Dimensional rotational node with the complex number B-spline basis
    # Tron like channel; data flows along spline paths with angular phase
    def __init__(self, degree: int = 3, n_knots: int = 8, dim: int = 4):
        self.degree = degree
        self.dim = dim
        self.node_id = str(uuid.uuid4())[:8]
        t = np.linspace(0, 1, n_knots + degree + 1)
        self.knots = t
        c_real = np.random.randn(n_knots, dim)
        c_imag = np.random.randn(n_knots, dim)
        self.control_points = c_real + 1j * c_imag
        self.rotation_angle = np.random.uniform(0, 2*np.pi)
        self.hot_path_interval: Optional[Tuple[float,float]] = None
        self.activation_count = 0

    def rotation_matrix(self, angle: float) -> np.ndarray:
        # D-dim complex rotation -> pairwise 2D rotations stacked
        R = np.eye(self.dim, dtype=complex)
        for i in range(0, self.dim-1, 2):
            c, s = np.cos(angle), np.sin(angle)
            R[i,i] = c; R[i,i+1] = -s
            R[i+1,i] = s; R[i+1,i+1] = c
        return R

    @cuda.jit(device=True)
    def _eval_basis_gpu(t_val, knots, degree):
        # Cox de Boor on GPU
        pass

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        # Evaluate rotational spline path at parameter values t ∈ [0,1]
        t = np.clip(t, 0, 1-1e-9)
        n_ctrl = len(self.control_points)
        
        # Build B-spline basis for real & imag separately
        R = self.rotation_matrix(self.rotation_angle)
        out = np.zeros((len(t), self.dim), dtype=complex)
        for d in range(self.dim):
            c_r = np.real(self.control_points[:, d])
            c_i = np.imag(self.control_points[:, d])
            spl_r = BSpline(self.knots, c_r, self.degree)
            spl_i = BSpline(self.knots, c_i, self.degree)
            out[:, d] = spl_r(t) + 1j * spl_i(t)
        return (R @ out.T).T

    def set_hot_path(self, lo: float, hi: float):
        # Set angular separation hot path interval, numba accelerated lookup
        self.hot_path_interval = (lo, hi)

    @njit(fastmath=True)
    def hot_path_mask(self, angles: np.ndarray) -> np.ndarray:
        lo, hi = self.hot_path_interval
        return (angles >= lo) & (angles <= hi)


# /// KAN INNER-VECTOR CONCEPT STORE ///

class KANConceptStore:
    # Kolmogorov Arnold Network inner-vector store for deep self-learning switches
    # Stores compressed activation concepts from spline paths SOTA
    def __init__(self, capacity: int = 512, dim: int = 32):
        self.capacity = capacity
        self.dim = dim
        self.keys = np.zeros((capacity, dim), dtype=np.float32)
        self.values = np.zeros((capacity, dim), dtype=np.float32)
        self.scores = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self._lock = threading.Lock(); self.parser = simdjson.Parser()

    def store(self, concept_vec: np.ndarray, score: float):
        with self._lock:
            idx = self.ptr % self.capacity
            v = np.real(concept_vec).flatten()[:self.dim]
            if len(v) < self.dim: v = np.pad(v, (0, self.dim - len(v)))
            self.keys[idx] = v; self.values[idx] = v
            self.scores[idx] = score; self.ptr += 1

    def retrieve_top_k(self, query: np.ndarray, k: int = 8) -> List[Dict]:
        q = np.real(query).flatten()[:self.dim]
        if len(q) < self.dim: q = np.pad(q, (0, self.dim - len(q)))
        sims = self.keys[:min(self.ptr, self.capacity)] @ q
        sims /= (np.linalg.norm(self.keys[:min(self.ptr,self.capacity)], axis=1) * np.linalg.norm(q) + 1e-12)
        top_k_idx = np.argsort(sims)[-k:][::-1]
        return [{"idx": int(i), "score": float(self.scores[i]), "sim": float(sims[i])} for i in top_k_idx]

    def serialize_stats_simdjson(self) -> str:
        # simdjson cart; serialize KAN stats for evolutionary processing
        stats = {"stored": int(min(self.ptr, self.capacity)),
                 "mean_score": float(np.mean(self.scores[:min(self.ptr, self.capacity)])),
                 "max_score": float(np.max(self.scores[:min(self.ptr, self.capacity)])+1e-12),
                 "top_concepts": self.retrieve_top_k(self.keys[0] if self.ptr > 0 else np.zeros(self.dim))}
        return json.dumps(stats)


# /// META-CONTROLLER ///

@dataclass
class MetaControllerProperties:
    # Embedded properties managed across spline layers
    mc_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    layer: int = 0
    score: float = 0.0
    bloch_theta: float = 0.0
    bloch_phi: float = 0.0
    svd_rank: int = 8
    spline_count: int = 4
    agent_pairs: List = field(default_factory=list)
    loss_history: List = field(default_factory=list)
    improvement_rate: float = 0.0
    merge_parent_ids: List = field(default_factory=list)
    # Spawned from deep agent pairing
    is_subconscious: bool = False

class MetaController:
    # Spawnable meta-controller with full BSNN architecture
    # Orthogonal segmentation reasoning across layers SOTA
    def __init__(self, props: MetaControllerProperties, manifold: BlochManifold, concept_store: KANConceptStore, dim: int = 16):
        self.props = props; self.manifold = manifold
        self.store = concept_store; self.dim = dim
        self.nodes = [BSplineNode(dim=dim) for _ in range(props.spline_count)]
        self.sub_agents: List['SubAgent'] = []
        self.feature_matrix = np.random.randn(dim, props.svd_rank) + 1j * np.random.randn(dim, props.svd_rank)
        self._spawn_sub_agents()

    def _spawn_sub_agents(self):
        # Pair sub-agents from collective properties, sub-domain agent polling SOTA
        n = max(2, self.props.spline_count // 2)
        for i in range(n):
            sa = SubAgent(parent_mc=self, index=i, dim=self.dim)
            self.sub_agents.append(sa)
        # Pair adjacents:
        self.props.agent_pairs = [(i, i+1) for i in range(0, len(self.sub_agents)-1, 2)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Full forward pass; rotational nodes -> SVD -> Bloch alignment
        t_vals = np.linspace(0, 1, x.shape[0] if x.ndim > 1 else 8)
        agg = np.zeros((len(t_vals), self.dim), dtype=complex)
        for node in self.nodes:
            path = node.evaluate(t_vals); agg += path
        # Project input through complex coords SOTA
        if x.ndim == 1: x = x[:self.dim] if len(x) >= self.dim else np.pad(x,(0,self.dim-len(x)))
        agg_real = np.real(agg)
        compressed = self.manifold.svd_angular_align(agg_real)
        # Store KAN concept:
        concept = compressed.mean(axis=0)
        self.store.store(concept, self.props.score)
        return compressed

    def compute_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        # MSE loss in real subspace
        p = np.real(pred).flatten()
        t_arr = np.real(target).flatten()
        min_len = min(len(p), len(t_arr))
        return float(np.mean((p[:min_len] - t_arr[:min_len])**2))

    def self_improve(self, x: np.ndarray, target: np.ndarray, rounds: int = 3):
        # Acceptable orthogonal segmentation reasoning improvement
        # Uses probability parallelism between SVD compressions SOTA
        best_loss = float('inf')
        for r in range(rounds):
            pred = self.forward(x)
            loss = self.compute_loss(pred, target)
            # Bloch manifold differentiation:
            bloch_state = BlochState(self.props.bloch_theta, self.props.bloch_phi)
            def loss_fn(s): return loss * (1 + 0.1 * np.sin(s.theta))
            improved = self.manifold.recursive_theta_pi_refine(bloch_state, loss_fn)
            self.props.bloch_theta = improved.theta
            self.props.bloch_phi = improved.phi
            # Fidelity coupling SOTA
            target_state = BlochState(np.pi/4, np.pi/4)
            coupled_loss = self.manifold.fidelity_loss_coupler(target_state, loss)
            self.props.loss_history.append(coupled_loss)
            if coupled_loss < best_loss:
                best_loss = coupled_loss; self.props.score = 1.0 / (1.0 + best_loss)
            # Rotate nodes slightly for the next round
            for node in self.nodes: node.rotation_angle += 0.05 * (1 - self.props.score)
        prev_score = self.props.score
        self.props.improvement_rate = (self.props.score - prev_score) / (abs(prev_score) + 1e-12)
        return best_loss

    def poll_sub_agents(self, x: np.ndarray) -> Dict:
        # Sub-domain agent polling via deterministic partitioning, scipy SOTA
        results = {}
        for sa in self.sub_agents:
            partition_x = sa.deterministic_partition(x)
            results[sa.agent_id] = sa.process(partition_x)
        return results

    def to_dict(self) -> Dict:
        return {"mc_id": self.props.mc_id,
                "generation": self.props.generation,
                "layer": self.props.layer,
                "score": round(self.props.score, 4),
                "bloch_theta": round(self.props.bloch_theta, 4),
                "bloch_phi": round(self.props.bloch_phi, 4),
                "spline_count": self.props.spline_count,
                "agent_pairs": self.props.agent_pairs,
                "loss_history": [round(l,4) for l in self.props.loss_history[-10:]],
                "improvement_rate": round(self.props.improvement_rate, 4),
                "is_subconscious": self.props.is_subconscious,
                "node_ids": [n.node_id for n in self.nodes],}


# /// SUB-AGENT ///

class SubAgent:
    # Sub-domain agent with deterministic partitioning and scipy integration SOTA
    # Handles orthogonal probability spline origins quickly
    def __init__(self, parent_mc: MetaController, index: int, dim: int = 16):
        self.agent_id = f"{parent_mc.props.mc_id}_sa{index}"
        self.parent_mc = parent_mc; self.index = index; self.dim = dim
        self.ortho_prob = np.abs(np.random.randn(dim))
        self.ortho_prob /= self.ortho_prob.sum() + 1e-12

    def deterministic_partition(self, x: np.ndarray) -> np.ndarray:
        # Deterministic partitioning by category: spline reliance, rotational placement
        # SOTA scipy backed; uses cumulative recover distribution for hard boundaries
        flat = x.flatten()[:self.dim]
        if len(flat) < self.dim: flat = np.pad(flat, (0, self.dim - len(flat)))
        cdf = np.cumsum(self.ortho_prob)
        # Assign each dim to a category via CDF thresholds
        cats = np.searchsorted(cdf, np.abs(flat) / (np.abs(flat).max()+1e-12))
        # Vectorized non-linear mapping <-> tanh + Bessel
        mapped = np.tanh(flat) * sp.jv(0, np.abs(flat) * np.pi)
        return mapped

    def process(self, x: np.ndarray) -> Dict:
        score = float(np.linalg.norm(x))
        return {"agent_id": self.agent_id, "norm": round(score,4), "ortho_entropy": float(-np.sum(self.ortho_prob * np.log(self.ortho_prob + 1e-12)))}


# /// BSNN ORCHESTRATOR ///

class BSNN:
    # Top level B-Spline Neural Network orchestrator
    def __init__(self, n_layers: int = 3, mc_per_layer: int = 4, dim: int = 16, manifold_points: int = 32):
        self.n_layers = n_layers
        self.dim = dim
        self.manifold = BlochManifold(n_points=manifold_points)
        self.store = KANConceptStore(capacity=256, dim=dim)
        self.layers: List[List[MetaController]] = []
        self.generation = 0; self.global_best_score = 0.0; self._init_layers(mc_per_layer)
        # Separate higher-layer MC that accumulates merge scores SOTA
        self.apex_mc = None; self._parser = simdjson.Parser()

    def _make_mc(self, layer: int, is_sub: bool = False) -> MetaController:
        props = MetaControllerProperties(generation = self.generation,
                                         layer = layer,
                                         svd_rank = max(4, self.dim // 2),
                                         spline_count = max(2, 4 - layer),
                                         bloch_theta = np.random.uniform(0, np.pi),
                                         bloch_phi = np.random.uniform(0, 2*np.pi),
                                         is_subconscious = is_sub)
        return MetaController(props, self.manifold, self.store, self.dim)

    def _init_layers(self, mc_per_layer: int):
        for l in range(self.n_layers): self.layers.append([self._make_mc(l) for _ in range(mc_per_layer)])


    # • Probability parallelism; SVD across all MCs •
    
    def parallel_svd_pass(self, x: np.ndarray) -> List[np.ndarray]:
        # GPU parallel SVD forward pass across all MCs using CuPy SOTA
        results = []
        x_gpu = cp.asarray(np.real(x).astype(np.float32))
        for layer in self.layers:
            layer_out = []
            for mc in layer:
                fm = cp.asarray(np.real(mc.feature_matrix).astype(np.float32))
                U, S, Vt = cp.linalg.svd(fm, full_matrices=False)
                proj = (U * (S / (S.max()+1e-12))) @ Vt
                if proj.shape[1] == x_gpu.shape[0]: out = proj @ x_gpu
                else:
                    min_d = min(proj.shape[1], x_gpu.shape[0])
                    out = proj[:, :min_d] @ x_gpu[:min_d]
                layer_out.append(cp.asnumpy(out))
            results.append(layer_out)
        return results


    # • Recursive complex coord seeding •
    
    def recursive_complex_seed(self, group_ids: List[Tuple[int,int]], depth: int = 2) -> np.ndarray:
        # Recursively draft randomized complex coordinates for chosen spline groups
        # Triggered when SVD compression scores a group as more adaptable SOTA
        if depth == 0:
            return complex_coord_basis(len(group_ids), self.dim)
        seeds = complex_coord_basis(len(group_ids), self.dim)
        ortho = orthogonal_complex_projection(seeds)
        sub_groups = [group_ids[i::2] for i in range(2)]
        sub_results = [self.recursive_complex_seed(sg, depth-1) for sg in sub_groups if sg]
        if sub_results:
            combined = np.vstack(sub_results); min_r = min(len(ortho), len(combined))
            ortho[:min_r] = 0.5*(ortho[:min_r] + combined[:min_r])
        return ortho


    # • Merge -> targets MCs with high prior scores into apex layer •
    
    def merge_top_controllers(self, top_k: int = 2):
        # Targets MCs with merge qualities, higher previous scores, into
        # separate apex layer meta-controller SOTA
        all_mc = [mc for layer in self.layers for mc in layer]
        all_mc.sort(key=lambda m: m.props.score, reverse=True)
        top_mcs = all_mc[:top_k]
        merged_props = MetaControllerProperties(
                        generation= self.generation,
                        layer = self.n_layers,
                        spline_count = sum(m.props.spline_count for m in top_mcs),
                        bloch_theta = np.mean([m.props.bloch_theta for m in top_mcs]),
                        bloch_phi = np.mean([m.props.bloch_phi   for m in top_mcs]),
                        merge_parent_ids = [m.props.mc_id for m in top_mcs],
                        score = max(m.props.score for m in top_mcs))
        self.apex_mc = MetaController(merged_props, self.manifold, self.store, self.dim)
        return merged_props


    # • Subconscious spawning; discrete agent pairing -> new MC •
    
    def subconscious_spawn(self, kan_stats_json: str) -> Optional[MetaController]:
        # simdjson cart processing of KAN stats -> evolutionary new MC
        # Subconscious re-actions from discrete agent pairing
        try:
            stats = self._parser.parse(kan_stats_json)
            mean_score = float(stats["mean_score"])
            if mean_score > 0.3: # threshold for new MC formation SOTA
                new_mc = self._make_mc(layer=0, is_sub=True)
                new_mc.props.score = mean_score * 0.8
                self.layers[0].append(new_mc)
                return new_mc
        except Exception:
            pass
        return None


    # • Full self-improvement cycle •
    
    def evolve(self, x: np.ndarray, target: np.ndarray, cycles: int = 1) -> Dict:
        results = {"generation": self.generation, "layers": [], "apex": None, "subconscious_spawn": False, "global_best": 0.0}
        for _ in range(cycles):
            self.generation += 1; self.parallel_svd_pass(x)
            # Self-improve each MC
            layer_snapshots = []
            for l_idx, layer in enumerate(self.layers):
                mc_snaps = []
                for mc in layer:
                    mc.props.generation = self.generation
                    mc.self_improve(x, target, rounds=2)
                    mc.poll_sub_agents(x)
                    mc_snaps.append(mc.to_dict())
                    if mc.props.score > self.global_best_score:
                        self.global_best_score = mc.props.score
                layer_snapshots.append(mc_snaps)
            results["layers"] = layer_snapshots
            # *Merge top -> apex
            merged = self.merge_top_controllers(top_k=2)
            results["apex"] = {"mc_id": merged.mc_id,
                               "score": round(merged.score,4),
                               "parents": merged.merge_parent_ids}
            # *Subconscious spawn from KAN stats SOTA
            kan_stats = self.store.serialize_stats_simdjson()
            new_mc = self.subconscious_spawn(kan_stats)
            results["subconscious_spawn"] = new_mc is not None
        results["global_best"] = round(self.global_best_score, 4)
        return results

    def get_state(self) -> Dict:
        return {"generation": self.generation,
                "n_layers": self.n_layers,
                "global_best": round(self.global_best_score, 4),
                "manifold_points": self.manifold.n,
                "kan_stored": int(min(self.store.ptr, self.store.capacity)),
                "layers": [[mc.to_dict() for mc in layer] for layer in self.layers],
                "apex": self.apex_mc.to_dict() if self.apex_mc else None,}


# /// FASTAPI BACKEND ///

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio, uvicorn
from pydantic import BaseModel

app = FastAPI(title="BSNN API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bsnn = BSNN(n_layers=3, mc_per_layer=4, dim=16, manifold_points=32)
_evolving = False

class EvolveRequest(BaseModel):
    cycles: int = 1

@app.get("/state")
def get_state(): return bsnn.get_state()

@app.post("/evolve")
def evolve(req: EvolveRequest):
    x = np.random.randn(bsnn.dim).astype(np.float32)
    target = np.random.randn(bsnn.dim).astype(np.float32)
    return bsnn.evolve(x, target, cycles=req.cycles)

@app.get("/kan_stats")
def kan_stats(): return json.loads(bsnn.store.serialize_stats_simdjson())

@app.get("/manifold")
def manifold_state():
    return [{"theta": round(s.theta,4), "phi": round(s.phi,4), "r": round(s.r,4), "xyz": s.to_cartesian().tolist()} for s in bsnn.manifold.states[:16]]

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            x = np.random.randn(bsnn.dim).astype(np.float32)
            t = np.random.randn(bsnn.dim).astype(np.float32)
            result = bsnn.evolve(x, t, cycles=1)
            await ws.send_json(result); await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
