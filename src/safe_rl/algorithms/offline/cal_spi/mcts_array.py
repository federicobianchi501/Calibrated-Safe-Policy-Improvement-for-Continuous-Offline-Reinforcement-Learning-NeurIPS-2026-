# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Array backend of the CSPI MCTS-DPW planner for the MuJoCo domains.

The search tree is stored in preallocated arrays and the simulation loop runs
in a compiled kernel (`mcts_array_kernel`). PUCT selection, progressive
widening, gate, frozen root, forced root visits, switch rule and RNG
consumption order are those of `OfflineMCTS_DPW_PPO_Numba_CSPI`; given the same
seed the two planners return the same actions. The calibrated bound B(u) is
evaluated as a compiled step-function lookup over the calibrator's bin edges.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numba import njit

from safe_rl.algorithms.offline.cal_spi.mcts_numba_scaled import (
    epistemic_from_deltas_scaled,
    usad_from_deltas_scaled,
)
from safe_rl.algorithms.offline.cal_spi.mcts_numba import (
    OfflineMCTS_DPW_PPO_Numba_CSPI,
)


# --------------------------------------------------------------------------- #
# Compiled calibrated bound
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def calibrated_bound(u, bin_edges, bin_bounds):
    """Piecewise-constant calibrated bound B(u).

    Same result as UncertaintyCalibrator.bound: right-side bin search, clamped
    to the first/last bin outside the edge range. Compiled without fastmath
    because the value is compared against eps.
    """
    nb = bin_bounds.shape[0]
    if u <= bin_edges[0]:
        return bin_bounds[0]
    if u >= bin_edges[bin_edges.shape[0] - 1]:
        return bin_bounds[nb - 1]
    # searchsorted(edges, u, side="right") - 1, clamped to [0, nb-1]
    lo = 0
    hi = bin_edges.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if bin_edges[mid] <= u:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx < 0:
        idx = 0
    elif idx > nb - 1:
        idx = nb - 1
    return bin_bounds[idx]


@njit(cache=True, fastmath=True)
def puct_select(n_actions, node_N, Nsa, Q, P, c_puct):
    """argmax_i Q[i] + c_puct * P[i] * sqrt(N) / max(1, Nsa[i])."""
    N = float(node_N)
    if N < 1.0:
        N = 1.0
    best_i = -1
    best = -1e30
    sqrtN = np.sqrt(N)
    for i in range(n_actions):
        nsa = float(Nsa[i])
        if nsa < 1.0:
            nsa = 1.0
        score = Q[i] + c_puct * P[i] * sqrtN / nsa
        if score > best:
            best = score
            best_i = i
    return best_i


# --------------------------------------------------------------------------- #
# Array-backed tree storage
# --------------------------------------------------------------------------- #
class _ArrayTree:
    """Preallocated node/edge storage for one decision.

    Nodes are indexed by integer. Edges of a node occupy a contiguous slice of
    the edge arrays, so selection and backup touch only flat buffers. Capacity
    is sized from the search budget: each simulation can create at most one node
    per level, and each node holds at most `cap` edges.
    """

    def __init__(self, max_nodes: int, max_edges: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((max_nodes, obs_dim), dtype=np.float32)
        self.depth = np.zeros(max_nodes, dtype=np.int32)
        self.N = np.zeros(max_nodes, dtype=np.int64)
        self.edge_start = np.zeros(max_nodes, dtype=np.int32)
        self.n_edges = np.zeros(max_nodes, dtype=np.int32)

        self.actions = np.zeros((max_edges, act_dim), dtype=np.float32)
        self.P = np.zeros(max_edges, dtype=np.float64)
        self.Q = np.zeros(max_edges, dtype=np.float64)
        self.Nsa = np.zeros(max_edges, dtype=np.int64)
        self.child = np.full(max_edges, -1, dtype=np.int32)

        self.n_nodes = 0
        self.n_alloc_edges = 0
        self._max_nodes = max_nodes
        self._max_edges = max_edges

    def new_node(self, obs: np.ndarray, depth: int, cap: int) -> int:
        """Allocate a node with room for `cap` edges; -1 if capacity is exhausted."""
        i = self.n_nodes
        if i >= self._max_nodes or self.n_alloc_edges + cap > self._max_edges:
            return -1
        self.obs[i, :] = obs
        self.depth[i] = depth
        self.N[i] = 0
        self.edge_start[i] = self.n_alloc_edges
        self.n_edges[i] = 0
        self.n_alloc_edges += cap
        self.n_nodes += 1
        return i

    def add_edge(self, node: int, action: np.ndarray, prior: float) -> int:
        """Append an edge with zero statistics to `node`; returns its edge index."""
        e = self.edge_start[node] + self.n_edges[node]
        self.actions[e, :] = action
        self.P[e] = prior
        self.Q[e] = 0.0
        self.Nsa[e] = 0
        self.child[e] = -1
        self.n_edges[node] += 1
        return e


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class OfflineMCTS_DPW_PPO_Array_CSPI(OfflineMCTS_DPW_PPO_Numba_CSPI):
    """CSPI planner with an array tree and a compiled search loop.

    Ensemble and PPO forward passes, the transition and the switch rule come
    from the numba CSPI planner. `delta_scale` (per-dimension scale of the
    predicted deltas) selects the scale-aware uncertainty, for ensembles whose
    calibrator was fitted on the scaled statistic.
    """

    def __init__(self, *args, delta_scale: Any = None, **kwargs):
        super().__init__(*args, **kwargs)
        cal = getattr(self, "calibrator", None)
        if cal is not None and getattr(cal, "bin_edges", None) is not None:
            self._bin_edges = np.asarray(cal.bin_edges, dtype=np.float64).ravel()
            self._bin_bounds = np.asarray(cal.bin_bounds, dtype=np.float64).ravel()
        else:
            self._bin_edges = None
            self._bin_bounds = None
        self._tree: Optional[_ArrayTree] = None
        self._pr_buf = np.empty(max(1, int(self.cfg.n_actions_per_node)),
                                dtype=np.float32)

        # Scale-aware gate: with `delta_scale`, deltas are divided by the
        # per-dimension scale before USAD / variance, matching a calibrator
        # fitted with `scale_aware_uncertainty`. Without it the kernel takes
        # the unscaled branch (no multiplication by ones).
        if delta_scale is None:
            self._inv_scale_k = np.ones(1, dtype=np.float32)
            self._use_scaled_k = False
        else:
            sc = np.asarray(delta_scale, dtype=np.float32).reshape(-1)
            d_obs = int(np.asarray(self._deltas).shape[1])
            if sc.shape[0] != d_obs:
                raise ValueError(
                    f"delta_scale has {sc.shape[0]} entries but the ensemble "
                    f"predicts {d_obs} dimensions"
                )
            self._inv_scale = (1.0 / np.maximum(sc, 1e-8)).astype(np.float32)
            self._inv_scale_k = self._inv_scale
            self._use_scaled_k = True
        if int(self.actions_dim()) > 128:
            raise ValueError(
                "the compiled control-cost sum reproduces NumPy's pairwise "
                "summation only up to 128 terms; above that the reward would "
                "differ from the reference in its last bits"
            )

    # ---------------------------------------------------------------- gate
    def _usad_from_cached_deltas(self) -> float:
        if not self._use_scaled_k:
            return super()._usad_from_cached_deltas()
        return float(usad_from_deltas_scaled(self._deltas, self._inv_scale))

    def _uncertainty_raw(self, obs, action) -> float:
        if not self._use_scaled_k:
            return super()._uncertainty_raw(obs, action)
        ut = str(self.cfg.uncertainty_type)
        if ut == "usad":
            return float(usad_from_deltas_scaled(
                self._compute_deltas(obs, action), self._inv_scale))
        if ut == "epistemic":
            return float(epistemic_from_deltas_scaled(
                self._compute_deltas(obs, action), self._inv_scale))
        return super()._uncertainty_raw(obs, action)

    def _bound_of(self, u: float) -> float:
        if self._bin_edges is None:
            return float("inf")
        return float(calibrated_bound(float(u), self._bin_edges, self._bin_bounds))

    def _is_safe_from_u(self, u: float) -> bool:
        """Same decision as the reference, via the compiled bound lookup."""
        if not self.use_gate:
            return True
        if str(self.cfg.gate_mode) == "uncertainty":
            return float(u) <= float(self.cfg.uncertainty_threshold)
        if self.calibrator is None:
            raise ValueError("gate_mode='calibrated' requires calibrator")
        if bool(self.calibrator.meta.get("low_data_regime", False)) and bool(
            self.cfg.conservative_on_low_data
        ):
            return False
        b = self._bound_of(float(u))
        if not np.isfinite(b):
            return False
        return b <= float(self.cfg.eps_threshold)

    # ---------------------------------------------------------------- tree
    def _cap_for(self, depth: int) -> int:
        """Maximum number of edges of a node at `depth`."""
        return int(self.cfg.n_actions_root) if depth == 0 else int(self.cfg.n_actions_per_node)

    def _alloc_tree(self) -> _ArrayTree:
        d = int(self.cfg.max_depth)
        sims = int(self.cfg.num_sims)
        cap_root = int(self.cfg.n_actions_root)
        cap_node = int(self.cfg.n_actions_per_node)
        # One new node per level per simulation is the worst case, plus the root.
        max_nodes = sims * d + 2
        max_edges = cap_root + (max_nodes - 1) * cap_node + cap_node
        return _ArrayTree(max_nodes, max_edges, int(self._ens_w.obs_dim
                          if hasattr(self._ens_w, "obs_dim") else self.model_env.cfg_obs_dim
                          if hasattr(self.model_env, "cfg_obs_dim") else self._obs_dim_fallback()),
                          int(self.actions_dim()))

    def _obs_dim_fallback(self) -> int:
        return int(np.asarray(self._deltas).shape[1])

    def actions_dim(self) -> int:
        """Action dimension, from the PPO actor output layer."""
        return int(self._ppo_w.W3.shape[0])

    def _expand_edges(self, tree: _ArrayTree, node: int, n_new: int) -> None:
        """Draw up to n_new actions from the PPO prior and append them as edges.

        Consumes the RNG like the reference: one `_sample_ppo` call, i.e. one
        (n, act_dim) standard-normal draw.
        """
        cap = self._cap_for(int(tree.depth[node]))
        room = cap - int(tree.n_edges[node])
        n_new = min(n_new, room)
        if n_new <= 0:
            return
        acts, priors = self._sample_ppo(tree.obs[node], n_new)
        for j in range(n_new):
            tree.add_edge(node, np.asarray(acts[j], np.float32), float(priors[j]))

    def run_compiled(self, tree: "_ArrayTree", model_indices, root_safe_ids) -> None:
        """Run all simulations of one decision in the compiled kernel.

        The normal deviates for action sampling come from the planner's own
        Generator, in the reference consumption order. A worst-case buffer is
        drawn up front; afterwards the generator is rewound and advanced by
        the number of deviates actually used, so its state after the decision
        equals that of the reference planner.
        """
        from safe_rl.algorithms.offline.cal_spi.mcts_array_kernel import run_simulations
        w = self._ens_w
        act_dim = int(self.actions_dim())
        # Upper bound on sampled actions: the smaller of the edge capacity and
        # 8 new actions per descent step.
        by_capacity = tree.obs.shape[0] * int(self.cfg.n_actions_per_node)
        by_descent = int(self.cfg.num_sims) * (int(self.cfg.max_depth) + 1) * 8
        max_normals = (min(by_capacity, by_descent) + int(self.cfg.n_actions_root)) * act_dim
        rng_state = self.rng.bit_generator.state
        normals = self.rng.standard_normal(size=max_normals).astype(np.float64)
        counters = np.array([tree.n_nodes, tree.n_alloc_edges, 0], dtype=np.int64)
        depth = int(self.cfg.max_depth)
        rsi = np.asarray(root_safe_ids, dtype=np.int64)
        low_data = bool(self.calibrator is not None
                        and self.calibrator.meta.get("low_data_regime", False)
                        and bool(self.cfg.conservative_on_low_data))
        run_simulations(
            tree.obs, tree.depth, tree.N, tree.edge_start, tree.n_edges,
            tree.actions, tree.P, tree.Q, tree.Nsa, tree.child, counters,
            w.W1, w.b1, w.W2, w.b2, w.W3, w.b3,
            w.input_mean, w.input_inv_std, w.has_norm,
            self._deltas, self._h1_e, self._h2_e, self._x_e,
            self._ppo_w.W1, self._ppo_w.b1, self._ppo_w.W2, self._ppo_w.b2,
            self._ppo_w.W3, self._ppo_w.b3, self._out_p, self._h1_p, self._h2_p,
            self._bin_edges, self._bin_bounds, float(self.cfg.eps_threshold),
            bool(self.use_gate), low_data, bool(self.cfg.gate_inside_tree),
            bool(self.cfg.suppress_unsafe_actions),
            int(self.cfg.num_sims), depth, float(self.cfg.gamma),
            float(self.cfg.dpw_k), float(self.cfg.dpw_alpha), float(self.cfg.c_puct),
            int(self.cfg.n_actions_root), int(self.cfg.n_actions_per_node),
            float(self.cfg.ppo_std_scale),
            bool(self._cspi_frozen_root), rsi, int(rsi.size),
            int(getattr(self, "_cspi_n_min_root_visits_eff", 0)),
            int(self._mc_vel_idx), int(self._mc_z_idx), int(self._mc_ang_idx),
            float(self._mc_ctrl), float(self._mc_h_r), float(self._mc_z_lo),
            float(self._mc_z_hi), float(self._mc_ang),
            float(self._mc_obs_clip), bool(self._mc_term_nan),
            self._inv_scale_k, bool(self._use_scaled_k), self._pr_buf,
            normals, np.asarray(model_indices, dtype=np.int64),
            np.empty(depth + 1, dtype=np.int64), np.empty(depth + 1, dtype=np.float64),
        )
        used = int(counters[2])
        self._last_normals_used, self._last_normals_drawn = used, max_normals
        if used > max_normals - act_dim:
            raise RuntimeError(
                f"action-sampling buffer exhausted ({used}/{max_normals}); the "
                f"kernel would have read past the drawn values"
            )
        self.rng.bit_generator.state = rng_state
        if used:
            self.rng.standard_normal(size=used)
        tree.n_nodes = int(counters[0])
        tree.n_alloc_edges = int(counters[1])

    def _simulate(self, tree: "_ArrayTree", root: int, mi: int) -> float:
        """One simulation in Python, iterative form of the recursive `_search`.

        The descent records (node, edge, reward) along the path; `_backup` then
        accumulates the discounted return backwards with incremental means.
        """
        path_nodes = np.empty(int(self.cfg.max_depth) + 1, dtype=np.int64)
        path_edges = np.empty(int(self.cfg.max_depth) + 1, dtype=np.int64)
        path_rew = np.zeros(int(self.cfg.max_depth) + 1, dtype=np.float64)
        n_path = 0
        node = root
        leaf_value = 0.0

        while True:
            tree.N[node] += 1
            depth = int(tree.depth[node])
            if depth >= int(self.cfg.max_depth):
                leaf_value = 0.0
                break

            is_root = (depth == 0)
            if not (is_root and self._cspi_frozen_root):
                max_children = int(float(self.cfg.dpw_k)
                                   * (float(tree.N[node]) ** float(self.cfg.dpw_alpha)))
                target = min(max_children, self._cap_for(depth))
                have = int(tree.n_edges[node])
                if have < target:
                    self._expand_edges(tree, node, min(8, target - have))

            n_e = int(tree.n_edges[node])
            if n_e == 0:
                leaf_value = self._leaf_value(tree.obs[node])
                break

            s = int(tree.edge_start[node])
            e_local = -1
            n_min_eff = getattr(self, "_cspi_n_min_root_visits_eff", 0)
            if (is_root and self._cspi_frozen_root and n_min_eff > 0
                    and self._cspi_root_safe_ids):
                for j in self._cspi_root_safe_ids:
                    if j < n_e and int(tree.Nsa[s + j]) < int(n_min_eff):
                        e_local = int(j)
                        break
            if e_local < 0:
                e_local = puct_select(n_e, tree.N[node], tree.Nsa[s:s + n_e],
                                      tree.Q[s:s + n_e], tree.P[s:s + n_e],
                                      float(self.cfg.c_puct))
            if e_local < 0:
                leaf_value = self._leaf_value(tree.obs[node])
                break

            e = s + e_local
            a = self._clip_action(tree.actions[e])
            tree.actions[e, :] = a

            self._compute_deltas(tree.obs[node], a)
            u = self._usad_from_cached_deltas()
            safe = self._is_safe_from_u(u)
            if self.use_gate and bool(self.cfg.gate_inside_tree) and (not safe):
                v = self._leaf_value(tree.obs[node])
                tree.Nsa[e] += 1
                tree.Q[e] += (v - tree.Q[e]) / float(tree.Nsa[e])
                if bool(self.cfg.suppress_unsafe_actions):
                    tree.P[e] = 0.0
                leaf_value = v
                # the blocked edge is not part of the backed-up path
                return self._backup(tree, path_nodes, path_edges, path_rew, n_path, v)

            obs_next, r, terminated, truncated = self._step_from_cached_deltas(
                tree.obs[node], a, int(mi))

            path_nodes[n_path] = node
            path_edges[n_path] = e
            path_rew[n_path] = float(r)
            n_path += 1

            if terminated or truncated:
                leaf_value = 0.0
                break

            child = int(tree.child[e])
            if child < 0:
                child = tree.new_node(obs_next, depth + 1, self._cap_for(depth + 1))
                if child < 0:                      # capacity exhausted
                    leaf_value = 0.0
                    break
                tree.child[e] = child
            node = child

        return self._backup(tree, path_nodes, path_edges, path_rew, n_path, leaf_value)

    def _backup(self, tree, path_nodes, path_edges, path_rew, n_path, leaf_value) -> float:
        """Discounted return along the recorded path, with incremental means."""
        G = float(leaf_value)
        gamma = float(self.cfg.gamma)
        for t in range(n_path - 1, -1, -1):
            G = float(path_rew[t]) + gamma * G
            e = int(path_edges[t])
            tree.Nsa[e] += 1
            tree.Q[e] += (G - tree.Q[e]) / float(tree.Nsa[e])
        return G


    # ------------------------------------------------------------------ act
    def act(self, obs: Any) -> np.ndarray:
        """Return the planned action, or the baseline action on fallback.

        Same steps as the numba CSPI `act`: early exits (low-data regime,
        gated-out baseline action) before any tree is built, frozen root,
        forced minimum visits over the safe root set, switch rule. The
        simulations run in the compiled kernel.
        """
        self._cspi_stats_n_calls += 1
        o = np.asarray(obs["observation"] if isinstance(obs, dict) else obs,
                       dtype=np.float32).reshape(-1)

        if (bool(self.cfg.conservative_on_low_data)
                and self.calibrator is not None
                and bool(getattr(self.calibrator, "meta", {}).get("low_data_regime", False))):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return self._clip_action(self._ppo_mean(o))

        a_b = self._cspi_baseline_action(o)
        baseline_safe, _, _ = self._is_safe(o, a_b)
        if not bool(baseline_safe):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            self._cspi_root_safe_ids = None
            return a_b

        act_dim = int(self.actions_dim())
        depth = int(self.cfg.max_depth)
        sims = int(self.cfg.num_sims)
        cap_r, cap_n = int(self.cfg.n_actions_root), int(self.cfg.n_actions_per_node)
        max_nodes = sims * depth + 2
        tree = _ArrayTree(max_nodes, cap_r + max_nodes * cap_n, o.shape[0], act_dim)
        root = tree.new_node(o, 0, cap_r)
        tree.add_edge(root, np.asarray(a_b, np.float32), 1.0)

        if self._cspi_frozen_root and cap_r > 1:
            self._expand_edges(tree, root, cap_r - 1)

        n_e = int(tree.n_edges[root])
        s0 = int(tree.edge_start[root])
        root_safe_ids = [i for i in range(n_e)
                         if self._is_safe(o, tree.actions[s0 + i])[0]]
        self._cspi_root_safe_ids = root_safe_ids
        self._cspi_stats_root_total_safe += len(root_safe_ids)
        self._cspi_n_min_root_visits_eff = (
            max(1, min(int(self._cspi_n_min_root_visits), sims // len(root_safe_ids)))
            if root_safe_ids else 0)

        model_indices = self.rng.integers(0, int(self._ens_w.K), size=sims)
        self.run_compiled(tree, model_indices, root_safe_ids)

        self.last_root_coverage = float(len(root_safe_ids)) / float(max(1, n_e))
        if not root_safe_ids:
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return self._clip_action(self._ppo_mean(o))

        tol = 1e-9
        q_best = max(tree.Q[s0 + j] for j in root_safe_ids)
        cands = [j for j in root_safe_ids if (q_best - tree.Q[s0 + j]) <= tol]
        best_i = max(cands, key=lambda j: tree.Nsa[s0 + j])
        a_star = np.asarray(tree.actions[s0 + best_i], dtype=np.float32)

        q_b = float(tree.Q[s0 + 0])
        eta_op = float(self._cspi_margin_mult) * float(self._cspi_e_plan_scale)
        if float(q_best) - q_b >= eta_op:
            self.last_root_fallback = False
            self._cspi_stats_n_switch += 1
            return a_star
        self.last_root_fallback = True
        self._cspi_stats_n_fallback_margin += 1
        return a_b



# Alias: the scale-aware variant is the same class with `delta_scale` set.
OfflineMCTS_DPW_PPO_Array_CSPI_Scaled = OfflineMCTS_DPW_PPO_Array_CSPI
