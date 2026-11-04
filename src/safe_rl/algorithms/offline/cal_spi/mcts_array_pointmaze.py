# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Compiled MCTS-DPW planner for PointMaze.

Differences from the locomotion kernel in `mcts_array`:
  * dict observations; the PPO input is [achieved_goal, desired_goal,
    observation], with achieved_goal equal to the leading components of
    the observation and the goal constant along an episode;
  * goal-based reward `-step_penalty - dist_penalty * d - living_cost`, plus
    `success_bonus` on reaching `goal_thresh`, which terminates the episode;
  * leaf value from the PPO critic on gate rejection and at nodes without
    actions (zero at maximum depth);
  * SPIBB variant without frozen root, minimum root visits or switch rule.

The bound lookup, gate and PPO-prior sampling are shared with `mcts_array_kernel`.
The kernel matches the reference planner in tree, widening, reward, termination,
leaf value and gate; the PPO forward is computed in float32, so actions agree
with the torch reference up to float32 rounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numba import njit

from safe_rl.offline.ensemble_numba import (
    _linear,
    _linear_tanh,
    ensemble_forward_all,
    extract_ensemble_weights,
    extract_ppo_actor_weights,
    ppo_actor_mean,
    usad_from_deltas,
)
from safe_rl.algorithms.offline.cal_spi.mcts_parametric import (
    OfflineMCTS_DPW_PPO_SPIBB_Parametric,
)
from safe_rl.algorithms.offline.cal_spi.mcts_array import _ArrayTree
from safe_rl.algorithms.offline.cal_spi.mcts_array_kernel import (
    _is_safe,
    _puct,
    _sample_ppo_block,
)


# --------------------------------------------------------------------------- #
# Behaviour-policy critic
# --------------------------------------------------------------------------- #
@dataclass
class PPOCriticWeights:
    """Float32 weights of the SB3 value network (two tanh layers, scalar head)."""
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    in_dim: int


def extract_ppo_critic_weights(ppo: Any) -> PPOCriticWeights:
    """Read the value network of an SB3 actor-critic policy.

    Layout Linear, Tanh, Linear, Tanh, Linear with one output. Raises
    ValueError if the head does not have exactly one output.
    """
    vnet = ppo.policy.mlp_extractor.value_net
    lin1, lin2 = vnet[0], vnet[2]
    head = ppo.policy.value_net
    if head.out_features != 1:
        raise ValueError(f"critic head has {head.out_features} outputs, expected 1")
    return PPOCriticWeights(
        W1=lin1.weight.detach().cpu().numpy().astype(np.float32),
        b1=lin1.bias.detach().cpu().numpy().astype(np.float32),
        W2=lin2.weight.detach().cpu().numpy().astype(np.float32),
        b2=lin2.bias.detach().cpu().numpy().astype(np.float32),
        W3=head.weight.detach().cpu().numpy().astype(np.float32),
        b3=head.bias.detach().cpu().numpy().astype(np.float32),
        in_dim=int(lin1.in_features),
    )


@njit(cache=True, fastmath=True, inline="always")
def critic_value(feat, W1, b1, W2, b2, W3, b3, out, h1, h2):
    """Critic value V(s) from the packed features (unclipped)."""
    _linear_tanh(feat, W1, b1, h1)
    _linear_tanh(h1, W2, b2, h2)
    _linear(h2, W3, b3, out)
    return out[0]


@njit(cache=True, fastmath=True, inline="always")
def pack_features(obs_vec, dgoal, feat):
    """Write [achieved_goal, desired_goal, observation] into `feat`.

    Key order of the SB3 extractor. `achieved_goal` is taken as the leading
    components of the observation (the position), as in `ModelPointMazeEnv`.
    """
    g = dgoal.shape[0]
    for i in range(g):
        feat[i] = obs_vec[i]
        feat[g + i] = dgoal[i]
    for i in range(obs_vec.shape[0]):
        feat[2 * g + i] = obs_vec[i]


# --------------------------------------------------------------------------- #
# Compiled search loop
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=True)
def run_simulations_pointmaze(
    # tree storage
    t_obs, t_depth, t_N, t_edge_start, t_n_edges,
    e_actions, e_P, e_Q, e_Nsa, e_child, counters,
    # ensemble
    W1, b1, W2, b2, W3, b3, in_mean, in_inv_std, has_norm,
    deltas, h1_e, h2_e, x_e,
    # behaviour actor
    pW1, pb1, pW2, pb2, pW3, pb3, p_out, p_h1, p_h2,
    # behaviour critic
    vW1, vb1, vW2, vb2, vW3, vb3, v_out, v_h1, v_h2, use_critic,
    # packed-feature scratch and the constant goal
    feat, dgoal,
    # gate
    bin_edges, bin_bounds, eps, use_gate, low_data, gate_inside_tree,
    suppress_unsafe,
    # planner config
    num_sims, max_depth, gamma, dpw_k, dpw_alpha, c_puct,
    cap_root, cap_node, ppo_std,
    # goal-based model env
    goal_thresh, step_penalty, dist_penalty, success_bonus, living_cost,
    continuing_task, obs_clip, term_nan,
    # scratch for the per-block priors
    pr_buf,
    # randomness, pre-drawn outside
    normals, model_indices,
    # scratch
    path_edges, path_rew,
):
    """Run `num_sims` simulations from root node 0; returns the deviates used.

    Iterative form of the reference `_search`: widen, select, gate, step,
    back up. With `use_critic`, the leaf value is the critic at a node with no
    selectable action and on gate rejection; it is zero at maximum depth, on
    termination and when tree capacity is exhausted.
    """
    obs_dim = t_obs.shape[1]
    act_dim = e_actions.shape[1]

    for sim in range(num_sims):
        mi = model_indices[sim]
        node = 0
        n_path = 0
        leaf_value = 0.0

        for depth in range(max_depth + 1):
            t_N[node] += 1
            if depth >= max_depth:
                leaf_value = 0.0
                break

            # ---- progressive widening -----------------------------------
            cap = cap_root if depth == 0 else cap_node
            target = int(dpw_k * (float(t_N[node]) ** dpw_alpha))
            if target > cap:
                target = cap
            have = t_n_edges[node]
            if have < target:
                add_n = target - have
                if add_n > 8:
                    add_n = 8
                room = cap - have
                if add_n > room:
                    add_n = room
                if add_n > 0:
                    pack_features(t_obs[node], dgoal, feat)
                    ppo_actor_mean(feat, pW1, pb1, pW2, pb2, pW3, pb3,
                                   p_out, p_h1, p_h2)
                    s0 = t_edge_start[node] + have
                    counters[2] = _sample_ppo_block(
                        p_out, normals, counters[2], add_n, act_dim, ppo_std,
                        e_actions, e_P, e_Q, e_Nsa, e_child, s0, pr_buf)
                    t_n_edges[node] = have + add_n

            n_e = t_n_edges[node]
            if n_e == 0:
                if use_critic:
                    pack_features(t_obs[node], dgoal, feat)
                    leaf_value = critic_value(feat, vW1, vb1, vW2, vb2, vW3, vb3,
                                              v_out, v_h1, v_h2)
                else:
                    leaf_value = 0.0
                break

            s = t_edge_start[node]
            e_local = _puct(n_e, t_N[node], e_Nsa, e_Q, e_P, s, c_puct)
            if e_local < 0:
                if use_critic:
                    pack_features(t_obs[node], dgoal, feat)
                    leaf_value = critic_value(feat, vW1, vb1, vW2, vb2, vW3, vb3,
                                              v_out, v_h1, v_h2)
                else:
                    leaf_value = 0.0
                break

            e = s + e_local
            for d in range(act_dim):
                if e_actions[e, d] < -1.0:
                    e_actions[e, d] = -1.0
                elif e_actions[e, d] > 1.0:
                    e_actions[e, d] = 1.0

            # ---- gate ----------------------------------------------------
            ensemble_forward_all(t_obs[node], e_actions[e], W1, b1, W2, b2, W3, b3,
                                 in_mean, in_inv_std, has_norm, deltas, h1_e, h2_e, x_e)
            u = usad_from_deltas(deltas)
            if use_gate and gate_inside_tree and (not _is_safe(
                    u, bin_edges, bin_bounds, eps, use_gate, low_data)):
                if use_critic:
                    pack_features(t_obs[node], dgoal, feat)
                    v = critic_value(feat, vW1, vb1, vW2, vb2, vW3, vb3,
                                     v_out, v_h1, v_h2)
                else:
                    v = 0.0
                e_Nsa[e] += 1
                e_Q[e] += (v - e_Q[e]) / float(e_Nsa[e])
                if suppress_unsafe:
                    e_P[e] = 0.0
                leaf_value = v
                break

            # ---- transition with the member fixed for this simulation -----
            ok = True
            for d in range(obs_dim):
                nv = t_obs[node, d] + deltas[mi, d]
                if not np.isfinite(nv):
                    ok = False
                if obs_clip > 0.0:
                    if nv < -obs_clip:
                        nv = -obs_clip
                    elif nv > obs_clip:
                        nv = obs_clip
                x_e[d] = nv                      # scratch reused for the next obs

            if (not ok) and term_nan:
                r = 0.0
                terminated = True
            else:
                # achieved_goal tracks the leading components of the observation
                dist2 = 0.0
                for i in range(dgoal.shape[0]):
                    diff = x_e[i] - dgoal[i]
                    dist2 += diff * diff
                dist = np.sqrt(dist2)
                success = dist < goal_thresh
                terminated = success and (not continuing_task)
                r = -step_penalty - dist_penalty * dist
                if terminated:
                    r += success_bonus
                r -= living_cost

            path_edges[n_path] = e
            path_rew[n_path] = r
            n_path += 1

            if terminated:
                leaf_value = 0.0
                break

            child = e_child[e]
            if child < 0:
                nn = counters[0]
                cap_c = cap_node
                if nn >= t_obs.shape[0] or counters[1] + cap_c > e_actions.shape[0]:
                    leaf_value = 0.0
                    break
                for d in range(obs_dim):
                    t_obs[nn, d] = x_e[d]
                t_depth[nn] = depth + 1
                t_N[nn] = 0
                t_edge_start[nn] = counters[1]
                t_n_edges[nn] = 0
                counters[1] += cap_c
                counters[0] = nn + 1
                e_child[e] = nn
                child = nn
            node = child

        # ---- backup ------------------------------------------------------
        G = leaf_value
        for t in range(n_path - 1, -1, -1):
            G = path_rew[t] + gamma * G
            e = path_edges[t]
            e_Nsa[e] += 1
            e_Q[e] += (G - e_Q[e]) / float(e_Nsa[e])

    return counters[2]


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class OfflineMCTS_DPW_PPO_Array_PointMaze(OfflineMCTS_DPW_PPO_SPIBB_Parametric):
    """SPIBB planner for PointMaze with a compiled search loop.

    Returns the highest-Q gate-certified root action, or the PPO action when
    no root action is certified or the calibrator is in the low-data regime.
    Supports a two-hidden-layer ensemble and the `speed_distance` reward only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ens = self._get_ensemble()
        if ens is None:
            raise ValueError("model_env does not expose an ensemble")
        self._ens_w = extract_ensemble_weights(ens._ens if hasattr(ens, "_ens") else ens)
        if self._ens_w.n_hidden != 2:
            raise ValueError(
                f"the compiled loop supports two hidden layers, got {self._ens_w.n_hidden}"
            )
        self._ppo_w = extract_ppo_actor_weights(self.ppo)
        self._val_w = extract_ppo_critic_weights(self.ppo)

        mc = self.model_env.cfg
        mode = str(getattr(mc, "reward_mode", "current"))
        if mode != "speed_distance":
            raise ValueError(
                f"only the speed_distance reward is ported, got {mode!r}; the "
                f"other modes would change what the planner optimises"
            )
        self._pm_goal_thresh = float(mc.goal_thresh)
        self._pm_step_penalty = float(mc.step_penalty)
        self._pm_dist_penalty = float(mc.dist_penalty)
        self._pm_success_bonus = float(mc.success_bonus)
        self._pm_living_cost = float(getattr(mc, "living_cost", 0.0))
        self._pm_continuing = bool(getattr(mc, "continuing_task", False))
        self._pm_obs_clip = float(getattr(mc, "obs_clip", 0.0) or 0.0)
        self._pm_term_nan = bool(getattr(mc, "terminate_on_nan", True))
        if bool(getattr(mc, "truncate_on_time_limit", False)):
            raise ValueError(
                "truncate_on_time_limit must be off for planning: the tree depth "
                "is the horizon, and a step counter shared across simulations "
                "would truncate them at arbitrary points"
            )

        cal = getattr(self, "calibrator", None)
        if cal is not None and getattr(cal, "bin_edges", None) is not None:
            self._bin_edges = np.asarray(cal.bin_edges, dtype=np.float64).ravel()
            self._bin_bounds = np.asarray(cal.bin_bounds, dtype=np.float64).ravel()
        else:
            self._bin_edges = np.zeros(2, dtype=np.float64)
            self._bin_bounds = np.zeros(1, dtype=np.float64)

        # Scratch, allocated once.
        self._h1_e = np.zeros(self._ens_w.W1.shape[1], dtype=np.float32)
        self._h2_e = np.zeros(self._ens_w.W2.shape[1], dtype=np.float32)
        self._x_e = np.zeros(self._ens_w.in_dim, dtype=np.float32)
        self._deltas = np.zeros((self._ens_w.K, self._ens_w.obs_dim), dtype=np.float32)
        self._h1_p = np.zeros(self._ppo_w.W1.shape[0], dtype=np.float32)
        self._h2_p = np.zeros(self._ppo_w.W2.shape[0], dtype=np.float32)
        self._out_p = np.zeros(self._ppo_w.act_dim, dtype=np.float32)
        self._h1_v = np.zeros(self._val_w.W1.shape[0], dtype=np.float32)
        self._h2_v = np.zeros(self._val_w.W2.shape[0], dtype=np.float32)
        self._out_v = np.zeros(1, dtype=np.float32)
        self._feat = np.zeros(self._ppo_w.obs_dim, dtype=np.float32)
        self._pr_buf = np.empty(max(1, int(self.cfg.n_actions_per_node)), dtype=np.float32)

    # ------------------------------------------------------------------ act
    def act(self, obs: Any) -> np.ndarray:
        """Plan from `obs` (dict observation) and return the action."""
        if (bool(self.cfg.conservative_on_low_data)
                and self.calibrator is not None
                and bool(getattr(self.calibrator, "meta", {}).get("low_data_regime", False))):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            return self._clip_action(self._ppo_mean(obs))

        o = np.asarray(obs["observation"], dtype=np.float32).ravel()
        dgoal = np.asarray(obs["desired_goal"], dtype=np.float32).ravel()

        sims = int(self.cfg.num_sims)
        depth = int(self.cfg.max_depth)
        cap_r, cap_n = int(self.cfg.n_actions_root), int(self.cfg.n_actions_per_node)
        act_dim = int(self._ppo_w.act_dim)
        max_nodes = sims * depth + 2
        tree = _ArrayTree(max_nodes, cap_r + max_nodes * cap_n, o.shape[0], act_dim)
        root = tree.new_node(o, 0, cap_r)
        # Root starts with the PPO action at prior 1.0; progressive widening
        # adds the rest.
        tree.add_edge(root, self._clip_action(self._ppo_mean(obs)), 1.0)

        model_indices = (self.rng.integers(0, int(self._ens_w.K), size=sims)
                         if str(self.cfg.model_sampling) == "fixed_per_sim"
                         else np.zeros(sims, dtype=np.int64))

        by_capacity = max_nodes * cap_n
        by_descent = sims * (depth + 1) * 8
        max_normals = (min(by_capacity, by_descent) + cap_r) * act_dim
        rng_state = self.rng.bit_generator.state
        normals = self.rng.standard_normal(size=max_normals).astype(np.float64)
        counters = np.array([tree.n_nodes, tree.n_alloc_edges, 0], dtype=np.int64)
        low_data = bool(self.calibrator is not None
                        and self.calibrator.meta.get("low_data_regime", False)
                        and bool(self.cfg.conservative_on_low_data))
        use_critic = bool(self.use_ppo_leaf_value
                          and str(self.cfg.leaf_value_mode) != "zero")

        w, p, v = self._ens_w, self._ppo_w, self._val_w
        run_simulations_pointmaze(
            tree.obs, tree.depth, tree.N, tree.edge_start, tree.n_edges,
            tree.actions, tree.P, tree.Q, tree.Nsa, tree.child, counters,
            w.W1, w.b1, w.W2, w.b2, w.W3, w.b3,
            w.input_mean, w.input_inv_std, w.has_norm,
            self._deltas, self._h1_e, self._h2_e, self._x_e,
            p.W1, p.b1, p.W2, p.b2, p.W3, p.b3, self._out_p, self._h1_p, self._h2_p,
            v.W1, v.b1, v.W2, v.b2, v.W3, v.b3, self._out_v, self._h1_v, self._h2_v,
            use_critic,
            self._feat, dgoal,
            self._bin_edges, self._bin_bounds, float(self.cfg.eps_threshold),
            bool(self.use_gate), low_data, bool(self.cfg.gate_inside_tree),
            bool(self.cfg.suppress_unsafe_actions),
            sims, depth, float(self.cfg.gamma),
            float(self.cfg.dpw_k), float(self.cfg.dpw_alpha), float(self.cfg.c_puct),
            cap_r, cap_n, float(self.cfg.ppo_std_scale),
            self._pm_goal_thresh, self._pm_step_penalty, self._pm_dist_penalty,
            self._pm_success_bonus, self._pm_living_cost, self._pm_continuing,
            self._pm_obs_clip, self._pm_term_nan,
            self._pr_buf,
            normals, np.asarray(model_indices, dtype=np.int64),
            np.empty(depth + 1, dtype=np.int64), np.empty(depth + 1, dtype=np.float64),
        )
        used = int(counters[2])
        if used > max_normals - act_dim:
            raise RuntimeError(
                f"action-sampling buffer exhausted ({used}/{max_normals})")
        # Rewind and advance by the deviates actually used, so the generator
        # state matches the reference planner.
        self.rng.bit_generator.state = rng_state
        if used:
            self.rng.standard_normal(size=used)
        tree.n_nodes, tree.n_alloc_edges = int(counters[0]), int(counters[1])

        s0 = int(tree.edge_start[root])
        n_e = int(tree.n_edges[root])
        safe_ids = [i for i in range(n_e)
                    if self._is_safe(obs, tree.actions[s0 + i])[0]]
        self.last_root_coverage = float(len(safe_ids)) / float(max(1, n_e))
        if not safe_ids:
            self.last_root_fallback = True
            return self._clip_action(self._ppo_mean(obs))

        self.last_root_fallback = False
        tol = 1e-9
        q_best = max(tree.Q[s0 + j] for j in safe_ids)
        cands = [j for j in safe_ids if (q_best - tree.Q[s0 + j]) <= tol]
        best_i = max(cands, key=lambda j: tree.Nsa[s0 + j])
        return np.asarray(tree.actions[s0 + best_i], dtype=np.float32)
