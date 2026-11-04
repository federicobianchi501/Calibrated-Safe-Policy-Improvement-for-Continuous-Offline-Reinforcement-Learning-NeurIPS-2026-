# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Compiled simulation loop for the array MCTS-DPW planner (`mcts_array`).

Selection, progressive widening, gate, model transition and backup run in one
njit function over the preallocated tree buffers. Action-sampling deviates are
drawn outside from the planner's numpy Generator and passed in as a flat
buffer consumed with a cursor; since `standard_normal(size=(n, d))` fills
row-major, this reproduces the reference sequence of per-expansion draws.
The gate (bound lookup and comparison against eps) is compiled without
fastmath; value arithmetic uses fastmath.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from safe_rl.offline.ensemble_numba import (
    ensemble_forward_all,
    ppo_actor_mean,
    usad_from_deltas,
)
from safe_rl.algorithms.offline.cal_spi.mcts_numba_scaled import (
    usad_from_deltas_scaled,
)


@njit(cache=True, fastmath=False, inline="always")
def _bound_lookup(u, bin_edges, bin_bounds):
    """Calibrated bound B(u); see UncertaintyCalibrator.bound."""
    nb = bin_bounds.shape[0]
    ne = bin_edges.shape[0]
    if u <= bin_edges[0]:
        return bin_bounds[0]
    if u >= bin_edges[ne - 1]:
        return bin_bounds[nb - 1]
    lo = 0
    hi = ne
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


@njit(cache=True, fastmath=False, inline="always")
def _is_safe(u, bin_edges, bin_bounds, eps, use_gate, low_data):
    """ModelSafe(s,a): B(u) <= eps, with low-data and non-finite forcing unsafe."""
    if not use_gate:
        return True
    if low_data:
        return False
    b = _bound_lookup(u, bin_edges, bin_bounds)
    if not np.isfinite(b):
        return False
    return b <= eps


@njit(cache=True, fastmath=False)
def _ctrl_cost_f32(a, n):
    """Sum of squared action components, bit-identical to `np.sum(a * a)`.

    Float32 accumulation in NumPy's pairwise order: sequential below 8 terms,
    eight accumulators combined as a balanced tree from 8 to 128 terms
    (`mcts_array` rejects action dimensions above 128). Not inlined and compiled
    without fastmath so the summation order is kept inside the fastmath caller;
    a float64 sum can differ in the last bits and change PUCT ties.
    """
    if n < 8:
        acc = np.float32(0.0)
        for i in range(n):
            acc += a[i] * a[i]
        return acc
    r0 = a[0] * a[0]
    r1 = a[1] * a[1]
    r2 = a[2] * a[2]
    r3 = a[3] * a[3]
    r4 = a[4] * a[4]
    r5 = a[5] * a[5]
    r6 = a[6] * a[6]
    r7 = a[7] * a[7]
    i = 8
    while i < n - (n % 8):
        r0 += a[i] * a[i]
        r1 += a[i + 1] * a[i + 1]
        r2 += a[i + 2] * a[i + 2]
        r3 += a[i + 3] * a[i + 3]
        r4 += a[i + 4] * a[i + 4]
        r5 += a[i + 5] * a[i + 5]
        r6 += a[i + 6] * a[i + 6]
        r7 += a[i + 7] * a[i + 7]
        i += 8
    acc = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    for j in range(n - (n % 8), n):
        acc += a[j] * a[j]
    return acc


@njit(cache=True, fastmath=False)
def _pairwise_sum_f32(v, n):
    """Float32 sum in NumPy's pairwise order; see `_ctrl_cost_f32`."""
    if n < 8:
        acc = np.float32(0.0)
        for i in range(n):
            acc += v[i]
        return acc
    r0 = v[0]; r1 = v[1]; r2 = v[2]; r3 = v[3]
    r4 = v[4]; r5 = v[5]; r6 = v[6]; r7 = v[7]
    i = 8
    while i < n - (n % 8):
        r0 += v[i]; r1 += v[i + 1]; r2 += v[i + 2]; r3 += v[i + 3]
        r4 += v[i + 4]; r5 += v[i + 5]; r6 += v[i + 6]; r7 += v[i + 7]
        i += 8
    acc = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    for j in range(n - (n % 8), n):
        acc += v[j]
    return acc


@njit(cache=True, fastmath=False)
def _dist2_f32(row, mean, n):
    """`np.linalg.norm(a - mean) ** 2` in float32, NumPy's order.

    The square root is taken and squared, as in the reference, since this is
    not the identity in float32.
    """
    if n < 8:
        acc = np.float32(0.0)
        for i in range(n):
            d = row[i] - mean[i]
            acc += d * d
    else:
        d0 = row[0] - mean[0]; d1 = row[1] - mean[1]
        d2 = row[2] - mean[2]; d3 = row[3] - mean[3]
        d4 = row[4] - mean[4]; d5 = row[5] - mean[5]
        d6 = row[6] - mean[6]; d7 = row[7] - mean[7]
        r0 = d0 * d0; r1 = d1 * d1; r2 = d2 * d2; r3 = d3 * d3
        r4 = d4 * d4; r5 = d5 * d5; r6 = d6 * d6; r7 = d7 * d7
        i = 8
        while i < n - (n % 8):
            e0 = row[i] - mean[i]; e1 = row[i + 1] - mean[i + 1]
            e2 = row[i + 2] - mean[i + 2]; e3 = row[i + 3] - mean[i + 3]
            e4 = row[i + 4] - mean[i + 4]; e5 = row[i + 5] - mean[i + 5]
            e6 = row[i + 6] - mean[i + 6]; e7 = row[i + 7] - mean[i + 7]
            r0 += e0 * e0; r1 += e1 * e1; r2 += e2 * e2; r3 += e3 * e3
            r4 += e4 * e4; r5 += e5 * e5; r6 += e6 * e6; r7 += e7 * e7
            i += 8
        acc = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        for j in range(n - (n % 8), n):
            d = row[j] - mean[j]
            acc += d * d
    nrm = np.float32(np.sqrt(acc))
    return np.float32(nrm * nrm)


@njit(cache=True, fastmath=False)
def _sample_ppo_block(mean, normals, cur, add_n, act_dim, ppo_std,
                      e_actions, e_P, e_Q, e_Nsa, e_child, s0, pr):
    """Append `add_n` PPO-prior actions to the edge arrays, as `_sample_ppo`.

    Actions are `mean + std * n` computed in float32 (deviate rounded to float32
    before scaling) and clipped to [-1, 1]. The PUCT prior is a Gaussian in the
    squared distance from the mean, normalised with a float32 pairwise sum. The
    exponential is evaluated in float64 and rounded to float32, so priors may
    differ from NumPy's float32 `exp` in the last bit.

    Returns the updated cursor into `normals`.
    """
    std32 = np.float32(ppo_std)
    var = ppo_std * ppo_std
    if var < 1e-6:
        var = 1e-6
    inv = np.float32(var)
    # Unnormalised priors are staged in the float32 caller-owned buffer `pr`,
    # then written to the float64 edge array after normalisation.
    for j in range(add_n):
        for d in range(act_dim):
            n32 = np.float32(normals[cur])
            cur += 1
            v = np.float32(mean[d] + np.float32(std32 * n32))
            if v < np.float32(-1.0):
                v = np.float32(-1.0)
            elif v > np.float32(1.0):
                v = np.float32(1.0)
            e_actions[s0 + j, d] = v
        d2 = _dist2_f32(e_actions[s0 + j], mean, act_dim)
        pr[j] = np.float32(np.exp(np.float64(np.float32(np.float32(-0.5) * d2 / inv))))
        e_Q[s0 + j] = 0.0
        e_Nsa[s0 + j] = 0
        e_child[s0 + j] = -1
    denom = np.float32(_pairwise_sum_f32(pr, add_n) + np.float32(1e-8))
    for j in range(add_n):
        e_P[s0 + j] = np.float32(pr[j] / denom)
    return cur


@njit(cache=True, fastmath=True, inline="always")
def _puct(n_e, node_N, Nsa, Q, P, s, c_puct):
    """PUCT argmax over the `n_e` edges starting at `s`; local index or -1."""
    N = float(node_N)
    if N < 1.0:
        N = 1.0
    sqrtN = np.sqrt(N)
    best_i = -1
    best = -1e30
    for i in range(n_e):
        nsa = float(Nsa[s + i])
        if nsa < 1.0:
            nsa = 1.0
        score = Q[s + i] + c_puct * P[s + i] * sqrtN / nsa
        if score > best:
            best = score
            best_i = i
    return best_i


@njit(cache=True, fastmath=True)
def run_simulations(
    # tree storage
    t_obs, t_depth, t_N, t_edge_start, t_n_edges,
    e_actions, e_P, e_Q, e_Nsa, e_child, counters,
    # ensemble weights
    W1, b1, W2, b2, W3, b3, in_mean, in_inv_std, has_norm,
    deltas, h1_e, h2_e, x_e,
    # ppo actor weights
    pW1, pb1, pW2, pb2, pW3, pb3, p_out, p_h1, p_h2,
    # gate
    bin_edges, bin_bounds, eps, use_gate, low_data, gate_inside_tree,
    suppress_unsafe,
    # planner config
    num_sims, max_depth, gamma, dpw_k, dpw_alpha, c_puct,
    cap_root, cap_node, ppo_std,
    # frozen root
    frozen_root, root_safe_ids, n_root_safe, n_min_visits,
    # model env
    vel_idx, z_idx, ang_idx, ctrl_w, healthy_r, z_lo, z_hi, ang_max,
    obs_clip, term_nan,
    # scale-aware disagreement
    inv_scale, use_scaled,
    # scratch for the per-block priors
    pr_buf,
    # randomness, pre-drawn outside
    normals, model_indices,
    # scratch
    path_edges, path_rew,
):
    """Run `num_sims` simulations from root node 0; returns the normals used.

    `counters` = [n_nodes, n_alloc_edges, normal_cursor], updated in place.
    `model_indices[sim]` is the ensemble member used for simulation `sim`.
    Gate rejections and depth/termination leaves back up a value of 0.
    """
    obs_dim = t_obs.shape[1]
    act_dim = e_actions.shape[1]
    K = W1.shape[0]

    for sim in range(num_sims):
        mi = model_indices[sim]
        node = 0
        n_path = 0
        leaf_value = 0.0

        while True:
            t_N[node] += 1
            depth = t_depth[node]
            if depth >= max_depth:
                leaf_value = 0.0
                break

            is_root = (depth == 0)
            cap = cap_root if is_root else cap_node

            if not (is_root and frozen_root):
                mx = int(dpw_k * (float(t_N[node]) ** dpw_alpha))
                target = mx if mx < cap else cap
                have = t_n_edges[node]
                if have < target:
                    add_n = target - have
                    if add_n > 8:
                        add_n = 8
                    room = cap - have
                    if add_n > room:
                        add_n = room
                    if add_n > 0:
                        # PPO mean at this node, clipped to [-1, 1]
                        ppo_actor_mean(t_obs[node], pW1, pb1, pW2, pb2, pW3, pb3,
                                       p_out, p_h1, p_h2)
                        for d in range(act_dim):
                            if p_out[d] < -1.0:
                                p_out[d] = -1.0
                            elif p_out[d] > 1.0:
                                p_out[d] = 1.0
                        s0 = t_edge_start[node] + have
                        counters[2] = _sample_ppo_block(
                            p_out, normals, counters[2], add_n, act_dim, ppo_std,
                            e_actions, e_P, e_Q, e_Nsa, e_child, s0, pr_buf)
                        t_n_edges[node] = have + add_n

            n_e = t_n_edges[node]
            if n_e == 0:
                leaf_value = 0.0
                break

            s = t_edge_start[node]
            e_local = -1
            if is_root and frozen_root and n_min_visits > 0 and n_root_safe > 0:
                for jj in range(n_root_safe):
                    j = root_safe_ids[jj]
                    if j < n_e and e_Nsa[s + j] < n_min_visits:
                        e_local = j
                        break
            if e_local < 0:
                e_local = _puct(n_e, t_N[node], e_Nsa, e_Q, e_P, s, c_puct)
            if e_local < 0:
                leaf_value = 0.0
                break

            e = s + e_local
            for d in range(act_dim):
                if e_actions[e, d] < -1.0:
                    e_actions[e, d] = -1.0
                elif e_actions[e, d] > 1.0:
                    e_actions[e, d] = 1.0

            ensemble_forward_all(t_obs[node], e_actions[e], W1, b1, W2, b2, W3, b3,
                                 in_mean, in_inv_std, has_norm, deltas, h1_e, h2_e, x_e)
            if use_scaled:
                u = usad_from_deltas_scaled(deltas, inv_scale)
            else:
                u = usad_from_deltas(deltas)

            if use_gate and gate_inside_tree and (not _is_safe(
                    u, bin_edges, bin_bounds, eps, use_gate, low_data)):
                v = 0.0
                e_Nsa[e] += 1
                e_Q[e] += (v - e_Q[e]) / float(e_Nsa[e])
                if suppress_unsafe:
                    e_P[e] = 0.0
                leaf_value = v
                break

            # transition with the member fixed for this simulation
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
                x_e[d] = nv                      # reuse scratch for next obs

            if (not ok) and term_nan:
                r = 0.0
                terminated = True
            else:
                z = x_e[z_idx]
                ang = x_e[ang_idx]
                healthy = (z >= z_lo) and (z <= z_hi) and (abs(ang) <= ang_max)
                ctrl = _ctrl_cost_f32(e_actions[e], act_dim)
                r = x_e[vel_idx] + (healthy_r if healthy else 0.0) - ctrl_w * ctrl
                terminated = not healthy

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

        # backup
        G = leaf_value
        for t in range(n_path - 1, -1, -1):
            G = path_rew[t] + gamma * G
            e = path_edges[t]
            e_Nsa[e] += 1
            e_Q[e] += (G - e_Q[e]) / float(e_Nsa[e])

    return counters[2]
