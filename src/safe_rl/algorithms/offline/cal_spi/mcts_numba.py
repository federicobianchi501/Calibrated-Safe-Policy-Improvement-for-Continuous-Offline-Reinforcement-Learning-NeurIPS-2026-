# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
Numba-accelerated drop-in for the parametric MCTS-DPW planner.

Subclasses `OfflineMCTS_DPW_PPO_SPIBB_Parametric` and overrides only the
hot calls:

  * `_ppo_mean`        -> njit PPO actor forward (extracted SB3 weights)
  * `_uncertainty_raw` -> njit ensemble forward + USAD (extracted weights)
  * `_search`          -> same algorithmic logic as the parent, but replaces
                          `model_env.step` with an inlined njit transition:
                          ensemble forward -> delta[mi] -> flat locomotion
                          reward and health-based termination.

The tree structure, PUCT selection, progressive widening, gating logic,
and root selection are inherited verbatim from the parametric class. This
is deliberate: we need byte-level algorithmic parity, and the only thing
that should differ is the numerical evaluation of the ensemble and of
the policy network (which differ from the torch equivalents within
float32 rounding, ~1e-5).

The numba planner is NOT a replacement for training or serialization.
It reads weights from a fitted EnsembleDynamicsSB3 and a loaded SB3 PPO
once at construction time, then runs read-only forward passes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from safe_rl.algorithms.offline.cal_spi.mcts_parametric import (
    Node,
    OfflineMCTSConfig,
    OfflineMCTS_DPW_PPO_SPIBB_Parametric,
)
from safe_rl.offline.ensemble_numba import (
    EnsembleNumbaWeights,
    PPOActorNumbaWeights,
    extract_ensemble_weights,
    extract_ppo_actor_weights,
    ensemble_forward_all,
    ensemble_forward_all_3h,
    ensemble_forward_all_4h,
    ppo_actor_mean,
    usad_from_deltas,
)


class OfflineMCTS_DPW_PPO_Numba(OfflineMCTS_DPW_PPO_SPIBB_Parametric):
    """
    Numba-backed drop-in replacement for the parametric planner.

    Same `act(obs)` signature and same config semantics. Works with the
    continuous locomotion ModelContinuousEnv (flat obs, reward and
    termination reconstructed from health/velocity indices).
    """

    def __init__(
        self,
        model_env: Any,
        ppo_policy: Any,
        cfg: OfflineMCTSConfig,
        calibrator: Optional[Any] = None,
        use_gate: bool = True,
        use_ppo_prior: bool = True,
        use_ppo_leaf_value: bool = False,
        rng_seed: int = 0,
    ):
        super().__init__(
            model_env=model_env,
            ppo_policy=ppo_policy,
            cfg=cfg,
            calibrator=calibrator,
            use_gate=use_gate,
            use_ppo_prior=use_ppo_prior,
            use_ppo_leaf_value=use_ppo_leaf_value,
            rng_seed=rng_seed,
        )
        ens = self._get_ensemble()
        if ens is None:
            raise ValueError("model_env does not expose an ensemble (no .ens / .ensemble)")
        if hasattr(ens, "_ens"):
            self._ens_w: EnsembleNumbaWeights = extract_ensemble_weights(ens._ens)
        else:
            self._ens_w = extract_ensemble_weights(ens)
        self._ppo_w: PPOActorNumbaWeights = extract_ppo_actor_weights(ppo_policy)

        # Scratch buffers reused across calls inside the hot loop.
        H1_ens = self._ens_w.W1.shape[1]
        H2_ens = self._ens_w.W2.shape[1]
        self._h1_e = np.zeros((H1_ens,), dtype=np.float32)
        self._h2_e = np.zeros((H2_ens,), dtype=np.float32)
        # Third / fourth hidden layer buffers (used when n_hidden in {3, 4});
        # zero-shape sentinels otherwise to keep call signatures uniform.
        if self._ens_w.n_hidden >= 3:
            H3_ens = self._ens_w.W_mid.shape[1]
            self._h3_e = np.zeros((H3_ens,), dtype=np.float32)
        else:
            self._h3_e = np.zeros((0,), dtype=np.float32)
        if self._ens_w.n_hidden == 4:
            H4_ens = self._ens_w.W_mid2.shape[1]
            self._h4_e = np.zeros((H4_ens,), dtype=np.float32)
        else:
            self._h4_e = np.zeros((0,), dtype=np.float32)
        self._x_e = np.zeros((self._ens_w.in_dim,), dtype=np.float32)
        self._deltas = np.zeros((self._ens_w.K, self._ens_w.obs_dim), dtype=np.float32)

        H1_p = self._ppo_w.W1.shape[0]
        H2_p = self._ppo_w.W2.shape[0]
        self._h1_p = np.zeros((H1_p,), dtype=np.float32)
        self._h2_p = np.zeros((H2_p,), dtype=np.float32)
        self._out_p = np.zeros((self._ppo_w.act_dim,), dtype=np.float32)

        # ModelContinuousEnv reward/termination parameters cached locally so
        # `_search` can inline the transition without a Python attribute walk
        # on every step.
        mc = getattr(model_env, "cfg", None)
        if mc is None:
            raise ValueError("model_env has no .cfg; expected ModelContinuousEnvConfig")
        self._mc_vel_idx = int(mc.vel_obs_idx)
        self._mc_z_idx = int(mc.z_obs_idx)
        self._mc_ang_idx = int(mc.angle_obs_idx)
        self._mc_ctrl = float(mc.ctrl_cost_weight)
        self._mc_h_r = float(mc.healthy_reward)
        self._mc_z_lo = float(mc.healthy_z_lo)
        self._mc_z_hi = float(mc.healthy_z_hi)
        self._mc_ang = float(mc.healthy_angle)
        self._mc_obs_clip = float(mc.obs_clip) if mc.obs_clip is not None else 0.0
        self._mc_term_nan = bool(mc.terminate_on_nan)

    # ------------------------------------------------------------------ PPO
    def _ppo_mean(self, obs) -> np.ndarray:
        obs_np = np.asarray(
            obs["observation"] if isinstance(obs, dict) else obs,
            dtype=np.float32,
        ).reshape(-1)
        ppo_actor_mean(
            obs_np,
            self._ppo_w.W1, self._ppo_w.b1,
            self._ppo_w.W2, self._ppo_w.b2,
            self._ppo_w.W3, self._ppo_w.b3,
            self._out_p, self._h1_p, self._h2_p,
        )
        # Return a fresh copy so callers can cache it safely.
        return self._out_p.copy()

    # ---------------------------------------------------------- Uncertainty
    def _usad_from_cached_deltas(self) -> float:
        """USAD computed from the current contents of the deltas buffer."""
        return float(usad_from_deltas(self._deltas))

    def _compute_deltas(self, obs, action) -> np.ndarray:
        obs_np = np.asarray(
            obs["observation"] if isinstance(obs, dict) else obs,
            dtype=np.float32,
        ).reshape(-1)
        act_np = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._ens_w.n_hidden == 4:
            ensemble_forward_all_4h(
                obs_np, act_np,
                self._ens_w.W1, self._ens_w.b1, self._ens_w.W2, self._ens_w.b2,
                self._ens_w.W_mid, self._ens_w.b_mid,
                self._ens_w.W_mid2, self._ens_w.b_mid2,
                self._ens_w.W3, self._ens_w.b3,
                self._ens_w.input_mean, self._ens_w.input_inv_std, self._ens_w.has_norm,
                self._deltas, self._h1_e, self._h2_e, self._h3_e, self._h4_e, self._x_e,
            )
        elif self._ens_w.n_hidden == 3:
            ensemble_forward_all_3h(
                obs_np, act_np,
                self._ens_w.W1, self._ens_w.b1, self._ens_w.W2, self._ens_w.b2,
                self._ens_w.W_mid, self._ens_w.b_mid,
                self._ens_w.W3, self._ens_w.b3,
                self._ens_w.input_mean, self._ens_w.input_inv_std, self._ens_w.has_norm,
                self._deltas, self._h1_e, self._h2_e, self._h3_e, self._x_e,
            )
        else:
            ensemble_forward_all(
                obs_np, act_np,
                self._ens_w.W1, self._ens_w.b1, self._ens_w.W2, self._ens_w.b2,
                self._ens_w.W3, self._ens_w.b3,
                self._ens_w.input_mean, self._ens_w.input_inv_std, self._ens_w.has_norm,
                self._deltas, self._h1_e, self._h2_e, self._x_e,
            )
        return self._deltas  # internal buffer — do not mutate externally

    def _is_safe_from_u(self, u: float) -> bool:
        """
        Replicates `_is_safe` gate logic given a precomputed USAD value.
        Used inside the search loop when we have already computed deltas.
        """
        if not self.use_gate:
            return True
        if str(self.cfg.gate_mode) == "uncertainty":
            return float(u) <= float(self.cfg.uncertainty_threshold)
        # calibrated
        if self.calibrator is None:
            raise ValueError("gate_mode='calibrated' requires calibrator")
        if bool(self.calibrator.meta.get("low_data_regime", False)) and bool(
            self.cfg.conservative_on_low_data
        ):
            return False
        b = float(self.calibrator.bound(float(u)))
        if not np.isfinite(b):
            return False
        return b <= float(self.cfg.eps_threshold)

    def _uncertainty_raw(self, obs, action) -> float:
        ut = str(self.cfg.uncertainty_type)
        if ut == "usad":
            deltas = self._compute_deltas(obs, action)
            return float(usad_from_deltas(deltas))
        if ut == "epistemic":
            deltas = self._compute_deltas(obs, action)
            return float(np.var(deltas, axis=0).mean())
        return super()._uncertainty_raw(obs, action)

    # ------------------------------------------------------------- Step fast
    def _step_from_cached_deltas(
        self, obs: np.ndarray, action: np.ndarray, mi: int,
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Same as `_step_fast` but uses the deltas already in `self._deltas`
        from a previous `_compute_deltas` call. Caller must guarantee that
        the buffer still corresponds to the intended (obs, action) pair.
        """
        obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
        act_np = np.asarray(action, dtype=np.float32).reshape(-1)
        obs_next = obs_np + self._deltas[int(mi)]
        return self._finalize_step(obs_next, act_np)

    def _step_fast(
        self, obs: np.ndarray, action: np.ndarray, mi: int,
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Inlined model-env step: ensemble forward -> delta[mi] -> next obs,
        locomotion reward, health-based termination.

        Matches ModelContinuousEnv.step up to the float32 rounding difference
        between the torch and numba forward passes.
        """
        deltas = self._compute_deltas(obs, action)
        obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
        act_np = np.asarray(action, dtype=np.float32).reshape(-1)
        obs_next = obs_np + deltas[int(mi)]
        return self._finalize_step(obs_next, act_np)

    def _finalize_step(
        self, obs_next: np.ndarray, act_np: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool]:
        ok = bool(np.isfinite(obs_next).all())
        if self._mc_obs_clip > 0.0:
            obs_next = np.clip(obs_next, -self._mc_obs_clip, self._mc_obs_clip).astype(np.float32)
        else:
            obs_next = obs_next.astype(np.float32)

        if (not ok) and self._mc_term_nan:
            return obs_next, 0.0, True, False

        z = float(obs_next[self._mc_z_idx])
        ang = float(obs_next[self._mc_ang_idx])
        healthy = (
            (z >= self._mc_z_lo) and (z <= self._mc_z_hi) and (abs(ang) <= self._mc_ang)
        )
        x_vel = float(obs_next[self._mc_vel_idx])
        ctrl = float(self._mc_ctrl * float(np.sum(act_np * act_np)))
        r = x_vel + (self._mc_h_r if healthy else 0.0) - ctrl
        terminated = not healthy
        return obs_next, float(r), bool(terminated), False

    # ------------------------------------------------------------------ MCTS
    def _search(self, node: Node, mi: Optional[int]) -> float:
        node.N += 1

        if node.depth >= int(self.cfg.max_depth):
            return 0.0

        max_children = int(float(self.cfg.dpw_k) * (float(node.N) ** float(self.cfg.dpw_alpha)))
        cap = int(self.cfg.n_actions_root) if node.depth == 0 else int(self.cfg.n_actions_per_node)
        target = min(max_children, cap)

        if len(node.actions) < target:
            add_n = min(8, target - len(node.actions))
            if add_n > 0:
                self._expand_action(node, add_n)

        if len(node.actions) == 0:
            return self._leaf_value(node.obs)

        a_idx = self._select_child(node)
        if a_idx < 0:
            return self._leaf_value(node.obs)

        a = self._clip_action(node.actions[a_idx])
        node.actions[a_idx] = a

        # Compute ensemble deltas once, then reuse for both the gate check
        # (USAD) and the transition. Bit-exact because deltas are a pure
        # function of (obs, action, weights) and the rng is not touched here.
        self._compute_deltas(node.obs, a)
        u = self._usad_from_cached_deltas()
        safe = self._is_safe_from_u(u)
        if self.use_gate and bool(self.cfg.gate_inside_tree) and (not safe):
            v = self._leaf_value(node.obs)
            node.Nsa[a_idx] += 1
            node.Q[a_idx] += (v - node.Q[a_idx]) / float(node.Nsa[a_idx])
            if bool(self.cfg.suppress_unsafe_actions):
                node.P[a_idx] = 0.0
            return float(v)

        mi_step = mi
        if str(self.cfg.model_sampling) == "resample_each_step":
            mi_step = int(self.rng.integers(0, int(self._ens_w.K)))
        if mi_step is None:
            mi_step = int(self.rng.integers(0, int(self._ens_w.K)))

        obs_next, r, terminated, truncated = self._step_from_cached_deltas(
            node.obs, a, int(mi_step)
        )

        child = node.children[a_idx]
        if child is None:
            child = Node(obs=obs_next, depth=node.depth + 1)
            node.children[a_idx] = child

        if bool(terminated) or bool(truncated):
            G = float(r)
        else:
            G = float(r) + float(self.cfg.gamma) * float(self._search(child, mi=mi_step))

        node.Nsa[a_idx] += 1
        node.Q[a_idx] += (G - node.Q[a_idx]) / float(node.Nsa[a_idx])
        return float(G)


# =============================================================================
# CSPI variant on top of the numba backend
# =============================================================================


from safe_rl.algorithms.offline.cal_spi.mcts_cspi import _CSPIMixin  # noqa: E402


class OfflineMCTS_DPW_PPO_Numba_CSPI(_CSPIMixin, OfflineMCTS_DPW_PPO_Numba):
    """
    Conservative Safe Policy Improvement on top of the numba-backed planner.

    Mirrors `OfflineMCTS_DPW_PPO_SPIBB_CSPI_Parametric` verbatim for the
    root-selection logic (baseline gate, V_hat_b rollout, margin test) and
    inherits the numba-accelerated `_search` from the numba base. The
    rollout for V_hat_b still goes through the torch-backed model_env and
    ppo.predict, but that is only `cspi_v_rollout_horizon` steps per
    `act()` call and is negligible compared to the MCTS simulations.
    """

    def __init__(
        self,
        *args,
        cspi_margin_mult: float = 1.0,
        cspi_v_rollout_horizon: int = 0,  # 0 = auto-match max_depth
        cspi_v_rollout_n_samples: int = 1,
        cspi_e_plan_scale: float = 0.5,
        cspi_delta_piB: float = 0.0,
        cspi_frozen_root: bool = True,
        cspi_n_min_root_visits: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cspi_configure(
            margin_mult=cspi_margin_mult,
            v_rollout_horizon=cspi_v_rollout_horizon,
            v_rollout_n_samples=cspi_v_rollout_n_samples,
            e_plan_scale=cspi_e_plan_scale,
            delta_piB=cspi_delta_piB,
            frozen_root=cspi_frozen_root,
            n_min_root_visits=cspi_n_min_root_visits,
        )
        self._cspi_root_safe_ids: Optional[list] = None
        self.cspi_reset_stats()

    # ------------------------------------------------------------------
    # Numba V_b rollout: averages `_cspi_v_rollout_n_samples` independent
    # rollouts of length `_cspi_v_rollout_horizon` using the njit PPO
    # actor and inlined ensemble step, then returns the mean return.
    # Cost per rollout step ~0.6 ms (vs ~7 ms torch), so even averaging
    # 5 rollouts at horizon 100 stays under 1 % of a paper-grade act().
    # ------------------------------------------------------------------
    def _cspi_v_b_rollout(self, obs_root: Any) -> float:
        n_samples = int(max(1, self._cspi_v_rollout_n_samples))
        gamma = float(self.cfg.gamma)
        horizon = int(self._cspi_v_rollout_horizon)
        K = int(self._ens_w.K)
        gate_inside = bool(self.cfg.gate_inside_tree)
        acc = 0.0
        for _s in range(n_samples):
            G = 0.0
            discount = 1.0
            obs = np.asarray(obs_root, dtype=np.float32).reshape(-1)
            for _t in range(horizon):
                a = self._ppo_mean(obs)
                if gate_inside:
                    safe, _, _ = self._is_safe(obs, a)
                    if not safe:
                        break
                mi = int(self.rng.integers(0, K))
                obs_next, rew, terminated, truncated = self._step_fast(obs, a, mi)
                G += discount * float(rew)
                discount *= gamma
                if bool(terminated) or bool(truncated):
                    break
                obs = obs_next
            acc += G
        return float(acc / float(n_samples))

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self._cspi_stats_n_calls += 1

        if (
            bool(self.cfg.conservative_on_low_data)
            and bool(getattr(self, "calibrator", None) is not None)
            and bool(getattr(self.calibrator, "meta", {}).get("low_data_regime", False))
        ):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return self._clip_action(self._ppo_mean(obs))

        a_b = self._cspi_baseline_action(obs)
        baseline_safe, _, _ = self._is_safe(obs, a_b)
        if not bool(baseline_safe):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return a_b

        root = Node(obs=obs, depth=0)
        root.actions.append(a_b)
        root.P.append(1.0)
        root.Q.append(0.0)
        root.Nsa.append(0)
        root.children.append(None)

        if self._cspi_frozen_root:
            K = int(self.cfg.n_actions_root)
            if K > len(root.actions):
                self._expand_action(root, K - len(root.actions))

        root_safe_ids = []
        for i, a in enumerate(root.actions):
            safe, _, _ = self._is_safe(obs, a)
            if safe:
                root_safe_ids.append(i)
        self._cspi_root_safe_ids = root_safe_ids
        self._cspi_stats_root_total_safe += len(root_safe_ids)

        if root_safe_ids:
            self._cspi_n_min_root_visits_eff = max(1, min(
                int(self._cspi_n_min_root_visits),
                int(self.cfg.num_sims) // len(root_safe_ids),
            ))
        else:
            self._cspi_n_min_root_visits_eff = 0

        model_indices = None
        if str(self.cfg.model_sampling) == "fixed_per_sim":
            model_indices = self.rng.integers(0, int(self._ens_w.K), size=int(self.cfg.num_sims))

        for s in range(int(self.cfg.num_sims)):
            mi = None
            if model_indices is not None:
                mi = int(model_indices[s])
            self._search(root, mi=mi)

        safe_ids = root_safe_ids
        self.last_root_coverage = float(len(safe_ids)) / float(max(1, len(root.actions)))

        if len(safe_ids) == 0:
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return self._clip_action(self._ppo_mean(obs))

        min_visits = min(int(root.Nsa[j]) for j in safe_ids)
        total_visits = sum(int(root.Nsa[j]) for j in safe_ids)
        self._cspi_stats_root_visits_sum += total_visits
        if self._cspi_stats_n_calls == 1 or min_visits < self._cspi_stats_root_visits_min:
            self._cspi_stats_root_visits_min = min_visits

        tol = 1e-9
        q_best = max(root.Q[j] for j in safe_ids)
        cands = [j for j in safe_ids if (q_best - root.Q[j]) <= tol]
        best_i = max(cands, key=lambda j: root.Nsa[j])
        a_star = np.asarray(root.actions[best_i], dtype=np.float32)

        # Root-symmetric switch rule: compare Q_hat(a*) with Q_hat(a_b)
        # from the same MCTS tree, same gate, same scale. a_b is
        # root.actions[0] by construction.
        q_b = float(root.Q[0])
        eta_op = float(self._cspi_margin_mult) * float(self._cspi_e_plan_scale)
        margin_observed = float(q_best) - float(q_b)
        if margin_observed >= eta_op:
            self.last_root_fallback = False
            self._cspi_stats_n_switch += 1
            self._cspi_stats_margin_sum += margin_observed
            if margin_observed < self._cspi_stats_margin_min:
                self._cspi_stats_margin_min = margin_observed
            if margin_observed > self._cspi_stats_margin_max:
                self._cspi_stats_margin_max = margin_observed
            return a_star
        self.last_root_fallback = True
        self._cspi_stats_n_fallback_margin += 1
        return a_b

    # ---- Overridden _search for frozen root + forced min visits ----
    def _search(self, node: Node, mi: Optional[int]) -> float:
        node.N += 1

        if node.depth >= int(self.cfg.max_depth):
            return 0.0

        is_root = (node.depth == 0)
        if not (is_root and self._cspi_frozen_root):
            max_children = int(
                float(self.cfg.dpw_k) * (float(node.N) ** float(self.cfg.dpw_alpha))
            )
            cap = (
                int(self.cfg.n_actions_root)
                if node.depth == 0
                else int(self.cfg.n_actions_per_node)
            )
            target = min(max_children, cap)
            if len(node.actions) < target:
                add_n = min(8, target - len(node.actions))
                if add_n > 0:
                    self._expand_action(node, add_n)

        if len(node.actions) == 0:
            return self._leaf_value(node.obs)

        a_idx = -1
        n_min_eff = getattr(self, "_cspi_n_min_root_visits_eff", 0)
        if (
            is_root
            and self._cspi_frozen_root
            and n_min_eff > 0
            and self._cspi_root_safe_ids
        ):
            for j in self._cspi_root_safe_ids:
                if int(node.Nsa[j]) < int(n_min_eff):
                    a_idx = int(j)
                    break
        if a_idx < 0:
            a_idx = self._select_child(node)
        if a_idx < 0:
            return self._leaf_value(node.obs)

        a = self._clip_action(node.actions[a_idx])
        node.actions[a_idx] = a

        # Cached-deltas pipeline: compute once, reuse for USAD + transition.
        self._compute_deltas(node.obs, a)
        u = self._usad_from_cached_deltas()
        safe = self._is_safe_from_u(u)
        if self.use_gate and bool(self.cfg.gate_inside_tree) and (not safe):
            v = self._leaf_value(node.obs)
            node.Nsa[a_idx] += 1
            node.Q[a_idx] += (v - node.Q[a_idx]) / float(node.Nsa[a_idx])
            if bool(self.cfg.suppress_unsafe_actions):
                node.P[a_idx] = 0.0
            return float(v)

        mi_step = mi
        if str(self.cfg.model_sampling) == "resample_each_step":
            mi_step = int(self.rng.integers(0, int(self._ens_w.K)))
        if mi_step is None:
            mi_step = int(self.rng.integers(0, int(self._ens_w.K)))

        obs_next, r, terminated, truncated = self._step_from_cached_deltas(
            node.obs, a, int(mi_step)
        )

        child = node.children[a_idx]
        if child is None:
            child = Node(obs=obs_next, depth=node.depth + 1)
            node.children[a_idx] = child

        if bool(terminated) or bool(truncated):
            G = float(r)
        else:
            G = float(r) + float(self.cfg.gamma) * float(self._search(child, mi=mi_step))

        node.Nsa[a_idx] += 1
        node.Q[a_idx] += (G - node.Q[a_idx]) / float(node.Nsa[a_idx])
        return float(G)
