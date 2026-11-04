# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Deployed Cal-SPI policy on top of the calibrated MCTS-DPW planners.

  - OfflineMCTS_DPW_PPO_SPIBB_CSPI_Parametric  (reference backend)
  - OfflineMCTS_DPW_PPO_Batched_CSPI           (batched backend)

Both override ``act`` and ``_search``; expansion, PUCT selection and the gate inside
the tree are inherited. Per call to ``act``:

  1. Low-data regime (with ``conservative_on_low_data``): return the baseline action
     a_b = pi_b(s) without searching (conservative collapse).
  2. If a_b fails the gate, return a_b without searching.
  3. Frozen-root regime (default): the root candidate set (a_b first, then PPO samples
     up to ``n_actions_root``) is drawn once before the simulations and not widened;
     the safe set A_safe(s) is fixed for the whole search. Every safe root action gets
     at least min(n_min_root_visits, num_sims // |A_safe(s)|) visits before PUCT.
  4. Baseline-relative switch: with a* the best safe root action by Q, return a* if
     Q(a*) - Q(a_b) >= cspi_margin_mult * cspi_e_plan_scale, otherwise return a_b.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from safe_rl.algorithms.offline.cal_spi.mcts_parametric import (
    OfflineMCTS_DPW_PPO_SPIBB_Parametric,
    OfflineMCTSConfig,
    Node,
)
from safe_rl.algorithms.offline.cal_spi.mcts_batched import (
    OfflineMCTS_DPW_PPO_Batched,
    _BatchedNode,
)


# =============================================================================
# Shared CSPI helpers
# =============================================================================


class _CSPIMixin:
    """CSPI configuration, statistics counters and baseline helpers.

    Expected attributes on the host class:
        model_env: step(obs, action, rng=...) -> (obs', r, terminated, truncated, info)
        ppo: predict(obs, deterministic=True) -> (action, state)
        rng: numpy.random.Generator
        cfg: planner config with eps_threshold, gamma, max_depth
    """

    # Set through the subclass constructor or cspi_configure().
    _cspi_margin_mult: float = 1.0
    _cspi_v_rollout_horizon: int = 20
    _cspi_v_rollout_n_samples: int = 1
    _cspi_e_plan_scale: float = 0.5
    _cspi_delta_piB: float = 0.0
    # Frozen-root regime: the root candidate set is expanded once at the start of
    # act() and not widened during the simulations; each safe root action receives
    # at least _cspi_n_min_root_visits visits (capped by num_sims // |A_safe|)
    # before PUCT selection takes over at the root.
    _cspi_frozen_root: bool = True
    _cspi_n_min_root_visits: int = 3

    # Statistics counters, reset by cspi_reset_stats().
    _cspi_stats_n_calls: int = 0
    _cspi_stats_n_switch: int = 0
    _cspi_stats_n_fallback_gate: int = 0
    _cspi_stats_n_fallback_margin: int = 0
    # Frozen-root-set statistics.
    _cspi_stats_root_total_safe: int = 0
    _cspi_stats_root_visits_sum: int = 0
    _cspi_stats_root_visits_min: int = 0
    _cspi_stats_margin_sum: float = 0.0
    _cspi_stats_margin_min: float = 0.0
    _cspi_stats_margin_max: float = 0.0

    def cspi_configure(
        self,
        margin_mult: float = 1.0,
        v_rollout_horizon: int = 0,
        v_rollout_n_samples: int = 1,
        e_plan_scale: float = 0.5,
        delta_piB: float = 0.0,
        frozen_root: bool = True,
        n_min_root_visits: int = 3,
    ) -> None:
        """Set the CSPI hyperparameters.

        Args:
            margin_mult: multiplier of the switch margin; the deployed rule switches
                when Q(a*) - Q(a_b) >= margin_mult * e_plan_scale.
            v_rollout_horizon: horizon of the baseline rollout; <= 0 uses cfg.max_depth.
            v_rollout_n_samples: number of baseline rollouts averaged (>= 1).
            e_plan_scale: planner-error scale, as a fraction of E_model.
            delta_piB: baseline suboptimality slack added to the theoretical margin.
            frozen_root: draw the root candidate set once and do not widen it.
            n_min_root_visits: minimum visits per safe root action in the frozen regime.
        """
        self._cspi_margin_mult = float(margin_mult)
        # A rollout horizon equal to max_depth keeps V_b and the MCTS Q estimates on
        # the same discounted-return scale.
        if int(v_rollout_horizon) <= 0:
            self._cspi_v_rollout_horizon = int(self.cfg.max_depth)
        else:
            self._cspi_v_rollout_horizon = int(v_rollout_horizon)
        self._cspi_v_rollout_n_samples = int(max(1, v_rollout_n_samples))
        self._cspi_e_plan_scale = float(e_plan_scale)
        self._cspi_delta_piB = float(delta_piB)
        self._cspi_frozen_root = bool(frozen_root)
        self._cspi_n_min_root_visits = int(n_min_root_visits)

    def cspi_reset_stats(self) -> None:
        """Reset all CSPI statistics counters."""
        self._cspi_stats_n_calls = 0
        self._cspi_stats_n_switch = 0
        self._cspi_stats_n_fallback_gate = 0
        self._cspi_stats_n_fallback_margin = 0
        self._cspi_stats_root_total_safe = 0
        self._cspi_stats_root_visits_sum = 0
        self._cspi_stats_root_visits_min = 0
        self._cspi_stats_margin_sum = 0.0
        self._cspi_stats_margin_min = float("inf")
        self._cspi_stats_margin_max = float("-inf")

    def cspi_stats(self) -> Dict[str, Any]:
        """Return switch/fallback counts and root-visit and margin statistics."""
        n_sw = max(1, int(self._cspi_stats_n_switch))
        n_safe_avg = (
            float(self._cspi_stats_root_total_safe) / float(max(1, self._cspi_stats_n_calls))
        )
        visits_avg = (
            float(self._cspi_stats_root_visits_sum) / float(max(1, self._cspi_stats_root_total_safe))
        )
        margin_avg = float(self._cspi_stats_margin_sum) / float(n_sw)
        return {
            "n_calls": int(self._cspi_stats_n_calls),
            "n_switch": int(self._cspi_stats_n_switch),
            "n_fallback_gate": int(self._cspi_stats_n_fallback_gate),
            "n_fallback_margin": int(self._cspi_stats_n_fallback_margin),
            "root_safe_actions_avg": float(n_safe_avg),
            "root_visits_per_safe_avg": float(visits_avg),
            "root_visits_min_over_calls": int(self._cspi_stats_root_visits_min),
            "margin_avg_on_switch": float(margin_avg),
            "margin_min_on_switch": (
                float(self._cspi_stats_margin_min)
                if self._cspi_stats_n_switch > 0 else 0.0
            ),
            "margin_max_on_switch": (
                float(self._cspi_stats_margin_max)
                if self._cspi_stats_n_switch > 0 else 0.0
            ),
        }

    def _cspi_delta_margin(self) -> float:
        """Theoretical switching margin mult * (E_model + E_plan + delta_piB).

        E_model = eps_threshold * (1 - gamma^H) / (1 - gamma) is the model-trust
        radius in discounted-return units, with the error-propagation constant absorbed
        into eps_threshold. E_plan = e_plan_scale * E_model is a heuristic bound on the
        finite-budget planner error. delta_piB >= 0 is the baseline suboptimality slack
        (default 0). Not used by the switch rule in ``act``.
        """
        gamma = float(self.cfg.gamma)
        H = int(self.cfg.max_depth)
        eps = float(self.cfg.eps_threshold)
        if abs(1.0 - gamma) < 1e-12:
            horizon_factor = float(H)
        else:
            horizon_factor = (1.0 - gamma ** H) / (1.0 - gamma)
        E_model = eps * horizon_factor
        E_plan = float(self._cspi_e_plan_scale) * E_model
        delta_piB = float(self._cspi_delta_piB)
        return float(self._cspi_margin_mult) * (E_model + E_plan + delta_piB)

    def _cspi_v_b_rollout(self, obs_root: Any) -> float:
        """Estimate V_b(s_root) with deterministic baseline rollouts in the learned model.

        Averages ``_cspi_v_rollout_n_samples`` rollouts of horizon
        ``_cspi_v_rollout_horizon``. A rollout stops on termination, at the horizon
        (tail value 0, as with ``leaf_value_mode="zero"``) or, with
        ``gate_inside_tree``, at the first unsafe baseline action. Requires ``_is_safe``
        on the host class. Not used by the switch rule in ``act``.
        """
        n_samples = int(max(1, self._cspi_v_rollout_n_samples))
        gamma = float(self.cfg.gamma)
        horizon = int(self._cspi_v_rollout_horizon)
        gate_inside = bool(self.cfg.gate_inside_tree)
        acc = 0.0
        for _s in range(n_samples):
            G = 0.0
            discount = 1.0
            obs = obs_root
            for _t in range(horizon):
                a, _ = self.ppo.predict(obs, deterministic=True)
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                if gate_inside:
                    safe, _, _ = self._is_safe(obs, a)
                    if not safe:
                        break
                obs_next, rew, terminated, truncated, _info = self.model_env.step(
                    obs, a, rng=self.rng
                )
                G += discount * float(rew)
                discount *= gamma
                if bool(terminated) or bool(truncated):
                    break
                obs = obs_next
            acc += G
        return float(acc / float(n_samples))

    def _cspi_baseline_action(self, obs: Any) -> np.ndarray:
        """Deterministic baseline action: PPO mean, clipped to [-1, 1]."""
        a, _ = self.ppo.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        return np.clip(a, -1.0, 1.0).astype(np.float32)


# =============================================================================
# Parametric CSPI planner
# =============================================================================


class OfflineMCTS_DPW_PPO_SPIBB_CSPI_Parametric(
    _CSPIMixin, OfflineMCTS_DPW_PPO_SPIBB_Parametric
):
    """Cal-SPI deployed policy on the reference planner.

    Adds the baseline gate check, the frozen-root regime with minimum root visits and
    the baseline-relative switch (see the module docstring). Expansion, PUCT selection
    and the in-tree gate are inherited.
    """

    def __init__(
        self,
        *args,
        cspi_margin_mult: float = 1.0,
        cspi_v_rollout_horizon: int = 0,  # 0 = max_depth
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

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a* if the baseline-relative switch accepts it, otherwise a_b."""
        self._cspi_stats_n_calls += 1

        # Low-data regime: B(u) is not constructible, return the baseline.
        if (
            bool(self.cfg.conservative_on_low_data)
            and bool(getattr(self, "calibrator", None) is not None)
            and bool(getattr(self.calibrator, "meta", {}).get("low_data_regime", False))
        ):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return self._clip_action(self._ppo_mean(obs))

        # Baseline gate check.
        a_b = self._cspi_baseline_action(obs)
        baseline_safe, _, _ = self._is_safe(obs, a_b)
        if not bool(baseline_safe):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return a_b

        # Root candidates: a_b at index 0, then the frozen PPO samples.
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
        ens = self._get_ensemble()
        if ens is not None and str(self.cfg.model_sampling) == "fixed_per_sim":
            E = getattr(getattr(ens, "cfg", None), "ensemble_size", 4)
            model_indices = self.rng.integers(0, int(E), size=int(self.cfg.num_sims))

        for s in range(int(self.cfg.num_sims)):
            mi = None
            if model_indices is not None:
                mi = int(model_indices[s])
            self._search(root, mi=mi)

        # Root selection on the safe set computed before the search.
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

        # Baseline-relative switch; a_b is root.actions[0].
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
        else:
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_margin += 1
            return a_b

    def _search(self, node: Node, mi: Optional[int]) -> float:
        """Parent search with no widening at a frozen root and forced minimum root visits."""
        node.N += 1

        if node.depth >= int(self.cfg.max_depth):
            return 0.0

        is_root = (node.depth == 0)
        if not (is_root and self._cspi_frozen_root):
            # APW as in the parent: non-root nodes, and the root when not frozen.
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

        # Forced minimum visits over the frozen safe root set.
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

        safe, _, _ = self._is_safe(node.obs, a)
        if self.use_gate and bool(self.cfg.gate_inside_tree) and (not safe):
            v = self._leaf_value(node.obs)
            node.Nsa[a_idx] += 1
            node.Q[a_idx] += (v - node.Q[a_idx]) / float(node.Nsa[a_idx])
            if bool(self.cfg.suppress_unsafe_actions):
                node.P[a_idx] = 0.0
            return float(v)

        rng = self.rng
        mi_step = mi
        if str(self.cfg.model_sampling) == "resample_each_step":
            ens = self._get_ensemble()
            E = getattr(getattr(ens, "cfg", None), "ensemble_size", 4) if ens is not None else 4
            mi_step = int(rng.integers(0, int(E)))

        obs_next, r, terminated, truncated, _info = self.model_env.step(
            node.obs, a, model_index=mi_step, rng=rng,
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
# Batched CSPI planner
# =============================================================================


class OfflineMCTS_DPW_PPO_Batched_CSPI(_CSPIMixin, OfflineMCTS_DPW_PPO_Batched):
    """Cal-SPI deployed policy on the batched planner.

    Same root logic as the parametric CSPI planner; gate decisions and transitions come
    from the per-node caches of the batched parent.
    """

    def __init__(
        self,
        *args,
        cspi_margin_mult: float = 1.0,
        cspi_v_rollout_horizon: int = 0,  # 0 = max_depth
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

    def _search(self, node: _BatchedNode, mi: int) -> float:
        """Batched search with the frozen-root regime of the parametric CSPI planner.

        Only expansion and selection differ from the parent; the gate check, cached
        transition and backup are the same.
        """
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

        # Forced minimum visits over the frozen safe root set.
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

        safe = node.safe_cache[a_idx] if self.use_gate else True
        if self.use_gate and bool(self.cfg.gate_inside_tree) and (not safe):
            v = self._leaf_value(node.obs)
            node.Nsa[a_idx] += 1
            node.Q[a_idx] += (v - node.Q[a_idx]) / float(node.Nsa[a_idx])
            if bool(self.cfg.suppress_unsafe_actions):
                node.P[a_idx] = 0.0
            return float(v)

        obs_next = node.next_obs_cache[a_idx][mi]
        r = float(node.reward_cache[a_idx][mi])
        terminated = bool(node.terminated_cache[a_idx][mi])

        child = node.children[a_idx]
        if child is None:
            child = _BatchedNode(obs=obs_next, depth=node.depth + 1)
            node.children[a_idx] = child

        if terminated:
            G = r
        else:
            G = r + float(self.cfg.gamma) * float(self._search(child, mi=mi))

        node.Nsa[a_idx] += 1
        node.Q[a_idx] += (G - node.Q[a_idx]) / float(node.Nsa[a_idx])
        return float(G)

    def act(self, obs: Any) -> np.ndarray:
        """Return a* if the baseline-relative switch accepts it, otherwise a_b."""
        self._cspi_stats_n_calls += 1

        # Low-data regime: return the baseline without searching.
        if (
            bool(self.cfg.conservative_on_low_data)
            and self.calibrator is not None
            and bool(getattr(self.calibrator, "meta", {}).get("low_data_regime", False))
        ):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return self._clip_action(self._ppo_mean(obs))

        # Baseline gate check.
        a_b = self._cspi_baseline_action(obs)
        baseline_safe, _, _ = self._is_safe_single(obs, a_b)
        if not bool(baseline_safe):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return a_b

        # Root candidates: a_b at index 0.
        root = _BatchedNode(obs=obs, depth=0)
        root.actions.append(a_b)
        root.P.append(1.0)
        root.Q.append(0.0)
        root.Nsa.append(0)
        root.children.append(None)
        self._populate_cache(root, [a_b])

        # Frozen root: draw the whole root candidate set before the simulations, so
        # A_safe(s) is fixed for the entire search.
        if self._cspi_frozen_root:
            K = int(self.cfg.n_actions_root)
            if K > len(root.actions):
                self._expand_action(root, K - len(root.actions))

        if self.use_gate:
            root_safe_ids = [i for i in range(len(root.actions)) if root.safe_cache[i]]
        else:
            root_safe_ids = list(range(len(root.actions)))
        self._cspi_root_safe_ids = root_safe_ids
        self._cspi_stats_root_total_safe += len(root_safe_ids)

        if root_safe_ids:
            self._cspi_n_min_root_visits_eff = max(1, min(
                int(self._cspi_n_min_root_visits),
                int(self.cfg.num_sims) // len(root_safe_ids),
            ))
        else:
            self._cspi_n_min_root_visits_eff = 0

        model_indices = self.rng.integers(0, self._E, size=int(self.cfg.num_sims))
        for s in range(int(self.cfg.num_sims)):
            mi = int(model_indices[s])
            self._search(root, mi=mi)

        if self._cspi_frozen_root:
            safe_ids = root_safe_ids
        elif self.use_gate:
            safe_ids = [i for i in range(len(root.actions)) if root.safe_cache[i]]
        else:
            safe_ids = list(range(len(root.actions)))

        self.last_root_coverage = float(len(safe_ids)) / float(max(1, len(root.actions)))

        if len(safe_ids) == 0:
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_gate += 1
            return self._clip_action(self._ppo_mean(obs))

        tol = 1e-9
        q_best = max(root.Q[j] for j in safe_ids)
        cands = [j for j in safe_ids if (q_best - root.Q[j]) <= tol]
        best_i = max(cands, key=lambda j: root.Nsa[j])
        a_star = np.asarray(root.actions[best_i], dtype=np.float32)

        # Baseline-relative switch; a_b is root.actions[0].
        q_b = float(root.Q[0])
        eta_op = float(self._cspi_margin_mult) * float(self._cspi_e_plan_scale)
        margin_observed = float(q_best) - float(q_b)
        if margin_observed >= eta_op:
            self.last_root_fallback = False
            self._cspi_stats_n_switch += 1
            return a_star
        else:
            self.last_root_fallback = True
            self._cspi_stats_n_fallback_margin += 1
            return a_b

    def _is_safe_single(self, obs: Any, action: np.ndarray):
        """Gate check on one (obs, action) pair through the batched cache routine.

        Returns ``(safe, u, None)``, matching the shape of the parametric ``_is_safe``.
        """
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        if not self.use_gate:
            return True, 0.0, None
        # Evaluate on a scratch node to reuse the cache logic.
        tmp = _BatchedNode(obs=obs, depth=0)
        tmp.actions.append(a)
        self._populate_cache(tmp, [a])
        safe = bool(tmp.safe_cache[0])
        u = float(tmp.u_cache[0]) if hasattr(tmp, "u_cache") else 0.0
        return safe, u, None
