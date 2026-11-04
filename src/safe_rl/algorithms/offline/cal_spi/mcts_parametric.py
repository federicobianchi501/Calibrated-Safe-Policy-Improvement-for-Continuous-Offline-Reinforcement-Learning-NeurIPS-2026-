# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Reference MCTS-DPW planner over a learned ensemble model, guided by a PPO baseline.

Provides the planner configuration, the tree node, the PPO-guided planner with a
calibrated (B(u) <= eps) or raw-uncertainty (u <= tau) gate and fallback to the
baseline action, and an ungated, prior-free planner used as an unprotected baseline.

Actions are clipped to [-1, 1] before computing the uncertainty score and the gate
and before stepping the model env, so the gate sees the same action as the dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OfflineMCTSConfig:
    """Planner hyperparameters shared by all MCTS-DPW variants."""

    # MCTS/DPW
    num_sims: int = 200
    max_depth: int = 20
    gamma: float = 0.99
    dpw_alpha: float = 0.5
    dpw_k: float = 1.0
    c_puct: float = 1.5

    # Action sampling
    n_actions_root: int = 64
    n_actions_per_node: int = 16
    ppo_std_scale: float = 0.1

    # Leaf value
    leaf_value_mode: str = "ppo"  # "ppo" or "zero"

    # Gate
    gate_mode: str = "calibrated"        # "calibrated" | "uncertainty"
    gate_inside_tree: bool = True
    uncertainty_type: str = "epistemic"  # "epistemic" | "total" | "usad"
    uncertainty_threshold: float = 1.0   # τ for uncertainty gate u <= τ
    eps_threshold: float = 0.10          # ε for calibrated gate B(u) <= ε
    conservative_on_low_data: bool = True
    suppress_unsafe_actions: bool = False  # set P[a] = 0 for unsafe actions inside the tree

    # Model sampling
    model_sampling: str = "fixed_per_sim"  # "fixed_per_sim" | "resample_each_step"


class Node:
    """Search-tree node with per-edge action, prior, mean return, visit count and child."""

    __slots__ = ("obs", "depth", "N", "actions", "children", "P", "Q", "Nsa")

    def __init__(self, obs: Dict[str, np.ndarray], depth: int):
        self.obs = obs
        self.depth = depth
        self.N = 0
        self.actions: List[np.ndarray] = []
        self.children: List[Optional["Node"]] = []
        self.P: List[float] = []
        self.Q: List[float] = []
        self.Nsa: List[int] = []


class OfflineMCTS_DPW_PPO_SPIBB_Parametric:
    """MCTS-DPW planner with PPO prior, safe gate and baseline fallback at the root.

    Root candidates are the PPO mean plus actions sampled from N(a_b(s), sigma^2 I).
    With ``gate_inside_tree``, an unsafe edge ends the simulation with the leaf value
    of its node. At the root the best safe action by Q is returned (ties broken by
    visit count); if no root action is safe, the PPO mean is returned. In the low-data
    regime with ``conservative_on_low_data`` the planner returns the PPO mean without
    searching (conservative collapse).

    After each ``act`` call, ``last_root_fallback`` and ``last_root_coverage`` (fraction
    of safe root actions) are set for evaluation logging.
    """

    def __init__(
        self,
        model_env: Any,
        ppo_policy: Any,
        cfg: OfflineMCTSConfig,
        calibrator: Optional[Any] = None,
        use_gate: bool = True,
        use_ppo_prior: bool = True,
        use_ppo_leaf_value: bool = True,
        rng_seed: int = 0,
    ):
        self.model_env = model_env
        self.ppo = ppo_policy
        self.cfg = cfg
        self.calibrator = calibrator
        self.use_gate = bool(use_gate)
        self.use_ppo_prior = bool(use_ppo_prior)
        self.use_ppo_leaf_value = bool(use_ppo_leaf_value)
        self.rng = np.random.default_rng(int(rng_seed))

        # Evaluation logging.
        self.last_root_fallback: bool = False
        self.last_root_coverage: float = 0.0
        self._debug_once = False

    # ---------- PPO helpers ----------
    def _ppo_mean(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Deterministic PPO action a_b(s) as a flat float32 vector."""
        a, _ = self.ppo.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    def _sample_ppo(self, obs: Dict[str, np.ndarray], n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n actions from N(a_b(s), sigma^2 I), clipped to [-1, 1].

        Returns ``(actions, priors)``; priors are Gaussian weights in the distance to
        the PPO mean, normalised to sum to one, and are used as P in PUCT.
        """

        mean = self._clip_action(self._ppo_mean(obs))
        act_dim = int(mean.shape[0])

        std = float(self.cfg.ppo_std_scale)
        acts = mean[None, :] + std * self.rng.standard_normal(size=(n, act_dim)).astype(np.float32)

        # Clip before storing so every tree action is in range.
        acts = np.clip(acts, -1.0, 1.0).astype(np.float32)

        # Priors are computed on the clipped actions.
        d = np.linalg.norm(acts - mean[None, :], axis=-1)
        priors = np.exp(-0.5 * (d ** 2) / max(1e-6, std ** 2)).astype(np.float32)
        priors = priors / (priors.sum() + 1e-8)
        return acts, priors

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip an action to [-1, 1].

        The model env clips internally; clipping before the uncertainty query keeps
        the gate evaluated on the same action the dynamics receive.
        """
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _leaf_value(self, obs: Dict[str, np.ndarray]) -> float:
        """PPO value estimate V_b(s), or 0 when PPO leaf values are disabled or unavailable.

        Used for empty nodes and gated edges; the max-depth cutoff always returns 0.
        """
        if (not self.use_ppo_leaf_value) or (str(self.cfg.leaf_value_mode) == "zero"):
            return 0.0
        try:
            import torch as th
            obs_t, _ = self.ppo.policy.obs_to_tensor(obs)
            with th.no_grad():
                v = self.ppo.policy.predict_values(obs_t)
            return float(v.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            return 0.0

    # ---------- Ensemble access ----------
    def _get_ensemble(self) -> Optional[Any]:
        """Return the dynamics ensemble (``model_env.ensemble`` or ``model_env.ens``)."""
        if hasattr(self.model_env, "ensemble"):
            return getattr(self.model_env, "ensemble")
        if hasattr(self.model_env, "ens"):
            return getattr(self.model_env, "ens")
        return None

    # ---------- Uncertainty ----------
    def _uncertainty_raw(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> float:
        """Uncertainty score u(s, a) of type ``cfg.uncertainty_type``; 0 without an ensemble."""
        ens = self._get_ensemble()
        if ens is None:
            return 0.0

        ut = str(self.cfg.uncertainty_type)
        if ut == "epistemic":
            return float(ens.uncertainty_epistemic(obs, action))
        if ut == "total":
            return float(ens.uncertainty_total(obs, action))
        if ut == "usad":
            if not hasattr(ens, "uncertainty_usad"):
                raise AttributeError("Ensemble missing uncertainty_usad for uncertainty_type='usad'")
            return float(ens.uncertainty_usad(obs, action))
        raise ValueError(f"Unknown uncertainty_type={ut}")

    def _is_safe(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> Tuple[bool, float, Optional[float]]:
        """Evaluate the gate on (s, a).

        Returns ``(safe, u, bound)``:
          - gate disabled: always safe, bound None;
          - ``gate_mode="uncertainty"``: safe iff u <= tau, bound None;
          - ``gate_mode="calibrated"``: safe iff B(u) <= eps, bound B(u). In the low-data
            regime (with ``conservative_on_low_data``) or for a non-finite bound, the
            action is unsafe and bound is inf.
        """
        action = self._clip_action(action)
        u = self._uncertainty_raw(obs, action)

        if not self.use_gate:
            return True, u, None

        if str(self.cfg.gate_mode) == "uncertainty":
            safe = (u <= float(self.cfg.uncertainty_threshold))
            return bool(safe), float(u), None

        if str(self.cfg.gate_mode) == "calibrated":
            if self.calibrator is None:
                raise ValueError("gate_mode='calibrated' requires calibrator")
            if bool(self.calibrator.meta.get("low_data_regime", False)) and bool(self.cfg.conservative_on_low_data):
                if self._debug_once is False:
                    print(f"[PLANNER DBG] low_data_regime=1 -> forcing PPO fallback", flush=True)
                    self._debug_once = True
                return False, float(u), float("inf")

            b = float(self.calibrator.bound(u))
            if not np.isfinite(b):
                return False, float(u), float("inf")
            safe = (b <= float(self.cfg.eps_threshold))
            return bool(safe), float(u), float(b)

        raise ValueError(f"Unknown gate_mode={self.cfg.gate_mode}")

    # ---------- MCTS core ----------
    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Plan from ``obs`` and return the selected action (PPO mean on fallback)."""
        # Low-data regime: B(u) is not constructible, return the baseline without searching.
        if bool(self.cfg.conservative_on_low_data) and bool(getattr(self, "calibrator", None) is not None) \
        and bool(getattr(self.calibrator, "meta", {}).get("low_data_regime", False)):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            return self._clip_action(self._ppo_mean(obs))
        
        root = Node(obs=obs, depth=0)
        try:
            a_ppo = self._clip_action(self._ppo_mean(obs))
            if not hasattr(root, "actions"):
                raise RuntimeError("Node has no actions field")
            root.actions.append(a_ppo)
            root.P.append(1.0)
            root.Q.append(0.0)
            root.Nsa.append(0)
            root.children.append(None)
        except Exception:
            a_ppo = None

        # fixed_per_sim (TS-inf): one ensemble member per simulation, all indices drawn
        # up front, before any action sampling.
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

        # Restrict root selection to safe actions; fall back to the PPO mean if none.
        safe_ids: List[int] = []
        for i, a in enumerate(root.actions):
            safe, _, _ = self._is_safe(root.obs, a)
            if safe:
                safe_ids.append(i)

        self.last_root_coverage = float(len(safe_ids)) / float(max(1, len(root.actions)))

        if len(safe_ids) == 0:
            self.last_root_fallback = True
            return self._clip_action(self._ppo_mean(obs))

        self.last_root_fallback = False
        tol = 1e-9
        # Maximise Q; ties within tol are broken by visit count.
        q_best = max(root.Q[j] for j in safe_ids)
        cands = [j for j in safe_ids if (q_best - root.Q[j]) <= tol]
        best_i = max(cands, key=lambda j: root.Nsa[j])

        return np.asarray(root.actions[best_i], dtype=np.float32)

    def _expand_action(self, node: Node, n_actions: int) -> None:
        """Append ``n_actions`` PPO-sampled (already clipped) actions to ``node``."""
        acts, priors = self._sample_ppo(node.obs, n_actions)
        for a, p in zip(acts, priors):
            node.actions.append(np.asarray(a, dtype=np.float32).reshape(-1))
            node.P.append(float(p))
            node.Q.append(0.0)
            node.Nsa.append(0)
            node.children.append(None)

    def _select_child(self, node: Node) -> int:
        """PUCT selection: argmax_i Q[i] + c_puct * P[i] * sqrt(N) / max(1, Nsa[i])."""
        N = float(max(1, node.N))
        best_i = -1
        best_score = -1e30

        for i in range(len(node.actions)):
            nsa = float(max(1, node.Nsa[i]))
            uct = float(self.cfg.c_puct) * float(node.P[i]) * np.sqrt(N) / nsa
            score = float(node.Q[i]) + float(uct)
            if score > best_score:
                best_score = score
                best_i = i

        return int(best_i)

    def _search(self, node: Node, mi: Optional[int]) -> float:
        """Run one simulation below ``node`` with ensemble member ``mi``; return its return.

        The value at ``max_depth`` is 0, not the PPO value.
        """
        node.N += 1

        if node.depth >= int(self.cfg.max_depth):
            return 0.0

        # Action progressive widening: |A(s)| <= min(floor(k * N^alpha), cap), at most
        # 8 new actions per visit.
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

        # Unsafe edge inside the tree: back up the leaf value and stop the simulation.
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
            node.obs,
            a,
            model_index=mi_step,
            rng=rng,
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
# Unprotected baseline: model-based MCTS-DPW without gate and without PPO
# =============================================================================

class OfflineMCTS_DPW_Ensemble_UNSAFE:
    """MCTS-DPW on the learned ensemble model with no gate and no baseline.

    Actions are sampled uniformly in [action_low, action_high] with uniform priors,
    leaf values are 0, and the root returns the best-Q expanded action with no
    fallback. Model sampling follows ``cfg.model_sampling`` as in the gated planner.
    Unconstrained, it can exploit model errors outside the data support.
    """

    def __init__(
        self,
        model_env: Any,
        cfg: OfflineMCTSConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        rng_seed: int = 0,
    ):
        self.model_env = model_env
        self.cfg = cfg
        self.rng = np.random.default_rng(int(rng_seed))

        low = np.asarray(action_low, dtype=np.float32).ravel()
        high = np.asarray(action_high, dtype=np.float32).ravel()
        if low.shape != high.shape:
            raise ValueError("action_low and action_high must have the same shape")
        self.action_low = low
        self.action_high = high
        self.act_dim = int(low.shape[0])

        # Same logging interface as the gated planner; no gate, so never a fallback
        # and coverage is always 1.
        self.last_root_fallback: bool = False
        self.last_root_coverage: float = 1.0

    def _get_ensemble(self) -> Optional[Any]:
        """Return the dynamics ensemble (``model_env.ensemble`` or ``model_env.ens``)."""
        if hasattr(self.model_env, "ensemble"):
            return getattr(self.model_env, "ensemble")
        if hasattr(self.model_env, "ens"):
            return getattr(self.model_env, "ens")
        return None

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip an action to [action_low, action_high]."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(a, self.action_low, self.action_high).astype(np.float32)

    def _sample_uniform_actions(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n uniform actions; returns ``(actions, uniform priors)``."""
        n = int(n)
        acts = self.rng.uniform(low=self.action_low, high=self.action_high, size=(n, self.act_dim)).astype(np.float32)
        pri = np.full((n,), 1.0 / max(1, n), dtype=np.float32)
        return acts, pri

    def _leaf_value(self, _obs: Dict[str, np.ndarray]) -> float:
        """Leaf value, always 0."""
        return 0.0

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Plan from ``obs`` and return the best-Q root action."""
        root = Node(obs=obs, depth=0)

        # fixed_per_sim: one ensemble member per simulation, drawn up front.
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

        # argmax Q over expanded root actions; a uniform action if the root is empty.
        if len(root.actions) == 0:
            a = self._sample_uniform_actions(1)[0][0]
            self.last_root_fallback = False
            self.last_root_coverage = 1.0
            return a

        best_i = int(np.argmax(np.asarray(root.Q, dtype=np.float32)))
        self.last_root_fallback = False
        self.last_root_coverage = 1.0
        return np.asarray(root.actions[best_i], dtype=np.float32)

    def _expand_action(self, node: Node, n_actions: int) -> None:
        """Append ``n_actions`` uniformly sampled actions to ``node``."""
        acts, priors = self._sample_uniform_actions(n_actions)
        for a, p in zip(acts, priors):
            a = self._clip_action(a)
            node.actions.append(a)
            node.P.append(float(p))
            node.Q.append(0.0)
            node.Nsa.append(0)
            node.children.append(None)

    def _select_child(self, node: Node) -> int:
        """PUCT selection, as in the gated planner."""
        N = float(max(1, node.N))
        best_i = -1
        best_score = -1e30
        for i in range(len(node.actions)):
            nsa = float(max(1, node.Nsa[i]))
            uct = float(self.cfg.c_puct) * float(node.P[i]) * np.sqrt(N) / nsa
            score = float(node.Q[i]) + float(uct)
            if score > best_score:
                best_score = score
                best_i = i
        return int(best_i)

    def _search(self, node: Node, mi: Optional[int]) -> float:
        """Run one simulation below ``node`` with ensemble member ``mi``; return its return."""
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

        rng = self.rng
        mi_step = mi
        if str(self.cfg.model_sampling) == "resample_each_step":
            ens = self._get_ensemble()
            E = getattr(getattr(ens, "cfg", None), "ensemble_size", 4) if ens is not None else 4
            mi_step = int(rng.integers(0, int(E)))

        obs_next, r, terminated, truncated, _info = self.model_env.step(
            node.obs,
            a,
            model_index=mi_step,
            rng=rng,
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