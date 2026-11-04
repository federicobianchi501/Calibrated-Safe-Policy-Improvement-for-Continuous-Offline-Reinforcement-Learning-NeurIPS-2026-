# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Batched-evaluation variant of the reference MCTS-DPW planner.

``OfflineMCTS_DPW_PPO_Batched`` has the same constructor, ``act`` method and logging
attributes as ``OfflineMCTS_DPW_PPO_SPIBB_Parametric``. Instead of querying the
ensemble once per (s, a) visit, it runs one batched forward pass when actions are
added to a node and caches, per action a_k:
    u_cache[k]              uncertainty score u(s, a_k)
    safe_cache[k]           gate decision for (s, a_k)
    next_obs_cache[k][mi]   next observation predicted by member mi
    reward_cache[k][mi]     reward of that transition
    terminated_cache[k][mi] termination flag (unhealthy state or non-finite prediction)
Later visits to (s, a_k) read from the cache without calling the ensemble.

Equivalence with the reference planner: for ``model_sampling="fixed_per_sim"`` (the
only supported mode) and the same seed, config, ensemble, calibrator, PPO policy and
observation, both planners return identical actions, because
  1. the RNG is consumed in the same order: ``rng.integers(0, E, num_sims)`` at the
     start of ``act``, then ``rng.standard_normal`` in ``_sample_ppo`` per expansion;
  2. member outputs on a batch equal per-sample outputs for stateless MLPs
     (no BatchNorm or Dropout);
  3. PUCT selection and root tie-breaking use the same statistics, tolerance and key;
  4. with an explicit model index the model step draws no random numbers, so caching
     skips no RNG call.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from safe_rl.algorithms.offline.cal_spi.mcts_parametric import (
    OfflineMCTSConfig,
)


class _BatchedNode:
    """Tree node with per-action precomputed caches."""

    __slots__ = (
        "obs", "depth", "N",
        "actions", "children", "P", "Q", "Nsa",
        "u_cache", "safe_cache",
        "next_obs_cache", "reward_cache", "terminated_cache",
    )

    def __init__(self, obs: Any, depth: int):
        self.obs = obs
        self.depth = int(depth)
        self.N = 0
        self.actions: List[np.ndarray] = []
        self.children: List[Optional["_BatchedNode"]] = []
        self.P: List[float] = []
        self.Q: List[float] = []
        self.Nsa: List[int] = []
        # Per-action caches, aligned with self.actions.
        self.u_cache: List[float] = []
        self.safe_cache: List[bool] = []
        self.next_obs_cache: List[np.ndarray] = []   # each entry: [E, obs_dim]
        self.reward_cache: List[np.ndarray] = []     # each entry: [E]
        self.terminated_cache: List[np.ndarray] = [] # each entry: [E] bool


class OfflineMCTS_DPW_PPO_Batched:
    """Batched-evaluation planner with the interface of the reference planner.

    Supports only ``model_sampling="fixed_per_sim"``; see the module docstring for the
    equivalence conditions.
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
        if str(cfg.model_sampling) != "fixed_per_sim":
            raise NotImplementedError(
                f"Batched planner only supports model_sampling='fixed_per_sim', "
                f"got '{cfg.model_sampling}'. Use the original planner for this mode."
            )
        if str(cfg.gate_mode) not in ("calibrated", "uncertainty"):
            raise ValueError(f"Unknown gate_mode={cfg.gate_mode}")
        if str(cfg.uncertainty_type) not in ("usad", "epistemic", "total"):
            raise ValueError(f"Unknown uncertainty_type={cfg.uncertainty_type}")

        self.model_env = model_env
        self.ppo = ppo_policy
        self.cfg = cfg
        self.calibrator = calibrator
        self.use_gate = bool(use_gate)
        self.use_ppo_prior = bool(use_ppo_prior)
        self.use_ppo_leaf_value = bool(use_ppo_leaf_value)
        self.rng = np.random.default_rng(int(rng_seed))

        self.last_root_fallback: bool = False
        self.last_root_coverage: float = 0.0
        self._debug_once: bool = False

        # model_env.ens may be a flat-obs wrapper; the raw ensemble is then at .ens._ens.
        wrapper = self.model_env.ens
        self._ens = getattr(wrapper, "_ens", wrapper)
        self._E = int(getattr(self._ens.cfg, "ensemble_size", 4))
        self._obs_dim = int(self._ens.obs_dim)
        self._act_dim = int(self._ens.act_dim)
        self._device = self._ens.device

    # ------------------------------------------------------------------
    # PPO helpers (same RNG consumption as the reference planner)
    # ------------------------------------------------------------------
    def _ppo_mean(self, obs: Any) -> np.ndarray:
        """Deterministic PPO action a_b(s) as a flat float32 vector."""
        a, _ = self.ppo.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    def _sample_ppo(self, obs: Any, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n clipped actions from N(a_b(s), sigma^2 I); returns ``(actions, priors)``."""
        mean = self._clip_action(self._ppo_mean(obs))
        act_dim = int(mean.shape[0])
        std = float(self.cfg.ppo_std_scale)
        acts = mean[None, :] + std * self.rng.standard_normal(size=(n, act_dim)).astype(np.float32)
        acts = np.clip(acts, -1.0, 1.0).astype(np.float32)
        d = np.linalg.norm(acts - mean[None, :], axis=-1)
        priors = np.exp(-0.5 * (d ** 2) / max(1e-6, std ** 2)).astype(np.float32)
        priors = priors / (priors.sum() + 1e-8)
        return acts, priors

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip an action to [-1, 1]."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _leaf_value(self, obs: Any) -> float:
        """PPO value estimate, or 0 when PPO leaf values are disabled or unavailable."""
        if (not self.use_ppo_leaf_value) or (str(self.cfg.leaf_value_mode) == "zero"):
            return 0.0
        try:
            obs_t, _ = self.ppo.policy.obs_to_tensor(obs)
            with torch.no_grad():
                v = self.ppo.policy.predict_values(obs_t)
            return float(v.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Batched ensemble forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _batched_predict_all(
        self,
        obs: Any,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Ensemble delta predictions for K actions at one observation.

        Args:
            obs: flat observation, or dict with an ``"observation"`` key.
            actions: array of shape [K, act_dim].

        Returns:
            Array of shape [E, K, obs_dim]: delta prediction per member and action.
        """
        if isinstance(obs, dict):
            obs_arr = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
        else:
            obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self._act_dim)
        K = int(actions.shape[0])

        obs_batch = np.broadcast_to(obs_arr[None, :], (K, self._obs_dim)).copy()
        X = np.concatenate([obs_batch, actions], axis=-1).astype(np.float32)
        if self._ens.input_mean is not None:
            X = (X - self._ens.input_mean) / self._ens.input_std

        xt = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        mus = np.empty((self._E, K, self._obs_dim), dtype=np.float32)
        for k_idx, m in enumerate(self._ens.ensemble):
            mus[k_idx] = m(xt).detach().cpu().numpy()
        return mus

    @staticmethod
    def _uncertainty_usad_batch(mus: np.ndarray) -> np.ndarray:
        """USAD score: max pairwise L2 distance between members. [E, K, d] -> [K]."""
        E = mus.shape[0]
        K = mus.shape[1]
        out = np.zeros(K, dtype=np.float32)
        for i in range(E):
            for j in range(i + 1, E):
                d = np.linalg.norm(mus[i] - mus[j], axis=-1)  # [K]
                np.maximum(out, d, out=out)
        return out

    @staticmethod
    def _uncertainty_epistemic_batch(mus: np.ndarray) -> np.ndarray:
        """Epistemic score: per-dimension variance across members, averaged over
        dimensions. [E, K, d] -> [K]."""
        return mus.var(axis=0).mean(axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Gate decision (same logic as the reference planner)
    # ------------------------------------------------------------------
    def _safe_from_u(self, u: float) -> bool:
        """Gate decision from a precomputed score u; unsafe in the low-data regime."""
        if not self.use_gate:
            return True
        if str(self.cfg.gate_mode) == "uncertainty":
            return bool(u <= float(self.cfg.uncertainty_threshold))
        if self.calibrator is None:
            raise ValueError("gate_mode='calibrated' requires calibrator")
        if (
            bool(self.calibrator.meta.get("low_data_regime", False))
            and bool(self.cfg.conservative_on_low_data)
        ):
            if not self._debug_once:
                print("[BATCHED DBG] low_data_regime=1 -> forcing PPO fallback", flush=True)
                self._debug_once = True
            return False
        b = float(self.calibrator.bound(u))
        if not np.isfinite(b):
            return False
        return bool(b <= float(self.cfg.eps_threshold))

    # ------------------------------------------------------------------
    # Cache population
    # ------------------------------------------------------------------
    def _populate_cache(self, node: _BatchedNode, new_actions: List[np.ndarray]) -> None:
        """Run one batched ensemble forward for the new actions and fill the node caches."""
        K = len(new_actions)
        if K == 0:
            return

        acts_arr = np.stack(new_actions, axis=0).astype(np.float32)
        mus = self._batched_predict_all(node.obs, acts_arr)  # [E, K, obs_dim]

        ut = str(self.cfg.uncertainty_type)
        if ut == "usad":
            u_arr = self._uncertainty_usad_batch(mus)
        else:
            # "epistemic" and "total" both use the member variance of the means.
            u_arr = self._uncertainty_epistemic_batch(mus)

        if isinstance(node.obs, dict):
            obs_flat = np.asarray(node.obs["observation"], dtype=np.float32).reshape(-1)
        else:
            obs_flat = np.asarray(node.obs, dtype=np.float32).reshape(-1)

        env_cfg = self.model_env.cfg
        obs_clip = float(getattr(env_cfg, "obs_clip", 0.0) or 0.0)
        terminate_on_nan = bool(getattr(env_cfg, "terminate_on_nan", True))

        for k in range(K):
            u = float(u_arr[k])
            safe = self._safe_from_u(u)

            # Next observation per member, [E, obs_dim].
            next_obs_per_mi = obs_flat[None, :] + mus[:, k, :]
            next_obs_per_mi = next_obs_per_mi.astype(np.float32)
            if obs_clip > 0:
                next_obs_per_mi = np.clip(next_obs_per_mi, -obs_clip, obs_clip)

            # Same clipping as model_env.step.
            a_clipped = np.clip(new_actions[k], -1.0, 1.0).astype(np.float32)

            r_per_mi = np.zeros(self._E, dtype=np.float32)
            t_per_mi = np.zeros(self._E, dtype=bool)
            for mi in range(self._E):
                obs_next_mi = next_obs_per_mi[mi]
                ok = bool(np.isfinite(obs_next_mi).all())
                if (not ok) and terminate_on_nan:
                    r_per_mi[mi] = 0.0
                    t_per_mi[mi] = True
                    continue
                r_per_mi[mi] = float(self.model_env._reward(obs_next_mi, a_clipped))
                t_per_mi[mi] = bool(not self.model_env._is_healthy(obs_next_mi))

            node.u_cache.append(u)
            node.safe_cache.append(safe)
            node.next_obs_cache.append(next_obs_per_mi)
            node.reward_cache.append(r_per_mi)
            node.terminated_cache.append(t_per_mi)

    # ------------------------------------------------------------------
    # MCTS internals
    # ------------------------------------------------------------------
    def _expand_action(self, node: _BatchedNode, n_actions: int) -> None:
        """Append ``n_actions`` PPO-sampled actions to ``node`` and cache their evaluations."""
        acts, priors = self._sample_ppo(node.obs, n_actions)
        new_actions: List[np.ndarray] = []
        for a, p in zip(acts, priors):
            a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
            node.actions.append(a_arr)
            node.P.append(float(p))
            node.Q.append(0.0)
            node.Nsa.append(0)
            node.children.append(None)
            new_actions.append(a_arr)
        self._populate_cache(node, new_actions)

    def _select_child(self, node: _BatchedNode) -> int:
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

    def _search(self, node: _BatchedNode, mi: int) -> float:
        """Run one simulation below ``node`` with ensemble member ``mi``; value 0 at max depth."""
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

        safe = node.safe_cache[a_idx] if self.use_gate else True
        if self.use_gate and bool(self.cfg.gate_inside_tree) and (not safe):
            v = self._leaf_value(node.obs)
            node.Nsa[a_idx] += 1
            node.Q[a_idx] += (v - node.Q[a_idx]) / float(node.Nsa[a_idx])
            if bool(self.cfg.suppress_unsafe_actions):
                node.P[a_idx] = 0.0
            return float(v)

        # Model step from the cache.
        obs_next = node.next_obs_cache[a_idx][mi]
        r = float(node.reward_cache[a_idx][mi])
        terminated = bool(node.terminated_cache[a_idx][mi])
        truncated = False  # no time-limit truncation inside the tree

        child = node.children[a_idx]
        if child is None:
            child = _BatchedNode(obs=obs_next, depth=node.depth + 1)
            node.children[a_idx] = child

        if terminated or truncated:
            G = r
        else:
            G = r + float(self.cfg.gamma) * float(self._search(child, mi=mi))

        node.Nsa[a_idx] += 1
        node.Q[a_idx] += (G - node.Q[a_idx]) / float(node.Nsa[a_idx])
        return float(G)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def act(self, obs: Any) -> np.ndarray:
        """Plan from ``obs`` and return the selected action (PPO mean on fallback)."""
        # Low-data regime: return the baseline without searching.
        if (
            bool(self.cfg.conservative_on_low_data)
            and self.calibrator is not None
            and bool(getattr(self.calibrator, "meta", {}).get("low_data_regime", False))
        ):
            self.last_root_coverage = 0.0
            self.last_root_fallback = True
            return self._clip_action(self._ppo_mean(obs))

        root = _BatchedNode(obs=obs, depth=0)

        # The PPO mean is the first root candidate.
        try:
            a_ppo = self._clip_action(self._ppo_mean(obs))
            root.actions.append(a_ppo)
            root.P.append(1.0)
            root.Q.append(0.0)
            root.Nsa.append(0)
            root.children.append(None)
            self._populate_cache(root, [a_ppo])
        except Exception:
            pass

        # One ensemble member per simulation, drawn before any action sampling.
        model_indices = self.rng.integers(0, self._E, size=int(self.cfg.num_sims))

        for s in range(int(self.cfg.num_sims)):
            mi = int(model_indices[s])
            self._search(root, mi=mi)

        # Root selection over safe actions (cached flags); PPO mean if none is safe.
        if self.use_gate:
            safe_ids = [i for i in range(len(root.actions)) if root.safe_cache[i]]
        else:
            safe_ids = list(range(len(root.actions)))

        self.last_root_coverage = float(len(safe_ids)) / float(max(1, len(root.actions)))

        if len(safe_ids) == 0:
            self.last_root_fallback = True
            return self._clip_action(self._ppo_mean(obs))

        self.last_root_fallback = False
        tol = 1e-9
        q_best = max(root.Q[j] for j in safe_ids)
        cands = [j for j in safe_ids if (q_best - root.Q[j]) <= tol]
        best_i = max(cands, key=lambda j: root.Nsa[j])

        return np.asarray(root.actions[best_i], dtype=np.float32)
