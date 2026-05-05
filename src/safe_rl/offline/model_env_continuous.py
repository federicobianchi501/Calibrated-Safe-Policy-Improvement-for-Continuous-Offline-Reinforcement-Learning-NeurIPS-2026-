# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
model_env_continuous.py — Learned dynamics model for flat-obs MuJoCo envs.

Analogous to ModelPointMazeEnv but for Walker2d, Hopper (no dict obs, no goal).

Key design:
  - Observations are flat numpy arrays.
  - The ensemble (EnsembleDynamicsSB3) expects dict obs with "observation" key.
    This module wraps flat obs as {"observation": obs} internally via EnsembleFlatWrapper.
  - Reward is reconstructed from predicted obs using known locomotion reward formulas.
  - Termination uses standard health conditions (z_height, angle bounds).
  - MCTS planner sees flat obs throughout; the wrapper handles dict conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from safe_rl.offline.ensemble_dynamics_sb3 import EnsembleDynamicsSB3


class EnsembleFlatWrapper:
    """
    Adapter: wraps EnsembleDynamicsSB3 so it accepts flat numpy obs
    (in addition to dict obs) without modifying the original class.

    The MCTS planner accesses model_env.ens and calls uncertainty_usad(obs, action)
    directly with whatever obs the planner holds. For continuous envs the planner
    holds flat obs. This wrapper transparently converts flat -> dict.
    """

    def __init__(self, ensemble: EnsembleDynamicsSB3):
        self._ens = ensemble

    def _to_dict(self, obs) -> Dict[str, np.ndarray]:
        if isinstance(obs, dict):
            return obs
        return {"observation": np.asarray(obs, dtype=np.float32)}

    # ---- Methods the MCTS planner calls via model_env.ens ----

    def uncertainty_usad(self, obs, action: np.ndarray) -> float:
        return float(self._ens.uncertainty_usad(self._to_dict(obs), action))

    def uncertainty_epistemic(self, obs, action: np.ndarray) -> float:
        return float(self._ens.uncertainty_epistemic(self._to_dict(obs), action))

    def uncertainty_total(self, obs, action: np.ndarray) -> float:
        return float(self._ens.uncertainty_total(self._to_dict(obs), action))

    def _predict_all(self, obs, action: np.ndarray) -> np.ndarray:
        return self._ens._predict_all(self._to_dict(obs), action)

    def sample_next_obs_vec(self, obs, action: np.ndarray, **kwargs) -> np.ndarray:
        return self._ens.sample_next_obs_vec(self._to_dict(obs), action, **kwargs)

    def predict_next_obs_vec_mean(self, obs, action: np.ndarray) -> np.ndarray:
        return self._ens.predict_next_obs_vec_mean(self._to_dict(obs), action)

    # ---- Pass-through everything else (e.g. .ensemble list for MOPO) ----

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ens, name)


@dataclass
class ModelContinuousEnvConfig:
    """Configuration for ModelContinuousEnv."""
    # Domain identity (from continuous_env_factory.DOMAIN_CONFIGS)
    domain: str = "walker2d"

    # Episode control
    max_steps: int = 1000
    truncate_on_time_limit: bool = True   # False for MCTS (depth = planning horizon)
    obs_clip: float = 1e3
    terminate_on_nan: bool = True

    # Reward reconstruction (must match gymnasium defaults)
    vel_obs_idx: int = 8          # index of x_velocity in obs
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0

    # Termination / health conditions
    z_obs_idx: int = 0
    angle_obs_idx: int = 1
    healthy_z_lo: float = 0.8
    healthy_z_hi: float = 2.0
    healthy_angle: float = 1.0    # |obs[angle_idx]| <= this

    @classmethod
    def for_domain(cls, domain: str) -> "ModelContinuousEnvConfig":
        """Create config from DOMAIN_CONFIGS entry."""
        from safe_rl.envs.continuous_env_factory import DOMAIN_CONFIGS
        d = DOMAIN_CONFIGS[domain]
        z_lo, z_hi = d["healthy_z_range"]
        return cls(
            domain=domain,
            max_steps=d["max_episode_steps"],
            vel_obs_idx=d["vel_obs_idx"],
            ctrl_cost_weight=d["ctrl_cost_weight"],
            healthy_reward=d["healthy_reward"],
            z_obs_idx=d["z_obs_idx"],
            angle_obs_idx=d["angle_obs_idx"],
            healthy_z_lo=float(z_lo),
            healthy_z_hi=float(z_hi),
            healthy_angle=float(d["healthy_angle"]),
        )


class ModelContinuousEnv:
    """
    Learned model environment for flat-obs continuous MuJoCo domains.

    Analogous to ModelPointMazeEnv. Key differences:
      - obs is a flat numpy array (not a dict)
      - reward reconstructed from predicted obs via known locomotion formula
      - termination from health conditions (z_height, angle)
      - self.ens is an EnsembleFlatWrapper for MCTS planner compatibility

    NOTE: truncate_on_time_limit=False for MCTS (planning horizon = max_depth).
    """

    def __init__(
        self,
        ensemble: EnsembleDynamicsSB3,
        cfg: ModelContinuousEnvConfig,
    ):
        self.ens = EnsembleFlatWrapper(ensemble)
        self.cfg = cfg
        self._t = 0

    def reset_counter(self) -> None:
        self._t = 0

    def _is_healthy(self, obs: np.ndarray) -> bool:
        z    = float(obs[self.cfg.z_obs_idx])
        ang  = float(obs[self.cfg.angle_obs_idx])
        z_ok = (z >= self.cfg.healthy_z_lo) and (z <= self.cfg.healthy_z_hi)
        a_ok = abs(ang) <= self.cfg.healthy_angle
        return bool(z_ok and a_ok)

    def _reward(self, obs_next: np.ndarray, action: np.ndarray) -> float:
        """Reconstruct locomotion reward from predicted next-obs and action."""
        x_vel    = float(obs_next[self.cfg.vel_obs_idx])
        ctrl     = float(self.cfg.ctrl_cost_weight * np.sum(np.asarray(action) ** 2))
        healthy  = float(self.cfg.healthy_reward) if self._is_healthy(obs_next) else 0.0
        return float(x_vel + healthy - ctrl)

    def _finite_and_clip(self, obs: np.ndarray) -> Tuple[np.ndarray, bool]:
        obs = np.asarray(obs, dtype=np.float32).ravel()
        ok  = bool(np.isfinite(obs).all())
        if self.cfg.obs_clip is not None and self.cfg.obs_clip > 0:
            obs = np.clip(obs, -self.cfg.obs_clip, self.cfg.obs_clip)
        return obs, ok

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        model_index: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the learned model.

        Args:
            obs:         flat observation [obs_dim]
            action:      action [act_dim] — clipped to [-1, 1] internally
            model_index: which ensemble member to use (None = random)
            rng:         random generator for member sampling

        Returns:
            (obs_next, reward, terminated, truncated, info)
        """
        if rng is None:
            rng = np.random.default_rng()

        if bool(self.cfg.truncate_on_time_limit):
            self._t += 1

        a = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1.0, 1.0)

        # Predict next obs via ensemble (wrapper handles flat->dict conversion)
        obs_next = self.ens.sample_next_obs_vec(obs, a, model_index=model_index, rng=rng)
        obs_next, ok = self._finite_and_clip(obs_next)

        if (not ok) and bool(self.cfg.terminate_on_nan):
            return obs_next, 0.0, True, False, {"terminated_on_nan": True}

        r          = self._reward(obs_next, a)
        terminated = not self._is_healthy(obs_next)
        truncated  = False

        if bool(self.cfg.truncate_on_time_limit) and (self._t >= int(self.cfg.max_steps)):
            truncated = True

        info: Dict[str, Any] = {
            "success": 0.0,  # locomotion has no binary success; survival is tracked by caller
            "x_velocity": float(obs_next[self.cfg.vel_obs_idx]),
            "is_healthy": float(self._is_healthy(obs_next)),
        }

        return obs_next, float(r), bool(terminated), bool(truncated), info
