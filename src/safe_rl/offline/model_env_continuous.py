# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Learned-model environment for flat-observation MuJoCo locomotion domains.

Same step interface as ModelPointMazeEnv, without goals. Flat observations are wrapped
as {"observation": obs} for the ensemble by EnsembleFlatWrapper. The reward is computed
from the predicted observation with the Gymnasium locomotion formula
(x velocity + healthy reward - control cost), and termination from the healthy
z-height and angle ranges of the domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from safe_rl.offline.ensemble_dynamics_sb3 import EnsembleDynamicsSB3


class EnsembleFlatWrapper:
    """Adapter that lets EnsembleDynamicsSB3 accept flat observation arrays.

    Flat inputs are converted to {"observation": obs}; dict inputs pass through.
    Attributes not defined here are forwarded to the wrapped ensemble.
    """

    def __init__(self, ensemble: EnsembleDynamicsSB3):
        self._ens = ensemble

    def _to_dict(self, obs) -> Dict[str, np.ndarray]:
        if isinstance(obs, dict):
            return obs
        return {"observation": np.asarray(obs, dtype=np.float32)}

    # ---- Methods called by the planner through model_env.ens ----

    def uncertainty_usad(self, obs, action: np.ndarray) -> float:
        """Return u_USAD for a flat or dict observation."""
        return float(self._ens.uncertainty_usad(self._to_dict(obs), action))

    def uncertainty_epistemic(self, obs, action: np.ndarray) -> float:
        """Return u_VAR for a flat or dict observation."""
        return float(self._ens.uncertainty_epistemic(self._to_dict(obs), action))

    def uncertainty_total(self, obs, action: np.ndarray) -> float:
        """Return the ensemble's total uncertainty (equal to u_VAR)."""
        return float(self._ens.uncertainty_total(self._to_dict(obs), action))

    def _predict_all(self, obs, action: np.ndarray) -> np.ndarray:
        return self._ens._predict_all(self._to_dict(obs), action)

    def sample_next_obs_vec(self, obs, action: np.ndarray, **kwargs) -> np.ndarray:
        """Predict the next observation; kwargs as in EnsembleDynamicsSB3."""
        return self._ens.sample_next_obs_vec(self._to_dict(obs), action, **kwargs)

    def predict_next_obs_vec_mean(self, obs, action: np.ndarray) -> np.ndarray:
        """Return the ensemble-mean next observation."""
        return self._ens.predict_next_obs_vec_mean(self._to_dict(obs), action)

    # ---- Remaining attributes (e.g. .ensemble) are forwarded ----

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ens, name)


@dataclass
class ModelContinuousEnvConfig:
    """Configuration for ModelContinuousEnv."""
    # Key of continuous_env_factory.DOMAIN_CONFIGS
    domain: str = "walker2d"

    # Episode control
    max_steps: int = 1000
    truncate_on_time_limit: bool = True   # False for MCTS (depth = planning horizon)
    obs_clip: float = 1e3
    terminate_on_nan: bool = True

    # Reward terms; defaults match Gymnasium
    vel_obs_idx: int = 8          # obs index of the x velocity
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0

    # Healthy ranges; leaving them terminates the episode
    z_obs_idx: int = 0
    angle_obs_idx: int = 1
    healthy_z_lo: float = 0.8
    healthy_z_hi: float = 2.0
    healthy_angle: float = 1.0    # |obs[angle_idx]| <= this

    @classmethod
    def for_domain(cls, domain: str) -> "ModelContinuousEnvConfig":
        """Build the config from the DOMAIN_CONFIGS entry of a domain."""
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
    """Learned-model environment for flat-observation MuJoCo domains.

    obs is a flat array; the reward is computed from the predicted observation and the
    action, and termination from the healthy ranges. self.ens is an EnsembleFlatWrapper.
    For MCTS use truncate_on_time_limit=False; the tree depth is then the horizon.
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
        """Reset the step counter used for time-limit truncation."""
        self._t = 0

    def _is_healthy(self, obs: np.ndarray) -> bool:
        z    = float(obs[self.cfg.z_obs_idx])
        ang  = float(obs[self.cfg.angle_obs_idx])
        z_ok = (z >= self.cfg.healthy_z_lo) and (z <= self.cfg.healthy_z_hi)
        a_ok = abs(ang) <= self.cfg.healthy_angle
        return bool(z_ok and a_ok)

    def _reward(self, obs_next: np.ndarray, action: np.ndarray) -> float:
        """Return x_velocity + healthy_reward * healthy - ctrl_cost_weight * ||a||^2."""
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
        """Advance one model step.

        Args:
            obs: flat observation of shape [obs_dim].
            action: action of shape [act_dim], clipped to [-1, 1].
            model_index: ensemble member (None draws one from rng, -1 uses the mean).
            rng: generator for member sampling.

        Returns:
            (obs_next, reward, terminated, truncated, info). Non-finite predictions end
            the episode with reward 0 when cfg.terminate_on_nan is set.
        """
        if rng is None:
            rng = np.random.default_rng()

        if bool(self.cfg.truncate_on_time_limit):
            self._t += 1

        a = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1.0, 1.0)

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
            "success": 0.0,  # no binary success in locomotion
            "x_velocity": float(obs_next[self.cfg.vel_obs_idx]),
            "is_healthy": float(self._is_healthy(obs_next)),
        }

        return obs_next, float(r), bool(terminated), bool(truncated), info
