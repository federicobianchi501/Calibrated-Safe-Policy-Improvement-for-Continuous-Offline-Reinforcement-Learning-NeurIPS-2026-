# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
# src/safe_rl/offline/model_env_pointmaze.py  [FIXED — no wall_penalty]
#
# BUG FIX: Removed wall_penalty from ModelEnvConfig.
# REASON: The ensemble is trained on dataset transitions WITHOUT wall_penalty
# (wall_penalty was only in the real env and defaults to 0.0). Having wall_penalty
# in ModelEnvConfig but not applying it in step() creates a dead-code config field.
# More importantly, if wall_penalty != 0 in real env but not in model env,
# the reward function is inconsistent: ensemble learns reward without wall_penalty
# but the gate/planner expects reward with wall_penalty.
# Safest fix: remove it entirely and keep model env reward matched to dataset reward.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from safe_rl.offline.ensemble_dynamics_sb3 import EnsembleDynamicsSB3


@dataclass
class ModelEnvConfig:
    continuing_task: bool = False
    reward_type: str = "sparse"
    goal_thresh: float = 0.45
    max_steps: int = 300
    truncate_on_time_limit: bool = True
    obs_clip: float = 1e3
    terminate_on_nan: bool = True
    living_cost: float = 0.0
    success_bonus: float = 0.0
    reward_mode: str = "current"
    step_penalty: float = 1.0
    dist_penalty: float = 0.2


class ModelPointMazeEnv:
    """
    Learned model env for Gymnasium-Robotics PointMaze dict observations.

    IMPORTANT:
      In PointMaze, achieved_goal is the (x,y) position.
      We MUST update achieved_goal from the predicted observation vector,
      otherwise distance-to-goal never changes in planning.

    NOTE ON TIME LIMIT:
      Set truncate_on_time_limit=False for MCTS (tree depth = planning horizon).
    """

    def __init__(
        self,
        ensemble: EnsembleDynamicsSB3,
        cfg: ModelEnvConfig,
        oracle_env: Optional[Any] = None,
    ):
        self.ens = ensemble
        self.cfg = cfg
        self.oracle_env = oracle_env
        self._t = 0

    @staticmethod
    def _dist_to_goal(obs: Dict[str, np.ndarray]) -> float:
        ag = np.asarray(obs["achieved_goal"], dtype=np.float32).ravel()
        dg = np.asarray(obs["desired_goal"],  dtype=np.float32).ravel()
        return float(np.linalg.norm(ag - dg))

    def reset_counter(self) -> None:
        self._t = 0

    def _reward_and_done(self, obs_next: Dict[str, np.ndarray]) -> Tuple[float, bool]:
        d = self._dist_to_goal(obs_next)
        success    = d < float(self.cfg.goal_thresh)
        terminated = bool(success and (not bool(self.cfg.continuing_task)))
        mode       = str(getattr(self.cfg, "reward_mode", "current"))

        if mode == "current":
            r = float(np.exp(-d)) if str(self.cfg.reward_type) == "dense" else (1.0 if success else 0.0)
            if terminated:
                r += float(self.cfg.success_bonus)
            r -= float(self.cfg.living_cost)

        elif mode == "speed":
            r = -float(self.cfg.step_penalty)
            if terminated:
                r += float(self.cfg.success_bonus)

        elif mode == "speed_distance":
            r = -float(self.cfg.step_penalty) - float(self.cfg.dist_penalty) * float(d)
            if terminated:
                r += float(self.cfg.success_bonus)

        else:
            raise ValueError(f"Unknown reward_mode={mode}")

        return float(r), bool(terminated)

    def _finite_and_clip(self, obs_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], bool]:
        out = {}
        ok  = True
        clip = float(self.cfg.obs_clip) if self.cfg.obs_clip is not None else None
        for k in ("observation", "achieved_goal", "desired_goal"):
            v = np.asarray(obs_dict[k], dtype=np.float32).ravel()
            if not np.isfinite(v).all():
                ok = False
            if clip is not None and clip > 0:
                v = np.clip(v, -clip, clip)
            out[k] = v
        return out, ok

    def _postprocess_pointmaze_goals(self, obs_next: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs_vec = np.asarray(obs_next["observation"], dtype=np.float32).ravel()
        ag      = np.asarray(obs_next["achieved_goal"], dtype=np.float32).ravel()
        ag_dim  = int(ag.shape[0])
        if obs_vec.shape[0] >= ag_dim:
            obs_next["achieved_goal"] = obs_vec[:ag_dim].copy()
        return obs_next

    def _call_predict_next_obs_dict(self, obs, action, model_index, rng):
        fn = getattr(self.ens, "predict_next_obs_dict", None)
        if fn is None:
            return None
        try:
            return fn(obs, action, model_index=model_index, rng=rng)
        except TypeError:
            pass
        try:
            return fn(obs, action, model_index=model_index)
        except TypeError:
            pass
        try:
            return fn(obs, action, rng=rng)
        except TypeError:
            pass
        return fn(obs, action)

    def _predict_next(self, obs, action, model_index, rng):
        # Path 1: dict-level predictor (preferred)
        out = self._call_predict_next_obs_dict(obs, action, model_index, rng)
        if out is not None:
            out = {
                "observation":   np.asarray(out["observation"],   dtype=np.float32).ravel(),
                "achieved_goal": np.asarray(out["achieved_goal"], dtype=np.float32).ravel(),
                "desired_goal":  np.asarray(out["desired_goal"],  dtype=np.float32).ravel(),
            }
            return self._postprocess_pointmaze_goals(out)

        # Path 2: vector mean predictor
        fn = getattr(self.ens, "predict_next_obs_vec_mean", None)
        if fn is not None:
            next_obs_vec = np.asarray(fn(obs, action), dtype=np.float32).ravel()
            out = {
                "observation":   next_obs_vec,
                "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
                "desired_goal":  np.asarray(obs["desired_goal"],  dtype=np.float32).ravel(),
            }
            return self._postprocess_pointmaze_goals(out)

        # Path 3: fallback via _predict_all
        if hasattr(self.ens, "_predict_all"):
            mus = self.ens._predict_all(obs, action)   # [E, obs_dim]
            mean_delta = mus.mean(axis=0).astype(np.float32)
            obs_vec = np.asarray(obs["observation"], dtype=np.float32).ravel()
            out = {
                "observation":   (obs_vec + mean_delta).astype(np.float32),
                "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
                "desired_goal":  np.asarray(obs["desired_goal"],  dtype=np.float32).ravel(),
            }
            return self._postprocess_pointmaze_goals(out)

        raise RuntimeError("Ensemble model missing prediction method(s).")

    def step(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        *,
        model_index: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        if bool(self.cfg.truncate_on_time_limit):
            self._t += 1

        a = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1.0, 1.0)

        obs_next = self._predict_next(obs, a, model_index=model_index, rng=rng)
        obs_next, ok = self._finite_and_clip(obs_next)

        info: Dict[str, Any] = {}

        if (not ok) and bool(self.cfg.terminate_on_nan):
            return obs_next, 0.0, True, False, {"terminated_on_nan": True}

        r, terminated = self._reward_and_done(obs_next)

        truncated = False
        if bool(self.cfg.truncate_on_time_limit) and (self._t >= int(self.cfg.max_steps)):
            truncated = True

        d = self._dist_to_goal(obs_next)
        info["success"] = 1.0 if float(d) < float(self.cfg.goal_thresh) else 0.0

        return obs_next, float(r), bool(terminated), bool(truncated), info
