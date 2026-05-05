# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np
import gymnasium as gym

# PointMaze factory already in your repo
from safe_rl.envs.pointmaze_factory import make_pointmaze_env

try:
    import gymnasium_robotics  # noqa: F401
except Exception:
    # ok if user doesn't run PointMaze
    pass


@dataclass
class EnvSpec:
    env_id: str
    obs_kind: str                 # "dict_goal" or "vec"
    max_steps: int
    action_low: np.ndarray
    action_high: np.ndarray


class DomainAdapter:
    def make_env(self, seed: int, continuing_task: bool, reward_type: str, disable_env_checker: bool = True):
        raise NotImplementedError

    def infer_spec(self, seed: int, continuing_task: bool, reward_type: str) -> EnvSpec:
        env = self.make_env(seed, continuing_task, reward_type)
        try:
            obs, _ = env.reset(seed=seed)
            act_low = np.asarray(env.action_space.low, dtype=np.float32).ravel()
            act_high = np.asarray(env.action_space.high, dtype=np.float32).ravel()
            max_steps = int(getattr(env, "_max_episode_steps", 0) or getattr(env, "spec", None).max_episode_steps or 0)
            obs_kind = "dict_goal" if isinstance(obs, dict) else "vec"
            return EnvSpec(env_id=str(env.unwrapped.spec.id) if getattr(env, "unwrapped", None) and getattr(env.unwrapped, "spec", None) else "unknown",
                           obs_kind=obs_kind,
                           max_steps=max_steps if max_steps > 0 else 1000,
                           action_low=act_low,
                           action_high=act_high)
        finally:
            env.close()

    @staticmethod
    def clip_action(a: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).ravel()
        return np.clip(a, low, high).astype(np.float32)


class PointMazeAdapter(DomainAdapter):
    def __init__(self, env_id: str = "PointMaze_UMaze-v3"):
        self.env_id = env_id

    def make_env(self, seed: int, continuing_task: bool, reward_type: str, disable_env_checker: bool = True):
        # your unified factory
        return make_pointmaze_env(
            seed=seed,
            continuing_task=continuing_task,
            reward_type=reward_type,
            disable_env_checker=disable_env_checker,
        )


class MujocoAdapter(DomainAdapter):
    """
    For Hopper/Walker2d via gymnasium MuJoCo envs (e.g., Hopper-v4, Walker2d-v4).
    Uses standard gym.make.
    """
    def __init__(self, env_id: str):
        self.env_id = env_id

    def make_env(self, seed: int, continuing_task: bool, reward_type: str, disable_env_checker: bool = True):
        # continuing_task/reward_type not used by these envs; kept for unified CLI.
        env = gym.make(self.env_id, disable_env_checker=disable_env_checker)
        env.reset(seed=seed)
        return env
