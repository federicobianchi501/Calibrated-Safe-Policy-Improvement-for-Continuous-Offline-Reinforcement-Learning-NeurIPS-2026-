# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Cloneable PointMaze wrapper with configurable reward modes."""
import copy
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import Wrapper


def _find_timelimit_wrapper(env) -> Optional[Any]:
    """Return the first wrapper in the chain that holds a TimeLimit step counter, or None."""
    cur = env
    while cur is not None:
        if hasattr(cur, "_elapsed_steps"):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _get_goal_obj(unwrapped) -> Tuple[Optional[Any], Optional[str]]:
    """Return a copy of the env goal and the attribute name it was read from."""
    for name in ("goal", "_goal", "desired_goal", "_desired_goal", "target_goal", "_target_goal"):
        if hasattr(unwrapped, name):
            try:
                return copy.deepcopy(getattr(unwrapped, name)), name
            except Exception:
                return getattr(unwrapped, name), name
    return None, None


def _set_goal_obj(unwrapped, goal_value: Any, goal_attr: Optional[str]) -> None:
    """Write a goal previously read by `_get_goal_obj` back to the env."""
    if goal_attr is None:
        return
    if hasattr(unwrapped, goal_attr):
        try:
            setattr(unwrapped, goal_attr, copy.deepcopy(goal_value))
        except Exception:
            setattr(unwrapped, goal_attr, goal_value)


class CloneablePointMazeWrapper(Wrapper):
    """Cloneable PointMaze wrapper.

    `clone_state` / `restore_state` cover the full MuJoCo physics state, the env RNG, the
    TimeLimit counter and the goal (which changes over time when continuing_task=True).

    Reward modes:
        current:         env reward, plus `success_bonus` on successful termination,
                         minus `living_cost` per step.
        speed:           -step_penalty per step, plus `success_bonus` on success.
        speed_distance:  -step_penalty - dist_penalty * ||achieved_goal - desired_goal||,
                         plus `success_bonus` on success.
    The reward contains only terms that the model environment can reproduce.
    """

    def __init__(
        self,
        env,
        living_cost: float = 0.0,
        success_bonus: float = 0.0,
        reward_mode: str = "current",
        step_penalty: float = 1.0,
        dist_penalty: float = 0.2,
    ):
        super().__init__(env)

        assert hasattr(self.unwrapped, "point_env"), (
            "Expected PointMazeEnv with attribute 'point_env'."
        )

        self.model = self.unwrapped.point_env.model
        self.data  = self.unwrapped.point_env.data

        self._state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self._state_size = mujoco.mj_stateSize(self.model, self._state_spec)

        self.living_cost   = float(living_cost)
        self.success_bonus = float(success_bonus)
        self.reward_mode   = str(reward_mode)
        self.step_penalty  = float(step_penalty)
        self.dist_penalty  = float(dist_penalty)

    def clone_state(self) -> Dict[str, Any]:
        """Return a snapshot of physics, step counter, RNG and goal state."""
        mj_state = np.zeros(self._state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, mj_state, self._state_spec)

        tl = _find_timelimit_wrapper(self)
        timelimit_state = {}
        if tl is not None and hasattr(tl, "_elapsed_steps"):
            timelimit_state["_elapsed_steps"] = int(tl._elapsed_steps)

        rng_state = None
        if hasattr(self.unwrapped, "np_random") and hasattr(self.unwrapped.np_random, "bit_generator"):
            rng_state = copy.deepcopy(self.unwrapped.np_random.bit_generator.state)

        goal_value, goal_attr = _get_goal_obj(self.unwrapped)

        return {
            "mj_state":   mj_state,
            "timelimit":  timelimit_state,
            "rng_state":  rng_state,
            "goal_value": goal_value,
            "goal_attr":  goal_attr,
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore a snapshot produced by `clone_state`."""
        mujoco.mj_setState(self.model, self.data, snapshot["mj_state"], self._state_spec)
        mujoco.mj_forward(self.model, self.data)

        tl = _find_timelimit_wrapper(self)
        if tl is not None and "_elapsed_steps" in snapshot["timelimit"]:
            tl._elapsed_steps = int(snapshot["timelimit"]["_elapsed_steps"])

        rng_state = snapshot.get("rng_state", None)
        if rng_state is not None and hasattr(self.unwrapped, "np_random"):
            self.unwrapped.np_random.bit_generator.state = copy.deepcopy(rng_state)

        _set_goal_obj(
            self.unwrapped,
            snapshot.get("goal_value", None),
            snapshot.get("goal_attr", None),
        )

    def step(self, action):
        """Step the env and replace the reward according to `reward_mode`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not isinstance(info, dict):
            info = {}
        else:
            info = dict(info)

        is_success  = bool(info.get("success", False))
        reward_mode = str(getattr(self, "reward_mode", "current"))

        if reward_mode == "current":
            sb = float(getattr(self, "success_bonus", 0.0))
            if sb != 0.0 and bool(terminated) and is_success:
                reward = float(reward) + sb
                info["success_bonus"] = sb
            lc = float(getattr(self, "living_cost", 0.0))
            if lc != 0.0:
                info["living_cost"] = lc
                info["reward_wo_living_cost"] = float(reward)
                reward = float(reward) - lc

        elif reward_mode == "speed":
            reward = -float(getattr(self, "step_penalty", 1.0))
            if bool(terminated) and is_success:
                sb = float(getattr(self, "success_bonus", 0.0))
                reward = float(reward) + sb
                info["success_bonus"] = sb
            info["step_penalty"] = float(getattr(self, "step_penalty", 1.0))

        elif reward_mode == "speed_distance":
            step_penalty = float(getattr(self, "step_penalty", 1.0))
            dist_penalty = float(getattr(self, "dist_penalty", 0.2))
            d = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            reward = -step_penalty - dist_penalty * d
            if bool(terminated) and is_success:
                sb = float(getattr(self, "success_bonus", 0.0))
                reward = float(reward) + sb
                info["success_bonus"] = sb
            info["step_penalty"]  = step_penalty
            info["dist_penalty"]  = dist_penalty
            info["goal_distance"] = d

        else:
            raise ValueError(
                f"Unknown reward_mode={reward_mode}. Expected 'current', 'speed' or 'speed_distance'."
            )

        return obs, float(reward), bool(terminated), bool(truncated), info
