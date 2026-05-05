# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
import copy
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import Wrapper


def _find_timelimit_wrapper(env) -> Optional[Any]:
    cur = env
    while cur is not None:
        if hasattr(cur, "_elapsed_steps"):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _get_goal_obj(unwrapped) -> Tuple[Optional[Any], Optional[str]]:
    for name in ("goal", "_goal", "desired_goal", "_desired_goal", "target_goal", "_target_goal"):
        if hasattr(unwrapped, name):
            try:
                return copy.deepcopy(getattr(unwrapped, name)), name
            except Exception:
                return getattr(unwrapped, name), name
    return None, None


def _set_goal_obj(unwrapped, goal_value: Any, goal_attr: Optional[str]) -> None:
    if goal_attr is None:
        return
    if hasattr(unwrapped, goal_attr):
        try:
            setattr(unwrapped, goal_attr, copy.deepcopy(goal_value))
        except Exception:
            setattr(unwrapped, goal_attr, goal_value)


class CloneablePointMazeWrapper(Wrapper):
    """
    Fully cloneable MuJoCo wrapper for PointMaze.

    - Full MuJoCo physics cloning
    - RNG cloning
    - TimeLimit counter cloning
    - Goal cloning (important for continuing_task!)

    FIXED: Removed wall_penalty (compared to the broken new version).
    wall_penalty was removed because:
    1. The model env has no wall_penalty in its reward -> reward mismatch if != 0
    2. The ensemble learns from data collected with this env's reward, so any 
       wall_penalty here would need to be mirrored in the model env (which is hard)
    3. wall_penalty=0.0 was the only value used in all experiments anyway
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
