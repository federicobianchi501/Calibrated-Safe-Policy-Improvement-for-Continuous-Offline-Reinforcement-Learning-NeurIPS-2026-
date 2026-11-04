# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Clone/restore wrapper for MuJoCo environments with flat observations.

Applies to any gymnasium MuJoCo env exposing `.model` and `.data` (Hopper, HalfCheetah,
Walker2d, Ant). Observations are passed through unchanged.
"""
import copy
from typing import Any, Dict, Optional

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


class ContinuousEnvWrapper(Wrapper):
    """Snapshot and restore of MuJoCo physics, TimeLimit counter and env RNG.

    Same clone/restore logic as CloneablePointMazeWrapper, without goal state.
    """

    def __init__(self, env):
        super().__init__(env)
        u = self.unwrapped
        self.model = u.model
        self.data  = u.data
        self._state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self._state_size = mujoco.mj_stateSize(self.model, self._state_spec)

    def clone_state(self) -> Dict[str, Any]:
        """Return a snapshot of the full physics state, step counter and RNG state."""
        mj_state = np.zeros(self._state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, mj_state, self._state_spec)

        tl = _find_timelimit_wrapper(self)
        timelimit_state = {}
        if tl is not None and hasattr(tl, "_elapsed_steps"):
            timelimit_state["_elapsed_steps"] = int(tl._elapsed_steps)

        rng_state = None
        if hasattr(self.unwrapped, "np_random") and hasattr(
            self.unwrapped.np_random, "bit_generator"
        ):
            rng_state = copy.deepcopy(self.unwrapped.np_random.bit_generator.state)

        return {
            "mj_state":  mj_state,
            "timelimit": timelimit_state,
            "rng_state": rng_state,
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore a snapshot produced by `clone_state` and recompute derived quantities."""
        mujoco.mj_setState(
            self.model, self.data, snapshot["mj_state"], self._state_spec
        )
        mujoco.mj_forward(self.model, self.data)

        tl = _find_timelimit_wrapper(self)
        if tl is not None and "_elapsed_steps" in snapshot["timelimit"]:
            tl._elapsed_steps = int(snapshot["timelimit"]["_elapsed_steps"])

        rng_state = snapshot.get("rng_state")
        if rng_state is not None and hasattr(self.unwrapped, "np_random"):
            self.unwrapped.np_random.bit_generator.state = copy.deepcopy(rng_state)
