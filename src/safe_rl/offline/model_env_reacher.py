# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
model_env_reacher.py — Reacher-v5 specialization of ModelContinuousEnv.

Inherits all clone/restore/step machinery from ModelContinuousEnv and only
overrides the reward function and the (trivial) health check. Reacher's reward
is target-distance based, not locomotion-velocity based, so the canonical
locomotion formula r = x_vel + healthy - ctrl does not apply.
"""
from __future__ import annotations

import numpy as np

from safe_rl.offline.model_env_continuous import (
    ModelContinuousEnv,
    ModelContinuousEnvConfig,
)


class ModelReacherEnv(ModelContinuousEnv):
    """
    Learned model environment for Reacher-v5.

    Reward: r = -||fingertip - target||_2 - ctrl_cost_weight * ||a||^2.
    The fingertip-target vector lives at obs[8:10] in Reacher-v5's flat
    observation (planar, so 2 components only).

    Termination: Reacher cannot fail catastrophically; episodes truncate at
    max_episode_steps. We therefore mark every state as healthy.
    """

    def _compute_reward(self, obs_next, action):  # type: ignore[override]
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        dist = float(np.linalg.norm(obs_next[8:10]))
        ctrl = float(self.cfg.ctrl_cost_weight) * float(np.sum(a ** 2))
        return -dist - ctrl

    def _is_healthy(self, obs_next) -> bool:  # type: ignore[override]
        return True
