# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
continuous_env_factory.py — Factory functions for Walker2d and Hopper.

Usage:
    env = make_walker2d_env(seed=0)
    env = make_hopper_env(seed=0)
    env = make_continuous_env("walker2d", seed=0)
"""
from typing import Optional

import gymnasium as gym

from safe_rl.envs.continuous_env_wrapper import ContinuousEnvWrapper

# Per-domain constants used by benchmark and model env
DOMAIN_CONFIGS = {
    "walker2d": {
        "env_id":           "Walker2d-v5",
        "obs_dim":          17,
        "act_dim":          6,
        "max_episode_steps": 1000,
        # Reward: r = x_vel + healthy_reward - ctrl_cost_weight * ||a||^2
        "vel_obs_idx":      8,          # obs[8] = qvel[0] = x_velocity
        "ctrl_cost_weight": 1e-3,
        "healthy_reward":   1.0,
        # Termination: unhealthy if z_height out of [z_lo, z_hi] or |angle| > angle_hi
        "z_obs_idx":        0,
        "angle_obs_idx":    1,
        "healthy_z_range":  (0.8, 2.0),
        "healthy_angle":    1.0,        # |angle| <= this
    },
    "hopper": {
        "env_id":           "Hopper-v5",
        "obs_dim":          11,
        "act_dim":          3,
        "max_episode_steps": 1000,
        "vel_obs_idx":      5,          # obs[5] = qvel[0] = x_velocity
        "ctrl_cost_weight": 1e-3,
        "healthy_reward":   1.0,
        "z_obs_idx":        0,
        "angle_obs_idx":    1,
        "healthy_z_range":  (0.7, float("inf")),
        "healthy_angle":    0.2,
    },
    "halfcheetah": {
        "env_id":           "HalfCheetah-v5",
        "obs_dim":          17,
        "act_dim":          6,
        "max_episode_steps": 1000,
        # HalfCheetah reward = forward_reward_weight * x_vel - ctrl_cost_weight * ||a||^2.
        # No healthy bonus, no termination on fall (the body cannot fall over).
        "vel_obs_idx":      8,          # obs[8] = qvel[0] = x_velocity
        "ctrl_cost_weight": 0.1,
        "healthy_reward":   0.0,
        "z_obs_idx":        0,
        "angle_obs_idx":    1,
        "healthy_z_range":  (-float("inf"), float("inf")),  # always healthy
        "healthy_angle":    float("inf"),
    },
    "reacher": {
        "env_id":           "Reacher-v5",
        "obs_dim":          10,
        "act_dim":          2,
        "max_episode_steps": 50,
        # Reacher reward = -||fingertip - target||_2 - ctrl_cost_weight * ||a||^2
        # The fingertip-target vector is obs[8:10] (2D since Reacher is planar).
        # No locomotion semantics — vel/z/angle entries are placeholders.
        "vel_obs_idx":      0,                # unused
        "ctrl_cost_weight": 1.0,              # Reacher default
        "healthy_reward":   0.0,
        "z_obs_idx":        0,                # unused
        "angle_obs_idx":    1,                # unused
        "healthy_z_range":  (-float("inf"), float("inf")),
        "healthy_angle":    float("inf"),
    },
    "swimmer": {
        "env_id":           "Swimmer-v5",
        "obs_dim":          8,
        "act_dim":          2,
        "max_episode_steps": 1000,
        # Swimmer reward = forward_reward_weight * x_vel - ctrl_cost_weight * ||a||^2.
        # obs = qpos[2:] (3 angles) + qvel (5) → x_velocity = qvel[0] = obs[3].
        # No fall termination, no health bonus.
        "vel_obs_idx":      3,                # obs[3] = qvel[0] = x_velocity
        "ctrl_cost_weight": 1e-4,             # Swimmer-v5 default
        "healthy_reward":   0.0,
        "z_obs_idx":        0,                # unused
        "angle_obs_idx":    1,                # unused
        "healthy_z_range":  (-float("inf"), float("inf")),
        "healthy_angle":    float("inf"),
    },
}


def make_continuous_env(
    domain: str,
    seed: int,
    *,
    max_episode_steps: Optional[int] = None,
) -> ContinuousEnvWrapper:
    """
    Create a ContinuousEnvWrapper for the given domain.

    Args:
        domain: "walker2d" or "hopper"
        seed: random seed for the environment
        max_episode_steps: override default episode length if provided
    """
    cfg = DOMAIN_CONFIGS[domain]
    n_steps = max_episode_steps if max_episode_steps is not None else cfg["max_episode_steps"]
    env = gym.make(cfg["env_id"], max_episode_steps=int(n_steps))
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    env.observation_space.seed(int(seed))
    return ContinuousEnvWrapper(env)


def make_walker2d_env(seed: int, *, max_episode_steps: int = 1000) -> ContinuousEnvWrapper:
    """Factory for Walker2d-v5."""
    return make_continuous_env("walker2d", seed, max_episode_steps=max_episode_steps)


def make_hopper_env(seed: int, *, max_episode_steps: int = 1000) -> ContinuousEnvWrapper:
    """Factory for Hopper-v5."""
    return make_continuous_env("hopper", seed, max_episode_steps=max_episode_steps)


