# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Factories and per-domain constants for the MuJoCo domains.

Supported domains: walker2d, hopper, halfcheetah, ant (gymnasium v5 environments).
`DOMAIN_CONFIGS` holds the observation indices and reward/termination constants that the
learned model environment uses to recompute rewards and termination from flat observations.

    env = make_continuous_env("halfcheetah", seed=0)
"""
from typing import Optional

import gymnasium as gym

from safe_rl.envs.continuous_env_wrapper import ContinuousEnvWrapper

# Per-domain constants shared by the benchmarks and the model environment.
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
        # Reward = forward_reward_weight * x_vel - ctrl_cost_weight * ||a||^2.
        # No healthy bonus and no termination.
        "vel_obs_idx":      8,          # obs[8] = qvel[0] = x_velocity
        "ctrl_cost_weight": 0.1,
        "healthy_reward":   0.0,
        "z_obs_idx":        0,
        "angle_obs_idx":    1,
        "healthy_z_range":  (-float("inf"), float("inf")),  # always healthy
        "healthy_angle":    float("inf"),
    },
    "ant": {
        "env_id":           "Ant-v5",
        "obs_dim":          27,
        "act_dim":          8,
        "max_episode_steps": 1000,
        # Reward = healthy_reward + x_vel - ctrl_cost_weight * ||a||^2 - contact_cost.
        # The env_kwargs make the reward exactly reconstructible from the flat obs:
        #   include_cfrc_ext_in_observation=False drops the 78 contact-force entries
        #     (obs_dim = 27);
        #   contact_cost_weight=0.0 removes the contact cost, which cannot be
        #     recomputed from the remaining observation.
        # obs = qpos[2:] (13) + qvel (14), so x_velocity = qvel[0] = obs[13].
        # Termination: torso z outside [0.2, 1.0]. There is no scalar trunk angle in 3D,
        # so the angle condition is disabled.
        "env_kwargs": {
            "include_cfrc_ext_in_observation": False,
            "contact_cost_weight": 0.0,
        },
        "vel_obs_idx":      13,
        "ctrl_cost_weight": 0.5,              # Ant-v5 default
        "healthy_reward":   1.0,
        "z_obs_idx":        0,                # obs[0] = qpos[2] = torso height
        "angle_obs_idx":    1,                # unused
        "healthy_z_range":  (0.2, 1.0),
        "healthy_angle":    float("inf"),
    },
}


def make_continuous_env(
    domain: str,
    seed: int,
    *,
    max_episode_steps: Optional[int] = None,
) -> ContinuousEnvWrapper:
    """Create a seeded ContinuousEnvWrapper for `domain`.

    Args:
        domain: key of DOMAIN_CONFIGS (walker2d, hopper, halfcheetah, ant).
        seed: seed for env reset and for the action/observation spaces.
        max_episode_steps: episode length; defaults to the domain value.
    """
    cfg = DOMAIN_CONFIGS[domain]
    n_steps = max_episode_steps if max_episode_steps is not None else cfg["max_episode_steps"]
    env = gym.make(cfg["env_id"], max_episode_steps=int(n_steps),
                   **cfg.get("env_kwargs", {}))
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


