# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Observation wrappers for goal-conditioned environments."""
import gymnasium as gym
import numpy as np

class FlattenPointMazeObs(gym.ObservationWrapper):
    """Flattens the PointMaze dict observation into [observation, achieved_goal, desired_goal]."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space

        low = np.concatenate([
            obs_space["observation"].low,
            obs_space["achieved_goal"].low,
            obs_space["desired_goal"].low
        ]).astype(np.float32)

        high = np.concatenate([
            obs_space["observation"].high,
            obs_space["achieved_goal"].high,
            obs_space["desired_goal"].high
        ]).astype(np.float32)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def observation(self, obs):
        """Return the concatenated float32 observation vector."""
        return np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"]
        ]).astype(np.float32)
