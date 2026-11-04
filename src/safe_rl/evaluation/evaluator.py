# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Episodic evaluation of a policy on PointMaze with flattened observations."""
import numpy as np
import gymnasium as gym
from safe_rl.envs.wrappers import FlattenPointMazeObs


def make_pointmaze_env(seed=None):
    """Create a PointMaze UMaze env with flat Box observations (FlattenPointMazeObs)."""
    env = gym.make("PointMaze_UMaze-v3", disable_env_checker=True)

    if seed is not None:
        env.reset(seed=seed)

    env = FlattenPointMazeObs(env)
    return env


def evaluate_policy_pointmaze(
    model,
    episodes=20,
    max_steps=300,
    success_threshold=0.5,
    seed=12345
):
    """Evaluate `model.predict(obs, deterministic=True)` on PointMaze.

    Each episode uses a fresh env seeded from `seed`. An episode ends on success
    (goal distance < `success_threshold`), termination, truncation or `max_steps`.

    Returns:
        Dict with mean_return, std_return, success_rate, final_distance_mean,
        final_distance_std and the per-episode lists returns, distances, success (0/1).
        The final distance is 999.0 when the env reports no goal information.
    """

    returns = []
    distances = []
    success_flags = []

    rng = np.random.default_rng(seed)

    for ep in range(episodes):

        # Fresh env per episode, seeded from the evaluation RNG.
        env = make_pointmaze_env(seed=int(rng.integers(0, 1e9)))
        obs, info = env.reset()
        desired_goal = info.get("desired_goal")

        ep_ret = 0.0
        final_dist = None

        for t in range(max_steps):

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)

            achieved = info.get("achieved_goal")
            dg = info.get("desired_goal", desired_goal)

            if achieved is not None and dg is not None:
                final_dist = float(np.linalg.norm(achieved - dg))

            # Stop early on success.
            if final_dist is not None and final_dist < success_threshold:
                break

            if terminated or truncated:
                break

        # No goal information in info: record a large distance.
        if final_dist is None:
            final_dist = 999.0

        returns.append(ep_ret)
        distances.append(final_dist)
        success_flags.append(1 if final_dist < success_threshold else 0)

        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(success_flags)),
        "final_distance_mean": float(np.mean(distances)),
        "final_distance_std": float(np.std(distances)),
        "returns": returns,
        "distances": distances,
        "success": success_flags,
    }
