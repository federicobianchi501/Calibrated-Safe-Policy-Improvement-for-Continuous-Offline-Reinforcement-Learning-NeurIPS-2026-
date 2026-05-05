# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
# evaluator_fixed.py
import numpy as np
import gymnasium as gym
from safe_rl.envs.wrappers import FlattenPointMazeObs


def make_pointmaze_env(seed=None):
    """
    Create a fresh evaluation environment.
    PointMaze is wrapped with FlattenPointMazeObs so PPO/MCTS see a flat Box obs.

    This function is used in every episode (no VecEnv).
    """
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
    """
    Evaluate any deterministic or stochastic policy (PPO, MCTS wrapper, etc.)
    on PointMaze with research-grade metrics used in literature.

    Metrics returned:
        - mean_return
        - std_return
        - success_rate
        - final_distance_mean
        - final_distance_std
        - returns (list)
        - distances (list)
        - success (list of 0/1)
    """

    returns = []
    distances = []
    success_flags = []

    rng = np.random.default_rng(seed)

    for ep in range(episodes):

        # fresh environment for each episode, with reproducible seed
        env = make_pointmaze_env(seed=int(rng.integers(0, 1e9)))
        obs, info = env.reset()
        desired_goal = info.get("desired_goal")

        ep_ret = 0.0
        final_dist = None

        for t in range(max_steps):

            # Policy prediction
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)

            achieved = info.get("achieved_goal")
            dg = info.get("desired_goal", desired_goal)

            # Compute distance every step
            if achieved is not None and dg is not None:
                final_dist = float(np.linalg.norm(achieved - dg))

            # STOP EARLY IF SUCCESS
            if final_dist is not None and final_dist < success_threshold:
                break

            if terminated or truncated:
                break

        # If no valid distance was extracted (should not happen), set large value
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
