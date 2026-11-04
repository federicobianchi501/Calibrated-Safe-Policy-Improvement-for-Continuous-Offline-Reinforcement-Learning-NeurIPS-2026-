# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Factory for the PointMaze UMaze environment used for data collection, planning and eval."""
import gymnasium as gym
from safe_rl.envs.cloneable_pointmaze_wrapper import CloneablePointMazeWrapper

_REGISTERED = False

def make_pointmaze_env(
    seed: int,
    *,
    continuing_task: bool,
    reward_type: str = "sparse",
    disable_env_checker: bool = True,
    living_cost: float = 0.0,
    success_bonus: float = 0.0,
    reward_mode: str = "current",
    step_penalty: float = 1.0,
    dist_penalty: float = 0.2,
):
    """Create a seeded PointMaze UMaze env wrapped in CloneablePointMazeWrapper.

    Args:
        seed: seed for env reset and for the action/observation spaces.
        continuing_task: if True, a new goal is sampled after each success.
        reward_type: "sparse" (PointMaze_UMaze-v3) or "dense" (PointMaze_UMazeDense-v3).
        living_cost, success_bonus, reward_mode, step_penalty, dist_penalty: reward
            shaping passed to CloneablePointMazeWrapper.
    """
    global _REGISTERED

    if not _REGISTERED:
        try:
            import gymnasium_robotics
            gym.register_envs(gymnasium_robotics)
            _REGISTERED = True
        except Exception:
            print("Gymnasium robotics not registered")
            pass

    if reward_type == "dense":
        env_id = "PointMaze_UMazeDense-v3"
    elif reward_type == "sparse":
        env_id = "PointMaze_UMaze-v3"
    else:
        raise ValueError(f"Unknown reward_type={reward_type}. Expected 'sparse' or 'dense'.")

    env = gym.make(
        env_id,
        continuing_task=continuing_task,
        disable_env_checker=disable_env_checker,
    )
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    env.observation_space.seed(int(seed))
    env = CloneablePointMazeWrapper(
        env,
        living_cost=float(living_cost),
        success_bonus=float(success_bonus),
        reward_mode=str(reward_mode),
        step_penalty=float(step_penalty),
        dist_penalty=float(dist_penalty),
    )
    return env
