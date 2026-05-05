# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
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
    """
    Factory for PointMaze env. Use this everywhere (train, eval, MCTS).

    FIXED: Removed wall_penalty parameter.
    wall_penalty was removed because the model env cannot mirror it,
    creating a reward mismatch between real env and learned model.
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
