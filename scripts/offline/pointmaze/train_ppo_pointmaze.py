# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
# scripts/offline/pointmaze/train_ppo_pointmaze_suboptimal.py
import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from safe_rl.envs.pointmaze_factory import make_pointmaze_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="models/ppo_pointmaze_baseline.zip")

    # ENV CONFIG (MUST MATCH EVAL — defaults below match the paper Cal-SPI
    # Point Maze setup so that a direct invocation of this script produces
    # the same behavior policy used by run_benchmark_pointmaze.sh).
    ap.add_argument("--continuing_task", type=int, default=0, choices=[0, 1])
    ap.add_argument("--reward_type", type=str, default="dense", choices=["sparse", "dense"])
    ap.add_argument("--reward_mode", type=str, default="speed_distance",
                    choices=["current", "speed", "speed_distance"])
    ap.add_argument("--living_cost", type=float, default=0.0)
    ap.add_argument("--success_bonus", type=float, default=1500.0)
    ap.add_argument("--step_penalty", type=float, default=1.0)
    ap.add_argument("--dist_penalty", type=float, default=0.2)

    # Suboptimality knobs (default 50,000 steps matches the paper Cal-SPI
    # Point Maze setup, Tab. ppo-config in the appendix).
    ap.add_argument("--total_timesteps", type=int, default=50_000)
    ap.add_argument("--net", type=str, default="32,32")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n_steps", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_range", type=float, default=0.1)
    ap.add_argument("--ent_coef", type=float, default=0.01)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    set_random_seed(args.seed)

    env = make_pointmaze_env(
        seed=args.seed,
        continuing_task=bool(args.continuing_task),
        reward_type=str(args.reward_type),
        disable_env_checker=True,
        reward_mode=str(args.reward_mode),
        living_cost=float(args.living_cost),
        success_bonus=float(args.success_bonus),
        step_penalty=float(args.step_penalty),
        dist_penalty=float(args.dist_penalty),
    )

    print(
        f"[TRAIN] continuing_task={bool(args.continuing_task)} "
        f"reward_type={args.reward_type} reward_mode={args.reward_mode} "
        f"living_cost={args.living_cost} success_bonus={args.success_bonus} "
        f"step_penalty={args.step_penalty} dist_penalty={args.dist_penalty}",
        flush=True,
    )
    print(f"[TRAIN] max_episode_steps={getattr(env, '_max_episode_steps', None)}", flush=True)

    net_arch = [int(x) for x in args.net.split(",") if x.strip()]
    policy_kwargs = dict(net_arch=net_arch)

    model = PPO(
        "MultiInputPolicy",
        env,
        seed=args.seed,
        verbose=1,
        gamma=float(args.gamma),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.lr),
        clip_range=float(args.clip_range),
        ent_coef=float(args.ent_coef),
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=int(args.total_timesteps))
    model.save(args.out)
    env.close()

    print("Saved PPO (suboptimal):", args.out, flush=True)


if __name__ == "__main__":
    main()
