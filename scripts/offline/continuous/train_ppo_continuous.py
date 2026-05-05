# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
train_ppo_continuous.py — Train a suboptimal PPO behavior policy for Walker2d or Hopper.

Intentionally suboptimal (small network, few timesteps) to match the Cal-SPI
experimental setup: a reasonable but imperfect behavior policy from which we
want to improve.

Usage:
    python train_ppo_continuous.py --domain walker2d --seed 0 --out models/ppo_walker2d_100k.zip
    python train_ppo_continuous.py --domain hopper   --seed 0 --out models/ppo_hopper_100k.zip
"""
import argparse
import os
import sys

# Ensure src is on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from safe_rl.envs.continuous_env_factory import make_continuous_env


def main():
    ap = argparse.ArgumentParser(description="Train suboptimal PPO for Walker2d/Hopper")
    ap.add_argument("--domain", type=str, required=True,
                    choices=["walker2d", "hopper", "halfcheetah", "reacher", "swimmer"],
                    help="Domain to train on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True,
                    help="Path to save PPO checkpoint (.zip)")

    # Suboptimality knobs — smaller/fewer = more suboptimal.
    # Default 70k matches the paper Hopper PPO baseline; HalfCheetah uses 200k
    # (set via --total_timesteps when invoking from run_benchmark_halfcheetah.sh).
    ap.add_argument("--total_timesteps", type=int, default=70_000,
                    help="Total environment steps for training "
                         "(default 70k matches paper Hopper).")
    ap.add_argument("--net", type=str, default="64,64",
                    help="Hidden layer sizes (smaller = more suboptimal)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.0)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    set_random_seed(args.seed)

    env = make_continuous_env(args.domain, args.seed)

    net_arch = [int(x) for x in args.net.split(",") if x.strip()]
    policy_kwargs = dict(net_arch=net_arch)

    print(
        f"[TRAIN] domain={args.domain} seed={args.seed} "
        f"total_timesteps={args.total_timesteps} net={net_arch}",
        flush=True,
    )

    model = PPO(
        "MlpPolicy",
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

    print(f"[TRAIN] Saved PPO checkpoint → {args.out}", flush=True)


if __name__ == "__main__":
    main()
