# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning

This repository accompanies the NeurIPS 2026 submission *Calibrated Safe Policy
Improvement for Continuous Offline Reinforcement Learning*. It contains the
reference implementation of **Cal-SPI** and the scripts needed to reproduce the
main empirical results on Point Maze, Hopper, and HalfCheetah.

## Abstract

Safe Policy Improvement (SPI) aims to improve a baseline policy offline from fixed data while avoiding performance degradation with high probability. Existing SPI methods work in discrete domains, but they do not extend to continuous control. We propose Calibrated Safe Policy Improvement (Cal-SPI), a Monte Carlo Tree Search-based continuous SPI method that replaces count-based support with calibrated model trust. Cal-SPI learns a deep ensemble dynamics model and calibrates ensemble disagreement on held-out data through a Wilks-style tolerance construction, turning raw uncertainty into a statistical certificate for one-step model error. This certificate defines a hard gate inside model-based tree search. We provide a theoretical analysis showing that, under an explicit transfer condition from held-out calibration data to the root actions considered at decision time, calibrated model trust and baseline-relative switching criterion yield a PAC-style improvement bound. Experiments across three MuJoCo domains show that Cal-SPI avoids performance degradation under limited data and improves over the baseline once calibrated model trust becomes reliable.

## What this code reproduces

The three launcher scripts in the project root reproduce the main paper
results across the evaluated MuJoCo domains. Each launcher trains a fresh
ensemble and calibrator from offline data for every `(dataset size, seed)`
pair, then evaluates the requested set of methods on the true environment.

| Script | Domain | Dataset sizes | Seeds | Methods evaluated |
|---|---|---|---|---|
| `run_benchmark_pointmaze.sh` | Point Maze (`PointMaze_UMazeDense-v3`) | `1, 5, 250, 500` trajectories | `0..4` | `ppo_only`, `calibrated_usad`, `mcts_only_no_gate`, `iql`, `td3bc`, `morel_full`, `mopo_mse`, `cql`, `bear`, `cal_ql` |
| `run_benchmark_hopper.sh` | `Hopper-v5` | `5, 50, 500, 1000` | `0..4` | `ppo_only`, `calibrated_usad`, `mcts_only_no_gate`, `iql`, `td3bc`, `morel_full`, `mopo_mse`, `cql`, `bear`, `cal_ql` |
| `run_benchmark_halfcheetah.sh` | `HalfCheetah-v5` | `50, 100, 250, 500` | `0..4` | `ppo_only`, `calibrated_usad`, `mcts_only_no_gate` (gate stress test) |

Point Maze and Hopper run the full benchmark with all model-free and
model-based offline-RL baselines. HalfCheetah is presented in the paper as an
additional stress test focused on the calibrated gate, comparing only PPO, Cal-SPI,
and ungated MCTS-DPW under the same learned dynamics model and search budget.

All three benchmarks share the same gate construction:
finite-candidate Wilks admissibility (`B†` selected from the candidate set
`{20, 40, 60, 80}`), triple-split calibration (`D_bin / D_wilks / D_thr`),
piecewise-constant bound `B(u)` with monotonicity enforcement, and operational
threshold `epsilon = quantile_{0.95}(B(u_n))` — i.e., calibration confidence
`alpha = 0.05`, within-bin coverage slack `delta = 0.05`, and target gate
quantile `q_des = 0.95`.

## Repository layout

```
Calibrated_Safe_Policy_Improvement/
├── README.md
├── requirements.txt
├── run_benchmark_pointmaze.sh        # reproduce PointMaze results
├── run_benchmark_hopper.sh           # reproduce Hopper results
├── run_benchmark_halfcheetah.sh      # reproduce HalfCheetah results
├── src/
│   └── safe_rl/
│       ├── algorithms/
│       │   └── offline/
│       │       ├── cal_spi/          # Cal-SPI MCTS-DPW planner (paper contribution)
│       │       │   ├── mcts_parametric.py    # reference (Python)
│       │       │   ├── mcts_batched.py       # batched-eval optimization
│       │       │   ├── mcts_numba.py         # numba-accelerated kernel for continuous envs
│       │       │   └── mcts_cspi.py          
│       │       ├── iql.py            # baseline: Implicit Q-Learning
│       │       ├── td3bc.py          # baseline: TD3 + behavior cloning
│       │       ├── morel_full.py     # baseline: MOReL 
│       │       ├── mopo.py           # baseline: MOPO
│       │       ├── bear.py           # baseline: BEAR
│       │       ├── cql.py            # baseline: Conservative Q-Learning
│       │       └── cal_ql.py         # baseline: Cal-QL
│       ├── envs/
│       │   ├── pointmaze_factory.py        # PointMaze env factory + speed/distance reward
│       │   ├── cloneable_pointmaze_wrapper.py  # state cloning for MCTS rollouts
│       │   ├── continuous_env_factory.py   # MuJoCo locomotion env factory (Hopper, HalfCheetah)
│       │   ├── continuous_env_wrapper.py   # state cloning for MuJoCo MCTS rollouts
│       │   └── wrappers.py                  # generic wrappers (reward shaping, time limit)
│       ├── offline/
│       │   ├── ensemble_dynamics_sb3.py    # K-member MSE-trained dynamics ensemble
│       │   ├── ensemble_numba.py            # numba-friendly ensemble views (continuous envs)
│       │   ├── uncertainty_calibration.py  # Wilks-strict piecewise-constant bound B(u)
│       │   ├── model_env_pointmaze.py      # learned-dynamics PointMaze (dict obs + goal)
│       │   └── model_env_continuous.py     # learned-dynamics MuJoCo locomotion (flat obs)
│       ├── data/                            # dataset collection + train/calibration split
│       ├── evaluation/                      # episode-level evaluator with checkpointing
│       ├── experiments/                     # adapters between agents and the eval harness
│       └── utils/                           # config / logging / seed utilities
├── scripts/
│   └── offline/
│       ├── pointmaze/
│       │   ├── benchmark_pointmaze.py             # main PointMaze experiment driver
│       │   ├── train_ppo_pointmaze.py             # train sub-optimal PPO baseline
│       │   └── aggregate_pointmaze_results.py    # aggregate per-seed results into CSV + plots
│       └── continuous/
│           ├── benchmark_continuous.py            # main Hopper / HalfCheetah experiment driver
│           ├── train_ppo_continuous.py            # train PPO baselines for continuous domains
│           └── aggregate_continuous_results.py   # aggregate per-seed results into CSV + plots
└── models/                                  # pretrained PPO baselines (created by the user)
```

## Installation

```bash
# Create a fresh environment (Python 3.8+).
python -m venv .venv
source .venv/bin/activate

# Install the package and all dependencies in editable mode.
pip install -e .
```

Dependencies are declared in `pyproject.toml` and pinned to the versions used
for the paper (PyTorch 2.4.1, Stable-Baselines3 2.4.1, Gymnasium 1.0.0,
Gymnasium-Robotics 1.3.1, MuJoCo 3.1.6, numpy 1.24.4, numba 0.58.1, and
scipy/pandas/matplotlib). A flat `requirements.txt` mirroring these pins is
also provided for `pip install -r requirements.txt` workflows.

MuJoCo locomotion environments (Hopper, HalfCheetah) require a working MuJoCo
installation; the `mujoco` Python wheel installs everything needed on Linux.

## Reproducing the experiments

### Step 1 — Train the PPO behavior / baseline policies

Cal-SPI uses a sub-optimal PPO policy as both the behavior policy (for offline
data collection) and the fallback. Train one PPO baseline per domain. PPO
hyperparameters follow Table 7 of the paper: a `[32, 32]` actor-critic on
Point Maze trained for 50k steps, and a `[64, 64]` actor-critic on Hopper
trained for 70k steps. The HalfCheetah PPO baseline is trained for 200k steps.

The launcher scripts in Step 2 train the PPO baseline themselves on first
invocation, so you can skip this step. Train manually only if you want to
reuse a single PPO checkpoint across multiple result directories. The
PointMaze training command must match the reward configuration used at
evaluation; the locomotion domains use the standard MuJoCo reward.

```bash
mkdir -p models

# Point Maze: [32,32] PPO for 50k steps with the dense + speed_distance reward
# the benchmark uses for evaluation. (All flags below match the script's
# paper-aligned defaults; they are listed explicitly only for clarity.)
python scripts/offline/pointmaze/train_ppo_pointmaze.py \
    --out models/ppo_pointmaze.zip --total_timesteps 50000 \
    --reward_type dense --reward_mode speed_distance \
    --step_penalty 1.0 --dist_penalty 0.2 --success_bonus 1500.0 \
    --living_cost 0.0 --continuing_task 0

# Hopper: [64,64] PPO for 70k steps.
python scripts/offline/continuous/train_ppo_continuous.py \
    --domain hopper --out models/ppo_hopper.zip --total_timesteps 70000

# HalfCheetah: PPO for 200k steps.
python scripts/offline/continuous/train_ppo_continuous.py \
    --domain halfcheetah --out models/ppo_halfcheetah.zip --total_timesteps 200000
```

### Step 2 — Run the benchmarks

Each launcher orchestrates the full sweep over `(dataset size, seed)` pairs
in parallel. The default parallelism is `MAX_JOBS=10` on Point Maze and
HalfCheetah, and `MAX_JOBS=5` on Hopper. Override with the `MAX_JOBS`
environment variable. Each launcher uses the planner and calibrator
hyperparameters reported in Tables 5 and 6 of the paper:

- **Point Maze.** 10 evaluation episodes per seed, max 300 steps per episode,
  ensemble of `K=4` MLP-`[512, 512]` MSE predictors trained for 50 epochs,
  MCTS-DPW with `M_sim=200` simulations per root, planning horizon `H=20`,
  DPW exponent `alpha_pw=0.5`, `kappa=2.0`, `K_root=64` root candidates,
  `K_node=16` non-root cap, `c_puct=1.5`, action-sampling scale
  `sigma=0.1`, leaf value at horizon `0`.
- **Hopper.** 20 evaluation episodes per seed, max 1000 steps per episode,
  ensemble of `K=4` MLP-`[512, 512]` MSE predictors trained for 200 epochs,
  MCTS-DPW with `M_sim=1000` simulations per root, planning horizon `H=10`,
  DPW exponent `alpha_pw=0.5`, `kappa=2.0`, `K_root=96` root candidates,
  `K_node=24` non-root cap, `c_puct=1.5`, action-sampling scale
  `sigma=0.1`, leaf value at horizon `0`.

```bash
bash run_benchmark_pointmaze.sh
bash run_benchmark_hopper.sh
bash run_benchmark_halfcheetah.sh
```

Each launcher writes per-cell outputs to `results_<domain>/ds<DS>/seed<SEED>/`
and finally invokes the aggregator to produce
`results_<domain>/aggregated/aggregated_results_absolute.csv` and per-metric
SVG plots.

### Step 3 — Inspect the results

The aggregated CSV contains one row per `(dataset_trajs, method)` pair with
mean and standard deviation across seeds for the following metrics:

- `return_mean` — mean episodic return (true environment).
- `cvar_15` — Conditional Value-at-Risk at the 15% tail.
- `success_rate` — task-success indicator (1 if goal reached / agent
  survived the full episode, 0 otherwise).
- `xvel_mean` — mean forward velocity (locomotion domains only).
- `root_fallback_rate` — fraction of root MCTS calls where the policy
  fell back to the PPO baseline action.
- `root_coverage_mean` — fraction of root candidate actions certified
  safe by the calibrated gate.

## Quick sanity check

To validate the installation in a few minutes, run a small PointMaze sweep:

```bash
DATASET_SIZES_OVERRIDE="5 250" SEEDS_OVERRIDE="0" \
    bash run_benchmark_pointmaze.sh
```

This evaluates two dataset sizes on a single seed; the run should finish
in 10–20 minutes on a modern CPU and produces results in `results_pointmaze/`.

## Compute requirements

The full sweep used in the paper takes the following wall-clock time on a
24-core CPU machine (no GPU required):

- PointMaze (4 ds × 5 seeds × 10 methods): **~12–18 hours** with `MAX_JOBS=10`.
- Hopper (4 ds × 5 seeds × 10 methods): **~24 hours** with `MAX_JOBS=5`.
- HalfCheetah (4 ds × 5 seeds × 3 methods): **~16 hours** with `MAX_JOBS=10`.

The `mcts_only_no_gate` variant on continuous locomotion is the slowest cell
because the policy is unfiltered and the cheetah/hopper drift into
out-of-distribution regions where the MCTS expansion does not contract.

## Output checkpointing and caching

Each `(ds, seed)` cell produces:

- `cache/seed_<S>/ds_<DS>/ensemble.pt` — trained ensemble weights.
- `cache/seed_<S>/ds_<DS>/calibrator_usad.pkl` — calibrated bound `B(u)`.
- `cache/seed_<S>/ds_<DS>/calibrator_epistemic.pkl` — variance-score variant.
- `cache/seed_<S>/ds_<DS>/transitions_{full,train,calib}.npz` — split.
- `ep_ckpt/<method>.jsonl` — per-episode return, length, success, fallback
  fraction. Episodes are appended atomically: re-running a partially
  completed cell resumes from where it stopped.
- `results.csv` — one row per evaluated method with the summary metrics.
- `results_partial.jsonl` — per-method summary written incrementally.

If `--load_from_cache 1` is passed and all cache artifacts are present, the
benchmark skips dataset collection and ensemble training and goes straight
to evaluation.

## Citation

This code is anonymized for double-blind review. Citation information will be
added on acceptance.

## License

This code is released for academic peer review.
