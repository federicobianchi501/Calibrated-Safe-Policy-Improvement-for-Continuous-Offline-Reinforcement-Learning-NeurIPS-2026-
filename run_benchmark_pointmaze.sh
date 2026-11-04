#!/bin/bash
# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
# Reproduce the Point Maze experiments (PointMaze_UMazeDense-v3).
#
# Configuration (Tables 17-19 of the paper):
#   - dataset sizes:      |D| in {1, 5, 250, 500} trajectories
#   - seeds:              {0, 1, 2, 3, 4}
#   - eval episodes:      40 per seed, at most 300 steps each
#   - methods:            ppo_only, calibrated_usad, mcts_only_no_gate,
#                         iql, td3bc, morel_full, mopo_mse, cql, bear, cal_ql
#   - calibration:        triple split, candidate bins {20,40,60,80},
#                         q_des=0.95, alpha=0.05, delta=0.05
#   - planner:            M_sim=200, H=20, kappa=2.0, alpha_pw=0.5,
#                         K_root=64, K_node=16, c_puct=1.5, sigma=0.1
#   - ensemble:           K=4, MLP [512,512], MSE, 50 epochs
#   - PPO baseline:       [32,32], 40k timesteps (models/ppo_pointmaze.zip)
#
# Each (ds, seed) cell collects its dataset, trains the ensemble and the
# calibrator, and evaluates the methods on the true environment.
# Output per cell: $OUT_ROOT/ds<DS>/seed<S>/
# The aggregator then writes $OUT_ROOT/aggregated/.
#
# Overrides: DATASET_SIZES_OVERRIDE, SEEDS_OVERRIDE, RUN_METHODS, MAX_JOBS,
# MCTS_BACKEND (array | parametric), PYTHON.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
SCRIPT="scripts/offline/pointmaze/benchmark_pointmaze.py"
TRAIN_PPO="scripts/offline/pointmaze/train_ppo_pointmaze.py"
OUT_ROOT="${OUT_ROOT:-results_pointmaze}"
PPO_PATH="${PPO_PATH:-models/ppo_pointmaze.zip}"

# Dataset sizes and seeds
DATASET_SIZES=(${DATASET_SIZES_OVERRIDE:-1 5 250 500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4})

# Cal-SPI, ungated MCTS-DPW, PPO and the offline RL baselines.
# Override with RUN_METHODS env var to restrict to a subset, e.g.:
#   RUN_METHODS="ppo_only calibrated_usad mcts_only_no_gate" bash run_benchmark_pointmaze.sh
RUN_METHODS="${RUN_METHODS:-ppo_only calibrated_usad mcts_only_no_gate iql td3bc morel_full mopo_mse cql bear cal_ql}"

# Number of training epochs for the offline-RL baselines (IQL/TD3+BC/MOReL/MOPO/CQL/BEAR/Cal-QL).
# Set to 0 to skip baselines (only Cal-SPI / no-gate / PPO will be evaluated).
BASELINE_EPOCHS="${BASELINE_EPOCHS:-100000}"

# Parallelism: maximum number of concurrent (ds, seed) cells
MAX_JOBS="${MAX_JOBS:-10}"
MCTS_BACKEND="${MCTS_BACKEND:-array}"

# Reward and environment configuration
REWARD_TYPE="dense"
REWARD_MODE="speed_distance"
STEP_PENALTY=1.0
DIST_PENALTY=0.2
SUCCESS_BONUS=1500.0
LIVING_COST=0.0
CONTINUING_TASK=0
PPO_TIMESTEPS=40000

# Planner and calibration hyperparameters
NUM_SIMS=200
MAX_DEPTH=20
GAMMA=0.99
EVAL_EPISODES=40
MAX_STEPS=300
DPW_K=2.0
DPW_ALPHA=0.5
C_PUCT=1.5
N_ACTIONS_ROOT=64
N_ACTIONS_PER_NODE=16
PPO_STD_SCALE=0.1
CALIB_FRAC=0.2
CALIB_BINS=0
CALIB_DELTA=0.05
CALIB_MIN_SAMPLES=0
EPS_QUANTILE=0.95
TAU=0.2

mkdir -p "$OUT_ROOT/logs"
mkdir -p "$(dirname "$PPO_PATH")"

# Train the PPO baseline once, synchronously, with the same reward configuration
# the benchmark uses for evaluation. Doing this here (instead of letting each
# parallel cell train its own PPO) avoids a race when MAX_JOBS > 1.
if [ ! -f "$PPO_PATH" ]; then
    echo "[LAUNCHER] Training PPO baseline -> $PPO_PATH"
    "$PYTHON" "$TRAIN_PPO" --seed 0 --out "$PPO_PATH" \
        --continuing_task "$CONTINUING_TASK" --reward_type "$REWARD_TYPE" \
        --reward_mode "$REWARD_MODE" --step_penalty "$STEP_PENALTY" \
        --dist_penalty "$DIST_PENALTY" --success_bonus "$SUCCESS_BONUS" \
        --living_cost "$LIVING_COST" --total_timesteps "$PPO_TIMESTEPS"
else
    echo "[LAUNCHER] PPO checkpoint already present at $PPO_PATH"
fi

run_job() {
    local SEED=$1 DS=$2
    local OUTDIR="${OUT_ROOT}/ds${DS}/seed${SEED}"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
    "$PYTHON" "$SCRIPT" \
        --outdir "$OUTDIR" \
        --train_ppo_script "$TRAIN_PPO" \
        --ppo_train_once 1 --ppo_global_path "$PPO_PATH" \
        --ppo_train_seed 0 --ppo_timesteps "$PPO_TIMESTEPS" \
        --reward_type "$REWARD_TYPE" --reward_mode "$REWARD_MODE" \
        --step_penalty "$STEP_PENALTY" --dist_penalty "$DIST_PENALTY" \
        --success_bonus "$SUCCESS_BONUS" --living_cost "$LIVING_COST" \
        --continuing_task "$CONTINUING_TASK" \
        --dataset_trajs "$DS" --seeds "$SEED" \
        --num_sims "$NUM_SIMS" --max_depth "$MAX_DEPTH" --gamma "$GAMMA" \
        --dpw_k "$DPW_K" --dpw_alpha "$DPW_ALPHA" --c_puct "$C_PUCT" \
        --n_actions_root "$N_ACTIONS_ROOT" --n_actions_per_node "$N_ACTIONS_PER_NODE" \
        --ppo_std_scale "$PPO_STD_SCALE" \
        --calib_frac "$CALIB_FRAC" --calib_bins "$CALIB_BINS" \
        --calib_delta "$CALIB_DELTA" --calib_min_samples "$CALIB_MIN_SAMPLES" \
        --eps_from_calibration 1 --eps_quantile "$EPS_QUANTILE" --tau "$TAU" \
        --eval_episodes "$EVAL_EPISODES" --max_steps "$MAX_STEPS" \
        --mcts_backend "$MCTS_BACKEND" \
        --run_methods $RUN_METHODS \
        --save_trajs_for_ds 1 5 250 500 --traj_eval_episodes 3 \
        --load_from_cache 0 \
        --baseline_epochs "$BASELINE_EPOCHS" \
        --debug_episodes 0 --debug_log_sims 0 \
        > "${OUT_ROOT}/logs/ds${DS}_seed${SEED}.log" 2>&1 \
        && echo "[DONE] ds=$DS seed=$SEED" \
        || echo "[FAIL] ds=$DS seed=$SEED  (see ${OUT_ROOT}/logs/ds${DS}_seed${SEED}.log)"
}

export -f run_job
export PYTHON SCRIPT TRAIN_PPO PPO_PATH OUT_ROOT
export REWARD_TYPE REWARD_MODE STEP_PENALTY DIST_PENALTY SUCCESS_BONUS
export LIVING_COST CONTINUING_TASK PPO_TIMESTEPS
export NUM_SIMS MAX_DEPTH GAMMA EVAL_EPISODES MAX_STEPS
export DPW_K DPW_ALPHA C_PUCT N_ACTIONS_ROOT N_ACTIONS_PER_NODE PPO_STD_SCALE
export CALIB_FRAC CALIB_BINS CALIB_DELTA CALIB_MIN_SAMPLES
export EPS_QUANTILE TAU RUN_METHODS BASELINE_EPOCHS MCTS_BACKEND

JOBS=()
for DS in "${DATASET_SIZES[@]}"; do
    for SEED in "${SEEDS[@]}"; do JOBS+=("$SEED $DS"); done
done
echo "[LAUNCHER] PointMaze: ${#JOBS[@]} jobs, MAX_JOBS=$MAX_JOBS, backend=$MCTS_BACKEND"

printf '%s\n' "${JOBS[@]}" | xargs -P "$MAX_JOBS" -I{} bash -c 'run_job $1 $2' _ {}

echo "[LAUNCHER] Aggregating results..."
"$PYTHON" scripts/offline/pointmaze/aggregate_pointmaze_results.py \
    --root "$OUT_ROOT" --outdir "${OUT_ROOT}/aggregated" \
    && echo "[DONE] aggregated -> ${OUT_ROOT}/aggregated" \
    || echo "[WARN] aggregation failed"

echo "[LAUNCHER] PointMaze run complete."
