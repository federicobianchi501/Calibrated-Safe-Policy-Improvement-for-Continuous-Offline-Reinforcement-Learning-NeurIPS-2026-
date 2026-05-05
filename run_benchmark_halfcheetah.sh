#!/bin/bash
# Reproduce the HalfCheetah experiments from the paper.
#
# Configuration:
#   - 4 dataset sizes:    |D| in {50, 100, 250, 500} trajectories
#   - 5 random seeds:     {0, 1, 2, 3, 4}
#   - 3 evaluated methods: ppo_only, calibrated_usad, mcts_only_no_gate
#   - calibration:        Wilks-strict B*, triple-split, q_eps=0.95
#   - planner:            MCTS-DPW (numba backend),
#                         num_sims=300, max_depth=5, eval_episodes=15
#   - PPO baseline:       200k training steps, used as fallback policy and
#                         source of behavior trajectories
#
# Each (ds, seed) cell trains a fresh ensemble + calibrator from the offline
# dataset, then evaluates the three methods on the true HalfCheetah-v5 env.
#
# Outputs (per cell): results_halfcheetah/ds<DS>/seed<S>/{cache, ep_ckpt,
# results.csv}
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
SCRIPT="scripts/offline/continuous/benchmark_continuous.py"
DOMAIN="halfcheetah"
OUT_ROOT="${OUT_ROOT:-results_halfcheetah}"
PPO_PATH="${PPO_PATH:-models/ppo_halfcheetah.zip}"

DATASET_SIZES=(${DATASET_SIZES_OVERRIDE:-50 100 250 500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4})
RUN_METHODS="${RUN_METHODS:-ppo_only calibrated_usad mcts_only_no_gate}"

MAX_JOBS="${MAX_JOBS:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"

# Sweep_C-replicated config: K=4 members, hidden=(512, 512), MSE
NUM_SIMS=300
MAX_DEPTH=5
EVAL_EPISODES=15
GAMMA=0.99
MAX_STEPS=1000
PPO_TIMESTEPS=200000
BASELINE_EPOCHS=1000000
ENSEMBLE_EPOCHS=100
ENSEMBLE_HIDDEN_DIMS="512,512"
ENSEMBLE_SIZE=4
DPW_K=10.0
DPW_ALPHA=0.5
C_PUCT=1.5
N_ACTIONS_ROOT=200
N_ACTIONS_PER_NODE=24
PPO_STD_SCALE=0.1
CALIB_FRAC=0.2
CALIB_BINS=0
CALIB_DELTA=0.05
CALIB_MIN_SAMPLES=0
CALIB_TRIPLE_SPLIT=1
Q_EPS=0.95
CSPI_E_PLAN_SCALE=0.1
CSPI_FROZEN_ROOT=1
CSPI_N_MIN_ROOT_VISITS=3
MCTS_BACKEND="numba"

mkdir -p "$OUT_ROOT/logs"
mkdir -p "$(dirname "$PPO_PATH")"

# Train the PPO baseline once, synchronously, before parallelizing cells.
if [ ! -f "$PPO_PATH" ]; then
    echo "[LAUNCHER] Training PPO baseline -> $PPO_PATH"
    "$PYTHON" scripts/offline/continuous/train_ppo_continuous.py \
        --domain halfcheetah --seed 0 --out "$PPO_PATH" \
        --total_timesteps "$PPO_TIMESTEPS"
else
    echo "[LAUNCHER] PPO checkpoint already present at $PPO_PATH"
fi

run_job() {
    local DS=$1 SEED=$2
    local OUTDIR="$OUT_ROOT/ds${DS}/seed${SEED}"
    local LOG="$OUT_ROOT/logs/ds${DS}_seed${SEED}.log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
    "$PYTHON" "$SCRIPT" \
        --domain "$DOMAIN" --outdir "$OUTDIR" \
        --ppo_global_path "$PPO_PATH" --ppo_timesteps "$PPO_TIMESTEPS" \
        --dataset_trajs "$DS" --seeds "$SEED" \
        --max_steps "$MAX_STEPS" --num_sims "$NUM_SIMS" --max_depth "$MAX_DEPTH" \
        --gamma "$GAMMA" --eval_episodes "$EVAL_EPISODES" \
        --calib_frac "$CALIB_FRAC" --calib_bins "$CALIB_BINS" --calib_delta "$CALIB_DELTA" \
        --calib_min_samples "$CALIB_MIN_SAMPLES" --calib_triple_split "$CALIB_TRIPLE_SPLIT" \
        --q_eps "$Q_EPS" \
        --baseline_epochs "$BASELINE_EPOCHS" \
        --ensemble_epochs "$ENSEMBLE_EPOCHS" \
        --ensemble_hidden_dims "$ENSEMBLE_HIDDEN_DIMS" --ensemble_size "$ENSEMBLE_SIZE" \
        --ensemble_normalize 0 --ensemble_bootstrap 0 \
        --dpw_k "$DPW_K" --dpw_alpha "$DPW_ALPHA" --c_puct "$C_PUCT" \
        --n_actions_root "$N_ACTIONS_ROOT" --n_actions_per_node "$N_ACTIONS_PER_NODE" \
        --ppo_std_scale "$PPO_STD_SCALE" \
        --cspi_margin_mult 1.0 \
        --cspi_v_rollout_horizon 0 --cspi_v_rollout_n_samples 1 \
        --cspi_e_plan_scale "$CSPI_E_PLAN_SCALE" \
        --cspi_delta_piB 0.0 \
        --cspi_frozen_root "$CSPI_FROZEN_ROOT" \
        --cspi_n_min_root_visits "$CSPI_N_MIN_ROOT_VISITS" \
        --use_ppo_leaf_value 0 \
        --eval_workers "$EVAL_WORKERS" \
        --mcts_backend "$MCTS_BACKEND" \
        --run_methods $RUN_METHODS \
        > "$LOG" 2>&1 \
        && echo "[DONE] ds=$DS seed=$SEED" \
        || echo "[FAIL] ds=$DS seed=$SEED  (see $LOG)"
}

export -f run_job
export PYTHON SCRIPT DOMAIN OUT_ROOT PPO_PATH PPO_TIMESTEPS
export NUM_SIMS MAX_DEPTH EVAL_EPISODES GAMMA MAX_STEPS BASELINE_EPOCHS
export ENSEMBLE_EPOCHS ENSEMBLE_HIDDEN_DIMS ENSEMBLE_SIZE
export DPW_K DPW_ALPHA C_PUCT N_ACTIONS_ROOT N_ACTIONS_PER_NODE PPO_STD_SCALE
export CALIB_FRAC CALIB_BINS CALIB_DELTA CALIB_MIN_SAMPLES CALIB_TRIPLE_SPLIT Q_EPS
export CSPI_E_PLAN_SCALE CSPI_FROZEN_ROOT CSPI_N_MIN_ROOT_VISITS
export EVAL_WORKERS MCTS_BACKEND RUN_METHODS

JOBS=()
for DS in "${DATASET_SIZES[@]}"; do
    for SEED in "${SEEDS[@]}"; do JOBS+=("$DS $SEED"); done
done
echo "[LAUNCHER] HalfCheetah: ${#JOBS[@]} jobs, MAX_JOBS=$MAX_JOBS"

printf '%s\n' "${JOBS[@]}" | xargs -P "$MAX_JOBS" -I{} bash -c 'run_job $1 $2' _ {}

echo "[LAUNCHER] Aggregating results..."
"$PYTHON" scripts/offline/continuous/aggregate_continuous_results.py --root "$OUT_ROOT" \
    && echo "[DONE] aggregated -> $OUT_ROOT/aggregated" \
    || echo "[WARN] aggregation failed"

echo "[LAUNCHER] HalfCheetah run complete."
