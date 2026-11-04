# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning

**Federico Bianchi, Alessandro Farinelli, Alberto Castellini**
NeurIPS 2026

Reference implementation of Cal-SPI and the scripts to reproduce the
experiments on Point Maze, Hopper, HalfCheetah, Walker2d and Ant.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.8, CPU only. Pinned dependencies are in `pyproject.toml`.

## Reproducing the experiments

```bash
bash run_benchmark_pointmaze.sh
bash run_benchmark_hopper.sh
bash run_benchmark_halfcheetah.sh
bash run_benchmark_walker2d.sh
bash run_benchmark_ant.sh
```

Each launcher runs all `(|D|, seed)` cells with the paper configuration and
writes results to `results_<domain>/`. The PPO baselines used in the paper are
in `models/`. Settings can be overridden through environment variables listed
at the top of each launcher, e.g.:

```bash
DATASET_SIZES_OVERRIDE="500" SEEDS_OVERRIDE="0" MAX_JOBS=1 bash run_benchmark_hopper.sh
```

## Citation

```bibtex
@inproceedings{bianchi2026CalSPI,
  title     = {Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning},
  author    = {Bianchi, Federico and Farinelli, Alessandro and Castellini, Alberto},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
