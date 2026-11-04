# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Numba planners with a scale-aware disagreement score.

When the ensemble is fitted with ``scale_aware_uncertainty``, USAD, the epistemic
score and the calibration error are computed on deltas divided by the
per-dimension ``delta_scale``. The kernels here apply the same scaling inside the
planner, so the gate tests the statistic the calibrator bounded. Tree search,
transitions and the CSPI switch rule are inherited from ``mcts_numba``.

Selected with ``backend="numba_scaled"`` in the continuous-control benchmark.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numba import njit

from safe_rl.algorithms.offline.cal_spi.mcts_numba import (
    OfflineMCTS_DPW_PPO_Numba,
    OfflineMCTS_DPW_PPO_Numba_CSPI,
)


@njit(cache=True, fastmath=True)
def usad_from_deltas_scaled(deltas, inv_scale):
    """max_{i<j} || (delta_i - delta_j) * inv_scale ||_2 over ensemble members."""
    K, D = deltas.shape
    best = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            acc = 0.0
            for d in range(D):
                t = (deltas[i, d] - deltas[j, d]) * inv_scale[d]
                acc += t * t
            if acc > best:
                best = acc
    return np.sqrt(best)


@njit(cache=True, fastmath=True)
def epistemic_from_deltas_scaled(deltas, inv_scale):
    """Mean over dimensions of the scaled per-dimension variance across members."""
    K, D = deltas.shape
    acc = 0.0
    for d in range(D):
        m = 0.0
        for k in range(K):
            m += deltas[k, d]
        m /= K
        v = 0.0
        for k in range(K):
            t = (deltas[k, d] - m) * inv_scale[d]
            v += t * t
        acc += v / K
    return acc / D


class _ScaledUSADMixin:
    """Overrides the disagreement computations of the numba planner with scaled versions."""

    def configure_scale(self, delta_scale: Optional[np.ndarray]) -> None:
        """Store 1 / max(delta_scale, 1e-8); raises if the ensemble has no scale."""
        if delta_scale is None:
            raise ValueError(
                "backend='numba_scaled' requires the ensemble to expose "
                "delta_scale; fit it with scale_aware_uncertainty=True or load "
                "a checkpoint that stores it."
            )
        sc = np.asarray(delta_scale, dtype=np.float32).reshape(-1)
        self._inv_scale = (1.0 / np.maximum(sc, 1e-8)).astype(np.float32)

    def _usad_from_cached_deltas(self) -> float:
        """Scaled USAD of the deltas in the planner's internal buffer."""
        return float(usad_from_deltas_scaled(self._deltas, self._inv_scale))

    def _uncertainty_raw(self, obs, action) -> float:
        """Scaled USAD or epistemic score; other types use the parent implementation."""
        ut = str(self.cfg.uncertainty_type)
        if ut == "usad":
            deltas = self._compute_deltas(obs, action)
            return float(usad_from_deltas_scaled(deltas, self._inv_scale))
        if ut == "epistemic":
            deltas = self._compute_deltas(obs, action)
            return float(epistemic_from_deltas_scaled(deltas, self._inv_scale))
        return super()._uncertainty_raw(obs, action)


class OfflineMCTS_DPW_PPO_Numba_Scaled(_ScaledUSADMixin, OfflineMCTS_DPW_PPO_Numba):
    """Plain numba planner with the scale-aware disagreement score."""

    def __init__(self, *args, delta_scale: Any = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_scale(delta_scale)


class OfflineMCTS_DPW_PPO_Numba_CSPI_Scaled(_ScaledUSADMixin,
                                            OfflineMCTS_DPW_PPO_Numba_CSPI):
    """Deployed CSPI planner with the scale-aware disagreement score."""

    def __init__(self, *args, delta_scale: Any = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_scale(delta_scale)
