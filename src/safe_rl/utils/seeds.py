# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Global seeding helper."""
import numpy as np
import torch
import random

def set_seed(seed):
    """Seed the NumPy, PyTorch and Python `random` global generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
