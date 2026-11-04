# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""CSV logger for scalar training statistics."""
import csv
import os

class Logger:
    """Writes one row per call to a CSV file with columns step, reward, value_loss, policy_loss."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "reward", "value_loss", "policy_loss"])

    def log(self, step, reward, value_loss, policy_loss):
        """Append one row and flush to disk."""
        self.writer.writerow([step, reward, value_loss, policy_loss])
        self.file.flush()
