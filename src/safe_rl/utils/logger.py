# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
import csv
import os

class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "reward", "value_loss", "policy_loss"])

    def log(self, step, reward, value_loss, policy_loss):
        self.writer.writerow([step, reward, value_loss, policy_loss])
        self.file.flush()
