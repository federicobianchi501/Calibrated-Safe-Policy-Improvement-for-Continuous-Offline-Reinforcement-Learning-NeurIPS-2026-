# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Append-only CSV/JSONL loggers and a JSON writer for experiment results."""
from __future__ import annotations
import csv
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Optional, Iterable


def _to_plain(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x


class CSVLogger:
    """Append-only CSV logger with a fixed column schema.

    The header is written when the file is created. Keys of a logged row that are not in
    `fieldnames` are ignored; missing keys are written as empty strings.
    """
    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """Append one row."""
        out = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(out)


class JSONLLogger:
    """Append-only JSON Lines logger; dataclasses are converted to dicts."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("")

    def log(self, obj: Dict[str, Any]) -> None:
        """Append one JSON record."""
        payload = _to_plain(obj)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """Write `payload` (dict or dataclass) to `path` as indented JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = _to_plain(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
