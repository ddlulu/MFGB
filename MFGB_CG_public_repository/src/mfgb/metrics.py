"""Evaluation metrics used in the MFGB experiments."""

from __future__ import annotations

from typing import Dict
import numpy as np


def calculate_metrics(pred: np.ndarray, ref: np.ndarray, invalid: float = -9999.0) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.shape != ref.shape:
        raise ValueError(f"pred shape {pred.shape} does not match ref shape {ref.shape}")
    mask = np.isfinite(pred) & np.isfinite(ref) & (pred != invalid) & (ref != invalid)
    p = pred[mask]
    r = ref[mask]
    if p.size == 0:
        raise ValueError("No valid cells available for metric calculation.")

    diff = p - r
    eps = 1e-12
    rmse = np.sqrt(np.mean(diff ** 2))
    smape = np.mean(2.0 * np.abs(diff) / (np.abs(p) + np.abs(r) + eps))
    # Matches the manuscript sign convention: (reference - estimate) / reference.
    mre = np.mean((r - p) / (r + eps))
    rmsle = np.sqrt(np.mean((np.log1p(np.maximum(p, 0)) - np.log1p(np.maximum(r, 0))) ** 2))
    return {
        "RMSE": float(rmse),
        "sMAPE": float(smape),
        "MRE": float(mre),
        "RMSLE": float(rmsle),
        "n": int(p.size),
    }
