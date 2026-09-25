"""Optional Bayesian-calibration helpers for MFGB."""

from __future__ import annotations

from typing import Callable, Dict
import numpy as np

from .core import MFGBModel




def calibration_loss_rmsle(
    simulated: np.ndarray,
    reference: np.ndarray,
    invalid: float = -9999.0,
) -> float:
    mask = (simulated != invalid) & (reference != invalid) & np.isfinite(simulated) & np.isfinite(reference)
    s = np.maximum(simulated[mask], 0.0)
    r = np.maximum(reference[mask], 0.0)
    if s.size == 0:
        raise ValueError("No valid cells available for calibration.")
    return float(np.sqrt(np.mean((np.log1p(s) - np.log1p(r)) ** 2)))


def optimize_parameters(
    model_kwargs: Dict,
    tca_reference: np.ndarray,
    bounds: Dict[str, tuple],
    loss_function: Callable[[np.ndarray, np.ndarray], float] = calibration_loss_rmsle,
    init_points: int = 30,
    n_iter: int = 90,
    random_state: int = 42,
) -> Dict[str, float]:
    """Calibrate MFGB parameters within caller-supplied bounds.

    Bounds are deliberately required: the study-specific search ranges must be
    reported by the paper and should not be silently replaced by defaults.
    """
    try:
        from bayes_opt import BayesianOptimization
    except ImportError as exc:
        raise ImportError("Install bayesian-optimization to use optimize_parameters().") from exc

    if not bounds:
        raise ValueError("bounds must contain at least one parameter range")

    def objective(**params):
        model = MFGBModel(params=params, **model_kwargs).fit()
        return -loss_function(model.tca, tca_reference)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=bounds,
        random_state=random_state,
        verbose=2,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=int(init_points), n_iter=int(n_iter))
    return {k: float(v) for k, v in optimizer.max["params"].items()}
