"""Run a small, self-contained MFGB functionality check."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfgb import MFGBModel


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    dem = np.loadtxt(example_dir / "example_dem.csv", delimiter=",")
    cell_size = 1.0

    gy, gx = np.gradient(dem, cell_size)
    slopes = np.hypot(gx, gy)
    curvature_direction = np.zeros_like(dem)

    model = MFGBModel(
        dem=dem,
        cell_size=cell_size,
        slopes=slopes,
        curvature_direction=curvature_direction,
        max_iter=50,
        tolerance=1e-6,
    ).fit()

    flow = model.get_flow_directions()
    sums = flow.sum(axis=2)
    routed = sums > 0

    assert model.tca.shape == dem.shape
    assert np.isfinite(model.tca).all()
    assert (model.tca >= 0).all()
    assert np.allclose(sums[routed], 1.0, atol=1e-10)

    output_dir = example_dir / "output"
    output_dir.mkdir(exist_ok=True)
    np.savetxt(output_dir / "mfgb_tca.csv", model.tca, delimiter=",", fmt="%.8f")

    summary = {
        "dem_shape": list(dem.shape),
        "iterations_run": model.iterations_run,
        "converged": model.converged,
        "maximum_tca": float(model.tca.max()),
        "flow_weight_sums_valid": bool(np.allclose(sums[routed], 1.0, atol=1e-10)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"DEM shape: {dem.shape[0]} x {dem.shape[1]}")
    print(f"Iterations: {model.iterations_run}")
    print(f"Converged: {model.converged}")
    print(f"Maximum TCA: {model.tca.max():.6f}")
    print(f"Output: {output_dir}")
    print("QUICK TEST PASSED")


if __name__ == "__main__":
    main()

