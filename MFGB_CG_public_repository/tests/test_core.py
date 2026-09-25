"""Automated tests for the public MFGB package."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfgb import MFGBModel, calculate_metrics, compute_tca
from mfgb.optimization import calibration_loss_rmsle


class MFGBTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[0:6, 0:6]
        self.dem = 50.0 - 2.0 * y - x
        gy, gx = np.gradient(self.dem)
        self.slopes = np.hypot(gx, gy)

    def test_model_executes_and_normalizes_weights(self) -> None:
        model = MFGBModel(
            self.dem,
            1.0,
            self.slopes,
            np.zeros_like(self.dem),
            max_iter=10,
        ).fit()
        self.assertEqual(model.tca.shape, self.dem.shape)
        self.assertTrue(np.isfinite(model.tca).all())
        sums = model.get_flow_directions().sum(axis=2)
        self.assertTrue(np.allclose(sums[sums > 0], 1.0))

    def test_compute_tca_rejects_wrong_flow_shape(self) -> None:
        with self.assertRaises(ValueError):
            compute_tca(np.zeros(self.dem.shape), self.dem, 1.0)

    def test_metrics(self) -> None:
        result = calculate_metrics(
            np.array([1.0, 2.1, 2.9]),
            np.array([1.0, 2.0, 3.0]),
        )
        self.assertGreaterEqual(result["RMSE"], 0.0)
        self.assertGreaterEqual(result["sMAPE"], 0.0)

    def test_calibration_loss(self) -> None:
        loss = calibration_loss_rmsle(
            np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )
        self.assertAlmostEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()

