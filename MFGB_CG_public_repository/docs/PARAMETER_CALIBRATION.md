# Parameter calibration

Bayesian calibration is optional and requires:

```bash
python -m pip install -e ".[calibration]"
```

Call `mfgb.optimization.optimize_parameters` with explicit parameter bounds,
model inputs, the reference TCA array, initialization count, optimization
budget, and random seed. Bounds are intentionally not hard-coded because the
exact ranges must match those reported in the manuscript.

The default calibration loss is RMSLE. A different callable can be supplied
through `loss_function` when reproducing an experiment that used another
objective. Report the objective, bounds, acquisition setup, initialization
count, iteration budget, seed, and repeat strategy in the manuscript.

