# Method-to-code map

| MFGB element | Main implementation |
|---|---|
| Eight-direction allocation vector | `_FlowAgent.weights` in `src/mfgb/core.py` |
| M1: terrain-informed evidence | `_FlowAgent.calculate_payoff` terrain terms |
| M2: path structure | same-direction neighbour interaction and upstream consistency |
| M3: flow regulation | TCA-dependent congestion term and `compute_tca` feedback |
| M4: curvature guidance | curvature-direction consistency term |
| Strategy update | `_FlowAgent.update_strategy` |
| Fixed-point iteration | `MFGBModel.run_iteration` and `MFGBModel.fit` |
| TCA generation | `compute_tca` |
| Evaluation metrics | `src/mfgb/metrics.py` |
| Bayesian calibration | `src/mfgb/optimization.py` |

The solver reports convergence when the maximum change in any directional
allocation weight is below the configured tolerance. This is a numerical
fixed-point criterion; the code does not claim or test a Nash equilibrium.

