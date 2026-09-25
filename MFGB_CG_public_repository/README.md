# MFGB: Multi-Factor Game-Based Flow Routing

This repository contains a runnable Python implementation of the
Multi-Factor Game-Based (MFGB) algorithm for routing flow over digital
elevation models (DEMs). It accompanies the MFGB manuscript submitted to
*Computers & Geosciences*.

MFGB represents each valid DEM cell by an eight-direction allocation vector.
The allocation is updated iteratively using terrain evidence, neighbouring
path organization, accumulation-dependent regulation, and curvature guidance.
The resulting allocation field is used to recompute total contributing area
(TCA), which feeds back into the next iteration.

## Repository contents

```text
MFGB_CG_public_repository/
├── src/mfgb/                 Core routing, metrics, and calibration helpers
├── examples/                 Quick test and its small example DEM
├── tests/                    Automated unit tests
├── configs/                  Example parameter file
├── docs/                     Method map and release checklist
├── requirements.txt          Minimal runtime dependency
├── environment.yml           Optional Conda environment
├── pyproject.toml            Installable package metadata
└── LICENSE                   MIT open-source license
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/ddlulu/MFGB.git
cd MFGB
python -m pip install -r requirements.txt
python -m pip install -e .
```

Bayesian calibration is optional. Install its extra dependency with:

```bash
python -m pip install -e ".[calibration]"
```

## Quick test

The repository includes a small DEM at `examples/example_dem.csv`. Run:

```bash
python examples/quick_test.py
```

Successful execution ends with:

```text
QUICK TEST PASSED
```

and creates:

```text
examples/output/mfgb_tca.csv
examples/output/summary.json
```

The quick test verifies installation, routing, finite output, and flow-weight
normalization. It is deliberately small and is not a reproduction of the
article's full analytical-surface experiments.

## Running the automated tests

```bash
python -m unittest discover -s tests -v
```

No test framework beyond the Python standard library is required.

## Minimal Python example

```python
import numpy as np
from mfgb import MFGBModel

dem = np.loadtxt("examples/example_dem.csv", delimiter=",")
gy, gx = np.gradient(dem, 1.0)
slopes = np.hypot(gx, gy)
curvature_direction = np.zeros_like(dem)

model = MFGBModel(
    dem=dem,
    cell_size=1.0,
    slopes=slopes,
    curvature_direction=curvature_direction,
    max_iter=50,
    tolerance=1e-6,
).fit()

tca = model.tca
directional_weights = model.get_flow_directions()
```

## Inputs

`MFGBModel` requires four inputs:

- `dem`: two-dimensional elevation array; `-9999.0` denotes NoData;
- `cell_size`: positive DEM cell size;
- `slopes`: array with the same shape as the DEM;
- `curvature_direction`: directional-curvature array in radians.

Optional arrays (`tpis`, `trds`, `roughness`, `ecvs`, and `kts`) can be passed
when those terrain attributes are used. Inputs must be spatially aligned and
have the same shape. The code assumes that terrain attributes have been
prepared using the normalization and sign conventions reported by the study.

## Outputs

After `fit()`, the main outputs are:

- `model.tca`: TCA array in area units;
- `model.get_flow_directions()`: `(rows, columns, 8)` allocation array;
- `model.iterations_run`: number of completed iterations;
- `model.converged`: whether the weight-change tolerance was reached.

Direction order is east, southeast, south, southwest, west, northwest, north,
and northeast.

## Calibration

`mfgb.optimization.optimize_parameters` provides optional Bayesian
calibration. Parameter bounds are caller-supplied by design so the ranges used
for a study are explicit. See `docs/PARAMETER_CALIBRATION.md`.

## Reproducing article experiments

The included quick test checks software functionality. Reproducing numerical
values in the article additionally requires the analytical/field DEMs,
reference TCA arrays, preprocessing settings, calibrated parameters, and
experiment drivers used for the reported tables and figures. Materials that
can legally be redistributed should be added under `data/` or linked with a
persistent public identifier before archival release.

## License

This repository is distributed under the MIT License. See `LICENSE`.

## Citation and contact

Please cite the associated MFGB article when using this code. Add the final
article citation and corresponding-author contact details here before archival
publication.

