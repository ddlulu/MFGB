# Changelog

## 0.1.0 - 2026-09-26

- Reorganized MFGB as an installable `src/`-layout Python package.
- Removed machine-specific paths, cache files, temporary outputs, and the
  non-runnable legacy script from the public release package.
- Added a relative-path quick test with a small example DEM.
- Added standard-library unit tests and a GitHub Actions workflow.
- Added input validation and explicit eight-direction ordering.
- Made convergence use maximum directional-weight change.
- Made Bayesian-optimization bounds caller-supplied and fixed the undefined
  bounds reference in the supplied calibration helper.
- Added publication-oriented documentation and the code-availability text.

