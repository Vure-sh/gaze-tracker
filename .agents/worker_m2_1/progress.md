# Progress Log — Milestone 2: ML & Gaze Estimation / Calibration

- Last visited: 2026-08-30T19:03:00Z
- Status: Complete

## Steps
1. [x] Audit baseline tests and interface requirements (272 tests passing).
2. [x] Implement `src/calibration/targets.py` (Grid Generator with Boustrophedon ordering & validation targets).
3. [x] Implement `src/calibration/calibrator.py` & update `src/calibrator.py` (Wall-clock dwell, saccade trimming, normalized IQR outlier filtering, 4/5-point holdout validation mode).
4. [x] Implement `src/models/regressor.py` & update `src/gaze_regressor.py` (BaseGazeRegressor, PolynomialRidgeRegressor with RidgeCV, RobustHuberRegressor, SVRGazeRegressor, LOPO Group CV).
5. [x] Implement `src/models/serializer.py` (Schema 2.0 serialization, metadata, validation, backward-compat legacy pickle loading).
6. [x] Implement `src/calibration/__init__.py` and `src/models/__init__.py`.
7. [x] Write extensive unit and integration tests covering M2 calibration, models, holdout validation, and serialization (`tests/test_m2_calibration_models.py`).
8. [x] Run full test suite (`pytest`) and verify 100% pass rate (290/290 passed in 13.33s).
9. [x] Write hard handoff report `handoff.md`.
