# BRIEFING — 2026-08-30T15:36:00Z

## Mission
Analyze ML, calibration pipeline, gaze regression models, validation metrics, profile serialization, and temporal filtering in the gaze-tracker project, producing a comprehensive findings and handoff report.

## 🔒 My Identity
- Archetype: explorer
- Roles: ML & Calibration Pipeline Explorer
- Working directory: /home/vure/gaze-tracker/.agents/explorer_ml_1
- Original parent: babbda7c-3a89-4322-9d89-2c0ab8c31a32
- Milestone: ML & Calibration Architecture Audit & Proposal

## 🔒 Key Constraints
- Read-only investigation — do NOT modify production source code
- Produce structured 5-component handoff report at /home/vure/gaze-tracker/.agents/explorer_ml_1/handoff.md
- Send message to parent orchestrator upon completion

## Current Parent
- Conversation ID: babbda7c-3a89-4322-9d89-2c0ab8c31a32
- Updated: 2026-08-30T15:36:00Z

## Investigation State
- **Explored paths**: `src/config.py`, `src/calibrator.py`, `src/gaze_regressor.py`, `src/filters.py`, `src/eye_extractor.py`, `src/head_pose.py`, `src/face_mesh_detector.py`, `src/visualizer.py`, `main.py`, `models/calibration_model.pkl`.
- **Key findings**:
  1. Critical Feature Bug: Left vs Right eye horizontal vectors are oppositely directed (nasal-to-temporal), causing `avg_norm_x` to be static constant 0.500 regardless of horizontal gaze direction.
  2. Over-parameterization: Degree-2 polynomial expansion on 14D feature vector creates 120 features; with 9 calibration points (~200 samples), this overfits training clusters and leads to edge extrapolation error. 8D-10D clean feature set generates 45-66 terms, yielding faster fit/predict and better generalization.
  3. Regression Benchmarks: `RidgeCV` and `HuberRegressor` outperform un-tuned Ridge (`alpha=1.0`) and SVR (which collapses outside point bandwidth with LOPO MAE > 135px). `RidgeCV` achieves ~13.0px LOPO MAE, fitting in 12ms and predicting in 0.21µs.
  4. Flawed Validation: Existing `metrics` reports training resubstitution error on the 9 training clusters. Needs Leave-One-Point-Out (LOPO) spatial cross-validation and a dedicated live 4-point/5-point holdout validation mode reporting pixel MAE/RMSE and visual angle degrees.
  5. One-Euro Filter Tuning & Timestamping: Default `beta=0.6` amplifies landmark noise into high velocity, letting jitter pass through (fixation StdDev 3.36px on 4px noise). Tuning `beta=0.01-0.05`, `min_cutoff=0.1-0.3Hz`, and adding a velocity deadband/gate suppresses fixation jitter by >70% while maintaining 0ms saccade latency. Missing frame timestamp argument in `main.py` causes timer race conditions.
  6. Model Serialization: Raw pickle without versioning, schema checks, or resolution normalization ($[0, 1]$ screen coordinates).
  7. Performance Bottlenecks: MediaPipe in `IMAGE` mode (instead of `VIDEO` tracking), synchronous `cap.read()` camera blocking, and per-frame full-canvas allocation overhead.
- **Unexplored areas**: None within ML/Calibration scope.

## Key Decisions Made
- Fully benchmarked regression models, feature compositions, filters, and cross-validation strategies.
- Synthesized clear architectural recommendations, interfaces, and concrete formulas for implementation.

## Artifact Index
- /home/vure/gaze-tracker/.agents/explorer_ml_1/handoff.md — Comprehensive findings and handoff report
