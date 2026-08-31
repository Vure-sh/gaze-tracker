# Orchestrator Plan: Real-Time Gaze Tracker Overhaul

## Objectives
Transform `/home/vure/gaze-tracker` into a production-grade, highly accurate, low-latency, modular, robust gaze tracking product meeting Requirements R1-R6 and all Acceptance Criteria.

## Phases

### Phase 0: Survey & Technical Baseline Audit
- Dispatch 3 parallel Explorers:
  1. `teamwork_preview_explorer` (CV & Pipeline): Audit `main.py`, `src/`, models, landmarks, head pose, feature extraction.
  2. `teamwork_preview_explorer` (ML & Performance): Audit calibration routines, regression models, persistence, latency/FPS bottlenecks.
  3. `teamwork_preview_spec_miner` (Requirements & Test Surface): Map all constraints, edge cases (blinks, occlusions, scale/roll invariance, display resolutions, CLI options).
- Aggregate findings into `PROJECT.md` (Feature Inventory, Architecture, Milestone Decomposition, Interface Contracts, Code Layout).

### Phase 1: Dual Track Execution
- **E2E Testing Track**:
  - Spawn E2E Testing Orchestrator / Subagents to build the independent 4-tier opaque-box test harness (`TEST_INFRA.md` and `TEST_READY.md`).
- **Implementation Track**:
  - **Milestone 1: CV & Robust Feature Extraction**: Iris normalization (scale/roll-invariant), 3D head pose decoupling via solvePnP, confidence & tracking quality metrics, blink/occlusion detection.
  - **Milestone 2: ML & Gaze Estimation / Calibration**: Multi-point calibration, saccade latency trimming, outlier rejection, regression models (Polynomial Ridge, SVR, Huber), holdout validation metrics (MAE/RMSE), profile serialization.
  - **Milestone 3: Temporal Filtering & Real-Time Performance**: Tuned One-Euro filter + velocity-aware smoothing, zero micro-jitter + responsive saccades, async/efficient frame acquisition & processing, FPS >= 30-60, frame latency < 35ms.
  - **Milestone 4: UX, Visualization, CLI & Debug HUD**: Screen gaze canvas (cursor + trailing gaze heat + animated pulsing targets), camera debug HUD with 3D pose axes & telemetry, keyboard controls and CLI flags.

### Phase 2: Final Milestone & Verification
- **Phase 2A (E2E Pass)**: Execute 100% of Tiers 1-4 E2E tests, fixing any regressions or interface mismatches.
- **Phase 2B (Adversarial Hardening)**: Tier 5 adversarial stress testing via Challenger, white-box coverage audit, gap closing.
- **Phase 2C (Forensic Integrity & Audit)**: Clean audit verdict from Forensic Auditor (`teamwork_preview_auditor`).
- **Phase 2D (Documentation & Handoff)**: Complete `README.md` with architectural diagram, math, benchmarks, and write final handoff report.
