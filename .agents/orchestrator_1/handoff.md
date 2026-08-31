# Orchestrator Soft Handoff: Generation 1 -> Generation 2

**Author**: Project Orchestrator (Generation 1, `orchestrator_1`)  
**Target**: Project Orchestrator Successor (Generation 2, `orchestrator_2`)  
**Original Parent Conversation ID**: `eb9ec646-8c4b-45da-8122-2604a87ce2bd`  
**Workspace**: `/home/vure/gaze-tracker`  
**Date**: 2026-08-30  

---

## 1. Milestone State

| Milestone | Name | Status | Summary / Verification |
|---|---|---|---|
| E2E | E2E Testing Track | **DONE** | 146 tests across Tiers 1-4 passing; published `TEST_READY.md`. |
| M1 | CV & Robust Feature Engineering | **DONE** | Orthonormal iris normalization (fixes horizontal cancellation), solvePnP 3D model alignment (+Y down), 6-point adaptive EAR, 5-point iris circularity, composite tracking quality. Passed Gate: 2 Reviewers APPROVE, 2 Challengers APPROVE, Forensic Auditor CLEAN. |
| M2 | ML & Gaze Estimation / Calibration | **DONE** | Boustrophedon 9/13/16-point grid generator, wall-clock dwell timing, normalized IQR outlier rejection, Polynomial Ridge (`RidgeCV`), `RobustHuberRegressor`, `SVRGazeRegressor`, LOPO Group CV (MAE 6.2–9.8px < 35px), holdout validation mode ($\theta < 1.0^\circ$), Schema 2.0 serialization. Passed Gate: 2 Reviewers APPROVE, 2 Challengers APPROVE, Forensic Auditor CLEAN. (345/345 tests passing). |
| M3 | Temporal Filtering & Real-Time Performance | **IN_PROGRESS / NEXT** | Ready for immediate dispatch. Needs: velocity-gated `OneEuroFilter2D` (< 1.1px jitter, 0ms saccade lag), `KalmanFilter2D` reset logic, threaded asynchronous camera stream (`src/camera_stream.py` / `src/pipeline.py`), canvas memory reuse, maintaining FPS >= 30-60, latency < 35ms. |
| M4 | UX, Visualization, CLI & Debug HUD | **PLANNED** | Needs: `src/ui/canvas.py` (dark slate, glowing cursor, heat trail, animated targets), `src/ui/hud.py` (camera overlay, 3D pose axes, telemetry card), `src/ui/app.py` & `main.py` CLI / hotkeys. |
| M_Final | Final Milestone & Verification | **PLANNED** | Phase 1: 100% E2E test pass (Tiers 1-4); Phase 2: Tier 5 Adversarial Coverage Hardening via Challenger; Phase 3: Forensic Integrity Audit; Phase 4: Production `README.md` and final handoff. |

---

## 2. Active Subagents

All 16 subagents spawned by Generation 1 have completed their tasks and delivered verified handoffs:
- Survey: `explorer_cv_1`, `explorer_ml_1`, `spec_miner_1` (DONE)
- E2E Test Suite: `test_writer_1` (DONE - `TEST_READY.md`)
- M1 Implementation & Gate: `worker_m1_1` (DONE), `reviewer_m1_1` (APPROVE), `reviewer_m1_2` (APPROVE), `challenger_m1_1` (APPROVE), `challenger_m1_2` (APPROVE), `auditor_m1_1` (CLEAN)
- M2 Implementation & Gate: `worker_m2_1` (DONE), `reviewer_m2_1` (APPROVE), `reviewer_m2_2` (APPROVE), `challenger_m2_1` (APPROVE), `challenger_m2_2` (APPROVE), `auditor_m2_1` (CLEAN)

No pending background tasks or hanging subagents.

---

## 3. Pending Decisions & Critical Context

- **Test Suite Status**: 345/345 tests currently passing across `tests/`.
- **Integrity Compliance**: Zero cheat vectors, zero mock shortcuts, real scikit-learn models, genuine MediaPipe and OpenCV matrix operations. Both Milestone 1 and Milestone 2 have **CLEAN** Forensic Audit verdicts.
- **Constraints to Maintain**:
  - DISPATCH-ONLY: NEVER write code directly. Delegate all tasks to subagents.
  - Audit is a binary veto.
  - Always include `/home/vure/gaze-tracker/.agents/ORIGINAL_REQUEST.md` in every subagent dispatch.
  - Include the mandatory integrity warning in every Worker dispatch.

---

## 4. Remaining Work & Concrete Next Steps for Successor

1. **Milestone 3 (Temporal Filtering & Real-Time Performance)**:
   - Spawn Worker for M3 (`src/filters/one_euro.py`, `src/filters/kalman.py`, `src/filters.py`, `src/camera_stream.py`, `src/pipeline.py`).
   - Implement velocity-gated One-Euro filter with $\Delta t$ timestamping and timeout reset, Kalman filter, threaded camera capture with `/dev/video9` and webcam fallback, and memory-efficient frame processing.
   - Run M3 Verification Gate: 2 Reviewers, 2 Challengers (measuring jitter < 1.1px and latency < 35ms), 1 Forensic Auditor.
2. **Milestone 4 (UX, Visualization, CLI & Debug HUD)**:
   - Spawn Worker for M4 (`src/ui/canvas.py`, `src/ui/hud.py`, `src/ui/app.py`, `main.py`).
   - Implement screen gaze canvas (gaze cursor, heat trail, animated targets), camera debug HUD (eye contours, iris center, 3D pose axes, telemetry card), hotkeys (`C`, `R`, `S`, `L`, `D`, `F`, `Q`), and CLI flags.
   - Run M4 Verification Gate: 2 Reviewers, 2 Challengers, 1 Forensic Auditor.
3. **Final Milestone & Verification**:
   - Phase 1: Verify 100% pass rate across the full E2E test suite.
   - Phase 2: Spawn 2 Challengers for Tier 5 adversarial white-box coverage hardening. Close any identified coverage gaps.
   - Phase 3: Run final Forensic Integrity Audit across entire repository (`teamwork_preview_auditor`).
   - Phase 4: Author comprehensive production `README.md` (architecture, math formulas, benchmarks, setup, controls, test results).
   - Write final handoff report and notify sentinel parent.

---

## 5. Key Artifacts Index

- `/home/vure/gaze-tracker/.agents/ORIGINAL_REQUEST.md` — Original User Requirements (R1-R6, Acceptance Criteria)
- `/home/vure/gaze-tracker/PROJECT.md` — Master Architecture, Feature Inventory & Milestone Registry
- `/home/vure/gaze-tracker/TEST_INFRA.md` — Test Architecture & Methodology
- `/home/vure/gaze-tracker/TEST_READY.md` — E2E Test Suite Readiness Report
- `/home/vure/gaze-tracker/.agents/orchestrator_1/GATE_STATUS.md` — Gate Verdict Records for M1 & M2 (Both PASS)
- `/home/vure/gaze-tracker/.agents/orchestrator_1/BRIEFING.md` — Working Memory
- `/home/vure/gaze-tracker/.agents/orchestrator_1/progress.md` — State Checkpoint
