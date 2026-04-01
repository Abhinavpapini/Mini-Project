# Experiment Execution Schedule (Core 35 + Post-35 Queue)

This schedule operationalizes the canonical tracker in results/experiment_tracker.csv.

## Current Execution Snapshot (2026-04-01)
- Completed so far: A1, A7, A8, B1, A2, A3, A4, A5, A6, C1, C2, C3, B7, C5, C6, F1, F4, D1, D4, D5, D2, D8, E1, E2, E3, E4
- **Total done: 27/35 core experiments**
- Next planned run: B2 *(Phase 4, Kernel PCA-RBF dimensionality benchmark)*
- Core35 progress: 27/35 complete

## Scope
- Core plan: 35 experiments (plan_order 1-35)
- Deferred queue: 5 experiments (plan_order 36-40)

## Rules
- Do not launch post-35 queue until all core35 runs are complete and reviewed.
- Every run must log run_id, config snapshot, fold, and metrics in the tracker.
- Promotion requires fold-mean gains and leakage-safe validation.

## Week-by-Week Plan

### Weeks 1-2 (Layer Baselines)
- A1 (must-run) [done]
- A7 (must-run) [done]
- A8 (optional) [done]

### Week 3 (Dimensionality Baseline)
- B1 (must-run) [done]

### Week 4 (Layer Aggregation)
- A2 (must-run) [done]
- A3 (must-run) [done]

### Week 5 (Extended Layer + Fusion Start)
- A4 (optional) [done]
- A5 (optional) [done]
- A6 (optional) [done]
- C1 (must-run) [done]

### Week 6 (Fusion + Production Baselines)
- C2 (must-run) [done]
- C3 (must-run) [done]
- C5 (must-run) [done]
- C6 (optional) [done]
- F1 (must-run) [done]
- F4 (must-run) [done]

### Weeks 7-8 (Deep Models) — CLOSED ✅
- D1 (must-run) [done] 
- D4 (must-run) [done]
- D5 (must-run) [done]
- D2 (optional) [done] 
- D8 (optional) [done] 

### Week 9 (Graph Entry) — CLOSED ✅
- E1 (must-run) [done] — Macro-F1=0.468, AUPRC=0.6023
- E2 (must-run) [done] — Macro-F1=0.515, AUPRC=0.6184 ⭐ BEST AUPRC + PROMOTED

### Week 10 (Graph Expansion) — CLOSED ✅
- E3 (must-run) [done] — Macro-F1=0.333 ⛔ FAIL — hyperedge over-smoothing
- E4 (must-run) [done] — Macro-F1=0.462 — temporal edges no benefit vs E1

### Week 11 (Dimensionality Reduction) ← NEXT
- B2 (must-run)
- B3 (must-run)
- B4 (must-run)
- B5 (must-run)

### Week 12 (Non-Linear Reduction II + Robustness)
- B6 (must-run)
- B7 (optional)
- B8 (optional)
- F2 (must-run)

### Week 13 (Speaker Robustness + Final Core Closure)
- F3 (must-run)
- F5 (must-run)
- Core35 closure review and freeze

## Post-35 Queue (Run Only After Core35)
- C4
- D3
- D6
- D7
- E5

## Deliverables at Core35 Completion
- Final promoted model shortlist
- Fold-level metrics and significance summary
- Locked config bundle and reproducibility note
