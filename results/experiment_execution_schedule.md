# Experiment Execution Schedule (Core 35 + Post-35 Queue)

This schedule operationalizes the canonical tracker in results/experiment_tracker.csv.

## Scope
- Core plan: 35 experiments (plan_order 1-35)
- Deferred queue: 5 experiments (plan_order 36-40)

## Rules
- Do not launch post-35 queue until all core35 runs are complete and reviewed.
- Every run must log run_id, config snapshot, fold, and metrics in the tracker.
- Promotion requires fold-mean gains and leakage-safe validation.

## Week-by-Week Plan

### Weeks 1-2 (Layer Baselines)
- A1 (must-run)
- A7 (must-run)
- A8 (optional)

### Week 3 (Dimensionality Baseline)
- B1 (must-run)

### Week 4 (Layer Aggregation)
- A2 (must-run)
- A3 (must-run)

### Week 5 (Extended Layer + Fusion Start)
- A4 (optional)
- A5 (optional)
- A6 (optional)
- C1 (must-run)

### Week 6 (Fusion + Production Baselines)
- C2 (must-run)
- C3 (must-run)
- C5 (must-run)
- C6 (optional)
- F1 (must-run)
- F4 (must-run)

### Weeks 7-8 (Deep Models)
- D1 (must-run)
- D2 (optional)
- D4 (must-run)
- D5 (must-run)
- D8 (optional)

### Week 9 (Graph Entry)
- E1 (must-run)
- E2 (must-run)

### Week 10 (Graph Expansion)
- E3 (must-run)
- E4 (must-run)

### Week 11 (Non-Linear Reduction I)
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
