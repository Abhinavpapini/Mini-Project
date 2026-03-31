# SEP-28k Stutter Type Classification: Final Execution Board

Scope: 35 experiments across layer analysis, dimensionality reduction, fusion, deep classifiers, graph models, and robustness.
Primary metric: Macro-F1 under speaker-independent evaluation.

## 1) Final Objective
Deliver one reproducible final system for SEP-28k that:
- Maximizes Macro-F1 on BL, PR, SR, WR, IJ and fluent.
- Handles multi-label co-occurrence.
- Generalizes under speaker-independent LOSO-style splits.
- Includes ablation-backed evidence for layer choice and dimensionality choice.

## 2) Final Ordered Pipeline (Dependency-Correct)
1. P0 Foundation and Reproducibility
2. A1-lite Quick Layer Scan
3. B1 Focused PCA Sweep on shortlisted layers
4. A2 and A3 and A4 Layer Aggregation with best reduced dimensions
5. C1 and C2 Fusion Baselines
6. F1 and F4 Imbalance and Multi-label Production Baselines
7. D1 and D5 Advanced Deep Models first, then D2 and D4
8. E1 and E2 Graph Models first, then E4 and E3 and E5
9. B2-B8 Nonlinear and sparse reductions
10. F2 and F3 and F5 robustness extensions
11. Final ablations, significance tests, model freeze, report tables

Rationale: this order avoids double work by caching all SSL layers once, then reusing them for A, B, C, D, E, and F.

## 3) Artifact Contract (Must Keep Fixed)
- Splits: artifacts/splits/
- Cached SSL features: artifacts/features/{ssl_model}/{fold}/layer_{k}.parquet
- Reducers: artifacts/reducers/{method}/{ssl_model}/{fold}/
- Trained checkpoints: artifacts/checkpoints/{exp_id}/{run_id}/
- Predictions: artifacts/predictions/{exp_id}/{run_id}.parquet
- Metrics: results/metrics_master.csv
- Figures: results/figures/
- Config snapshots: results/configs/{run_id}.yaml

If any path or schema changes, update all experiments before running more jobs.

## 4) Phase Gate Checklist

### P0 Foundation
- [x] Implement canonical label loader for binary and multi-label targets.
- [ ] Implement LOSO-style speaker-independent split generator. *(partial — fold0 used, full LOSO manifests pending)*
- [ ] Freeze split manifests and store in artifacts/splits/. *(pending)*
- [x] Implement unified evaluator: Macro-F1, per-class F1, Precision, Recall, AUPRC.
- [x] Implement run logger with seed, fold, model, layer, reducer, classifier, loss, augmentation.
- [x] Add deterministic seed control and environment capture.

Output required:
- split_manifest.csv
- metrics schema file
- one smoke-test run log

Stop/Go criteria:
- Go only if two repeated smoke runs with same seed produce matching metrics within tiny tolerance.

### A1-lite Quick Layer Scan ✅ DONE (2026-03-25)
- [x] Run single-layer probing on 3 core SSL models first: HuBERT-large, WavLM-large, Wav2Vec2-large.
- [x] Use same small classifier head for fairness.
- [x] Produce per-layer Macro-F1 and per-class F1 curves.
- [x] Keep top 2 to 3 layers per model. *(Layer 9 selected as best for wav2vec2-base)*

Output required:
- results/figures/a1_layer_curves.png
- results/tables/a1_top_layers.csv

Stop/Go criteria:
- Go only after stable top-layer shortlist is identified on validation folds.

### B1 Focused PCA Sweep ✅ DONE (2026-03-25)
- [x] Run PCA only on A1-shortlisted layers.
- [x] Sweep dimensions 512, 256, 128, 64, 32, 16, 8.
- [x] Run fine search around best point plus/minus 4 dims where applicable.
- [x] Select best dimension per model and stutter type view. *(Best dim=32 selected)*

Output required:
- results/figures/b1_pca_sweep.png
- results/tables/b1_best_dims.csv

Stop/Go criteria:
- Go only if reduced dimensions match or beat full-dim baseline Macro-F1 on validation.

### A2 and A3 and A4 Layer Aggregation ✅ DONE (2026-03-25)
- [x] A2 weighted-sum layer interface with softmax weights. *(F1=0.559, layer 6 dominant)*
- [x] A3 Gumbel softmax hard selection with temperature schedule. *(F1=0.514, hard-selected layer 9)*
- [x] A4 dimension-wise Gumbel layer selection. *(F1=0.000 — classifier collapsed, not promoted)*
- [x] Run with best dims from B1 where applicable.

Output required:
- results/figures/a2_layer_weight_heatmap.png
- results/tables/a3_a4_selection_stats.csv

Stop/Go criteria:
- Promote only methods with consistent fold gains over A1 plus B1 baseline.

### C1 and C2 Fusion Baselines ✅ DONE
- [x] C1 SSL plus MFCC with PCA controls. *(DONE 2026-03-25, best MFCC dim=52, F1=0.527)*
- [x] C2 SSL plus handcrafted feature groups: articulatory, phonatory, temporal. *(DONE 2026-03-26, best=ssl+zcr, F1=0.555)*
- [x] Perform feature-group ablations. *(energy, zcr, jitter, formant — ZCR wins)*

Output required:
- results/tables/c_fusion_results.csv
- results/figures/c2_ablation_bars.png

Stop/Go criteria:
- Keep fusion branch only if Macro-F1 gain is consistent and not only from one fold.

### F1 and F4 Production Baselines
- [ ] F1 focal loss and class-balanced sampling.
- [ ] F4 full multi-label setup with sigmoid head and BCE-style objective.
- [ ] Evaluate co-occurrence-heavy subset separately.

Output required:
- results/tables/f1_f4_results.csv
- results/tables/f4_cooccurrence_subset.csv

Stop/Go criteria:
- Continue only after minority-class F1 improves without severe precision collapse.

### D Advanced Deep Models
- [ ] D1 Conformer baseline with selected features.
- [ ] D5 multi-task detection plus type head.
- [ ] D2 BiLSTM attention and D4 Transformer as challengers.

Output required:
- results/tables/d_model_comparison.csv
- results/figures/d_attention_examples.png

Stop/Go criteria:
- Keep top 2 deep models by mean Macro-F1 and fold stability.

### E Graph Models
- [ ] E1 temporal graph plus GCN.
- [ ] E2 dynamic GAT.
- [ ] E4 ST-GCN then E3 hypergraph then E5 graph transformer if budget permits.

Output required:
- results/tables/e_graph_results.csv
- results/figures/e_graph_sensitivity.png

Stop/Go criteria:
- Continue graph branch only if it beats best non-graph baseline on at least 3 folds.

### B2-B8 and F2 and F3 and F5 Extensions
- [ ] B2 KPCA
- [ ] B3 low-rank MDS
- [ ] B4 autoencoder bottleneck
- [ ] B5 VAE latent
- [ ] B6 UMAP reduction
- [ ] B7 sparse PCA
- [ ] B8 ICA
- [ ] F2 mixup and SpecAugment
- [ ] F3 adversarial speaker disentanglement
- [ ] F5 speed perturbation and pitch shift

Output required:
- results/tables/extensions_results.csv

Stop/Go criteria:
- Keep only extensions that improve Macro-F1 and speaker-independence metrics together.

### Finalization
- [ ] Run end-to-end ablation of winning system.
- [ ] Run significance tests across folds.
- [ ] Freeze best config and seed bundle.
- [ ] Prepare final report tables and figures.

Output required:
- results/tables/final_main_table.csv
- results/tables/final_ablation_table.csv
- results/tables/final_significance.csv
- best_model_card.md

## 5) Experiment Priority
Must-run first:
- A1, B1, A2, C1, C2, F1, F4, D1, D5, E1, E2, F3

Run next if time and compute allow:
- A3, A4, A5, A6, A7, A8, B2-B8, C3-C6, D2, D3, D4, D6, D7, D8, E3, E4, E5, F2, F5

## 6) Daily Operating Rules
- Every run must have unique run_id and saved config.
- No experiment runs on ad-hoc splits.
- No model comparison is valid without same folds and same metrics schema.
- Promote methods by fold-mean performance, not single best run.
- Keep compute log: start time, end time, GPU, memory, failures.

## 7) Leakage and Label-Policy Guards (Mandatory)
- Fit PCA, scaler, normalizer, and any reducer on train fold only, then transform val and test.
- Do not tune hyperparameters on test folds; test is for one final locked evaluation pass.
- Keep one label aggregation policy fixed for all experiments; if policy changes, version it and rerun compared methods.
- Ensure speaker IDs do not cross train/val/test within a fold.
- For multi-label setup, fix threshold strategy globally (default 0.5 or tuned on validation only).

Output required:
- artifacts/splits/fold_integrity_report.csv
- results/tables/label_policy_version.csv

Stop/Go criteria:
- Stop immediately if any speaker leakage or train-test preprocessing leakage is detected.

## 8) Quantitative Promotion Thresholds
- Candidate beats baseline if fold-mean Macro-F1 gain is at least +0.01.
- Candidate must not degrade minority-class mean F1 by more than 0.005.
- Candidate must improve or match at least 3 out of 5 folds.
- Candidate must pass significance test (paired bootstrap or paired t-test) with p < 0.05.
- For multi-label runs, also require non-decreasing micro-F1 and AUPRC.

Output required:
- results/tables/promotion_decisions.csv

## 9) Compute Budget and Fail-Fast Rules
- Stage expensive runs behind one-fold pilot checks before full-fold launch.
- Abort long runs early if validation Macro-F1 is below baseline minus 0.02 after 30 percent of planned epochs.
- Limit each branch to top 2 promoted variants before entering next phase.
- Reserve at least 20 percent compute budget for final ablations and reruns.

## 10) Status Update (2026-03-31 — 14/35 core experiments done)

### ✅ Completed (plan_order 1–14)
A1, A2, A3, A4, A5, A6, A7, A8, B1, B7, C1, C2, C3, C5

### C5 Final Results (all 5 targets)
| Target | F1 | Accuracy | AUROC | Dominant |
|---|---|---|---|---|
| Block | 0.6311 | 0.6528 | 0.6590 | HuBERT |
| Prolongation | 0.6168 | 0.7367 | 0.7249 | HuBERT |
| SoundRep | 0.6093 | 0.7948 | 0.7800 | Whisper |
| WordRep | 0.5679 | 0.8089 | 0.7633 | HuBERT |
| Interjection | 0.7160 | 0.7936 | 0.7780 | Whisper |
| **Mean** | **0.6282** | **0.7574** | **0.7410** | Tied |

### 🔜 Next Up — Week 6 (remaining)
1. **C6** — Speaker embedding fusion (HuBERT + ECAPA-TDNN, optional)
2. **F1** — Focal loss + class-balanced sampling (must-run)
3. **F4** — Multi-label sigmoid head + BCE (must-run)

### Remaining Core-35 Pipeline
- Weeks 7–8: D1, D2, D4, D5, D8 (deep models)
- Weeks 9–10: E1, E2, E3, E4 (graph models)
- Weeks 11–12: B2–B6, B8, F2 (non-linear reductions)
- Week 13: F3, F5, core-35 closure + final ablations

## 11) Completion Definition
Project is complete only when all are true:
- Reproducible best run can be re-executed with same metrics.
- Final model beats baseline on Macro-F1 and minority stutter-type F1.
- Multi-label and speaker-independent evaluations are both reported.
- Ablation and significance evidence is included.
- All outputs are saved under results/ and artifacts/ with run traceability.
