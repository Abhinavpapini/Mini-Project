# Project Explanation: Stuttering Detection Benchmark

This document explains the full project and each experiment: intent, method, key results, and takeaway. Metrics: Macro-F1 is the primary balance metric; AUPRC is the calibration/ranking metric.

## Project Goal
Build and compare models for multi-label stutter event detection (Block, Prolongation, SoundRep, WordRep, Interjection) on SEP-28k clips using SSL features, dimensionality reduction, fusion, deep classifiers, graph models, and robustness techniques.

## Series A: SSL Layer/Feature Aggregation

### A1: SSL Layer Sweep
- Goal: Find the best HuBERT layer for stutter classification.
- Method: Extract HuBERT-large layers and evaluate PCA-32 + classifier per layer.
- Result: Best performance at layer 9.
- Takeaway: Intermediate layers are most informative for stutter events.

### A2: Weighted Layer Sum
- Goal: Learn a weighted blend of layers instead of a single layer.
- Method: Trainable weighted sum of SSL layers, then classify.
- Result: Macro-F1 0.559.
- Takeaway: Learned mixing helps but gains are modest.

### A3: Gumbel Layer Selection
- Goal: Select one or a small set of layers via Gumbel-Softmax.
- Method: Gumbel selection over layers, then classification.
- Result: Macro-F1 0.514.
- Takeaway: Discrete layer selection underperforms learned soft mixing.

### A4: Dimension-wise Gumbel
- Goal: Allow each feature dimension to select its own layer.
- Method: Gumbel over layers per dimension.
- Result: Collapsed; Macro-F1 0.0.
- Takeaway: Over-parameterized and unstable at this scale.

### A5: Hierarchical Conv Layers
- Goal: Combine layer information using hierarchical convolution.
- Method: Conv stack across layers to build fused representation.
- Result: Macro-F1 0.564.
- Takeaway: Best A-series result; structured fusion helps.

### A6: Group Weighted Sum
- Goal: Blend groups of layers instead of all layers independently.
- Method: Grouped weighting of layers, then classifier.
- Result: Macro-F1 0.302.
- Takeaway: Coarse grouping lost fine-grained layer signals.

### A7: Feature Cache + CKA/SVCCA Probing
- Goal: Measure similarity between layer features to guide selection.
- Method: Cache layer features, compute CKA/SVCCA.
- Result: High similarity in nearby layers; divergence across depth.
- Takeaway: Supports using mid-layers and mixing adjacent layers.

### A8: Transition Analysis
- Goal: Identify where representations change most across layers.
- Method: Measure similarity deltas across consecutive layers.
- Result: Largest transition around layer 9 to 12.
- Takeaway: Feature distribution shifts align with A1 optimum.

## Series B: Dimensionality Reduction

### B1: PCA Sweep
- Goal: Find best PCA dimension for HuBERT features.
- Method: Train classifiers across PCA sizes.
- Result: PCA-32 best overall.
- Takeaway: 32 dims is the optimal compression point.

### B2: Kernel PCA (RBF)
- Goal: Nonlinear dimensionality reduction.
- Method: KPCA-32 + classifier.
- Result: Macro-F1 0.394.
- Takeaway: KPCA harms separability for stutter labels.

### B3: MDS
- Goal: Preserve pairwise distances.
- Method: Low-rank MDS-32 + classifier.
- Result: Macro-F1 0.464.
- Takeaway: Better than KPCA, still below PCA.

### B4: Autoencoder Bottleneck
- Goal: Learn nonlinear compression.
- Method: AE bottleneck 32 dims.
- Result: Macro-F1 0.378.
- Takeaway: AE collapses discriminative detail in this setup.

### B5: VAE
- Goal: Regularized latent space for compression.
- Method: VAE mu-32, beta=0.5.
- Result: Macro-F1 0.473 (best non-PCA).
- Takeaway: VAE is the strongest alternative to PCA.

### B6: UMAP
- Goal: Preserve local structure.
- Method: UMAP-32 + classifier.
- Result: Macro-F1 0.466.
- Takeaway: Competitive but not better than PCA.

### B7: Sparse PCA
- Goal: Enforce interpretability via sparsity.
- Method: Sparse PCA reconstruction analysis.
- Result: Reconstruction tradeoffs, no F1 gain.
- Takeaway: Sparsity reduces fidelity for stutter classification.

### B8: ICA
- Goal: Independent component decomposition.
- Method: ICA-32 + classifier.
- Result: Macro-F1 0.276 (worst in B-series).
- Takeaway: Independence objective is misaligned with stutter labels.

## Series C: SSL Fusion

### C1: SSL + MFCC Fusion
- Goal: Add classic acoustic features.
- Method: HuBERT + MFCC fused; sweep MFCC count.
- Result: Best MFCC=52, Macro-F1 0.527.
- Takeaway: MFCC adds complementary signal.

### C2: SSL + Handcrafted Feature Ablation
- Goal: Identify best handcrafted features.
- Method: Fuse HuBERT with features like ZCR, jitter, pause ratio.
- Result: ZCR fusion best, Macro-F1 0.555.
- Takeaway: Simple temporal energy cues help most.

### C3: Multi-SSL Temporal Concatenation
- Goal: Combine multiple SSL models.
- Method: HuBERT + Wav2Vec2 + WavLM concatenation.
- Result: Macro-F1 0.567.
- Takeaway: Multi-SSL improves over single SSL.

### C4: Cross-Attention HuBERT + Whisper
- Goal: Learn cross-modal SSL fusion.
- Method: Bidirectional cross-attention between HuBERT and Whisper.
- Result: Macro-F1 0.659, AUPRC 0.746 (best overall).
- Takeaway: Dual-SSL with attention is the top-performing design.

### C5: Whisper + HuBERT BiLSTM (5 binary heads)
- Goal: Binary per-class modeling for stronger class focus.
- Method: Train 5 independent classifiers.
- Result: Mean F1 0.628.
- Takeaway: Binary heads outperform multi-label baselines.

### C6: HuBERT + Speaker Embedding Fusion
- Goal: Add speaker identity features.
- Method: Fuse SSL with speaker embeddings.
- Result: Mean F1 0.562.
- Takeaway: Speaker features help but less than cross-attn fusion.

## Series D: Deep Temporal Models

### D1: Conformer-4 Multi-label
- Goal: Model temporal structure with a modern sequence model.
- Method: 4-block Conformer, MLSM loss.
- Result: Macro-F1 0.548, AUPRC 0.602 (top AUPRC at the time).
- Takeaway: Strong baseline; good calibration.

### D2: BiLSTM + Attention Pool
- Goal: Sequence modeling with interpretable attention.
- Method: BiLSTM + attention pooling.
- Result: Macro-F1 0.453; attention focused on top PCA dims.
- Takeaway: Helps interpretability but not best accuracy.

### D3: SE-ResNet1D + BiLSTM
- Goal: Blend CNN feature extraction and temporal modeling.
- Method: 1D ResNet + BiLSTM.
- Result: Macro-F1 0.550 with high recall.
- Takeaway: Strong general model for stutter patterns.

### D4: Transformer (no conv)
- Goal: Pure attention on feature tokens.
- Method: Transformer encoder only.
- Result: Macro-F1 0.272 (failed).
- Takeaway: Without conv or strong inductive bias, performance collapses.

### D5: Multitask CNN
- Goal: Auxiliary detection tasks to improve main task.
- Method: Multi-head CNN with multitask loss.
- Result: Macro-F1 0.489, detection AUROC 0.811.
- Takeaway: Multi-task helps detection but not core F1.

### D6: TDNN SSL + MFCC
- Goal: Use TDNN for context aggregation.
- Method: TDNN with SSL+MFCC fusion.
- Result: Macro-F1 0.585 (rank #2 at the time).
- Takeaway: TDNN is a strong balanced model across classes.

### D7: Atrous CNN (dual SSL)
- Goal: Multi-scale temporal modeling.
- Method: Dilated CNN with HuBERT+Whisper.
- Result: Macro-F1 0.649 (rank #2 overall at the time).
- Takeaway: Multi-scale conv competes with cross-attn fusion.

### D8: Prototypical Network
- Goal: Few-shot style metric learning for stutter types.
- Method: ProtoNet with episodic training.
- Result: Macro-F1 0.514; best recall.
- Takeaway: Good recall but not best balance.

## Series E: Graph Models

### E1: GCN on k-NN Graph
- Goal: Exploit inter-clip similarity.
- Method: GCN on k-NN graph, HuBERT PCA-32.
- Result: Macro-F1 0.468, AUPRC 0.602 (best AUPRC at the time).
- Takeaway: Graph averaging improves calibration and precision.

### E2: GAT with Multi-head Attention
- Goal: Learn neighbor weights instead of uniform averaging.
- Method: 4-head GAT on k-NN graph, PCA-64.
- Result: Macro-F1 0.515, AUPRC 0.618 (best AUPRC overall in E-series).
- Takeaway: Attention is critical; strongest graph model.

### E3: HGNN with Hyperedges
- Goal: Capture group-level relations via hyperedges.
- Method: SSL and MFCC k-means hyperedges.
- Result: Macro-F1 0.333 (negative result).
- Takeaway: Large hyperedges over-smooth and blur labels.

### E4: Spatio-Temporal GCN
- Goal: Add within-episode temporal edges.
- Method: k-NN graph plus temporal edges (lambda=0.5).
- Result: Macro-F1 0.462 (slightly worse than E1).
- Takeaway: Sequential clip ordering does not add useful label signal.

### E5: Graph-Transformer (HuBERT + Whisper)
- Goal: Combine graph bias with dual-SSL fusion.
- Method: Batch k-NN + adjacency-biased attention.
- Result: Macro-F1 0.653, AUPRC 0.735 (rank #2 overall).
- Takeaway: Graph attention plus dual-SSL is highly competitive; best SoundRep F1.

## Series F: Robustness and Loss Variants

### F1: Focal Loss + Class-Balanced Sampling
- Goal: Multi-label baseline with recall focus.
- Method: CNN-1D, focal loss, weighted sampler.
- Result: Macro-F1 0.550, high recall 0.726.
- Takeaway: Strong screening baseline but low precision.

### F2: Mixup + DimMask Augmentation
- Goal: Feature-level augmentation for robustness.
- Method: Mixup and dimensional masking on PCA features.
- Result: Macro-F1 0.434 (negative), Prolongation collapses.
- Takeaway: Feature-space mixup harms label fidelity.

### F3: GRL Speaker Disentanglement
- Goal: Remove speaker/show bias.
- Method: Encoder with GRL adversary predicting show ID.
- Result: Macro-F1 0.540, AUPRC 0.627 (best overall at the time).
- Takeaway: Speaker disentanglement gives balanced gains across all classes.

### F4: Conformer + BCE Loss
- Goal: Test deeper model on same inputs.
- Method: Conformer with BCE and weighted sampling.
- Result: Macro-F1 0.545, slightly below F1 baseline.
- Takeaway: Conformer adds little with only 53 feature tokens.

### F5: Acoustic Perturbation
- Goal: Add noise robustness in feature space.
- Method: Gaussian jitter, amplitude scaling, dropout.
- Result: Macro-F1 0.446 (negative), recall drops to 0.362.
- Takeaway: Feature-level perturbation causes train-test mismatch.

## Final Conclusions

1. Dual-SSL fusion is the strongest driver of performance.
2. Best overall model: C4 cross-attn HuBERT + Whisper (Macro-F1 0.659).
3. Best runner-up: E5 graph-transformer with dual SSL (Macro-F1 0.653).
4. Graph attention improves calibration (best AUPRC from E2/E5).
5. PCA-32 is the most reliable compression; KPCA/ICA/AE are inferior.
6. Feature-level augmentation tends to hurt; apply augmentation at raw audio level instead.
7. Speaker disentanglement (F3) is the best robustness intervention.
8. WordRep remains the hardest class; best results come from dual-SSL fusion.

## Production Recommendations

- Lightweight: F3 (GRL), 163K params, Macro-F1 ~0.540.
- Balanced: D6 (TDNN), Macro-F1 0.585, good per-class fairness.
- Maximum: C4 (cross-attn), Macro-F1 0.659, best overall accuracy.
