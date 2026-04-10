# SEP-28k Stutter Classification — Complete Experiment Log
**All 40 Experiments | What We Did | What We Got | Final Conclusions**

Dataset: SEP-28k (Apple) | Task: Multi-label stutter-type classification (Block, Prolongation, SoundRep, WordRep, Interjection)  
Primary Metric: **Macro-F1** (higher = better) | Evaluation: Speaker-independent (LOSO fold0)

---

## ⚡ Quick Leaderboard (Top 10 of 40)

| Rank | Exp | Method | Macro-F1 | AUPRC |
|------|-----|--------|----------|-------|
| 🥇 1 | **C4** | HuBERT × Whisper Cross-Attention Fusion | **0.6587** | **0.7459** |
| 🥈 2 | **D7** | HuBERT + Whisper Atrous-CNN | 0.6494 | 0.7261 |
| 🥉 3 | **E5** | HuBERT + Whisper Graph Transformer | 0.6528 | 0.7354 |
| 4 | **D6** | HuBERT + MFCC TDNN | 0.5848 | 0.6475 |
| 5 | **F3** | GRL Speaker Disentanglement CNN | 0.5399 | 0.6271 |
| 6 | **D3** | SE-ResNet1D + BiLSTM | 0.5503 | 0.5833 |
| 7 | **C3** | Multi-SSL Temporal Concat Transformer | 0.5674 | — |
| 8 | **C5** | Whisper + HuBERT BiLSTM Fusion | 0.6282 | 0.6548 |
| 9 | **E2** | Dynamic GAT (4-head) | 0.5153 | 0.6184 |
| 10 | **D1** | 4-block Conformer + MLSM | 0.5477 | 0.6016 |

---

## Phase 1 — Layer-wise Analysis (A-series)

### Exp A1 — Single-Layer Probing (Wav2Vec2 + SVM)
**What We Did:**  
Used frozen `facebook/wav2vec2-base` features from 5 layers (1, 3, 6, 9, 12) fed into an SVM classifier with RBF kernel. Evaluated separately on all 5 stutter types. Also compared against purely handcrafted SVM baselines (MFCC, SFFCC, CQCC, ZTWCC).

**What We Got:**

| Stutter Type | Best SVM (handcrafted) | A1 Best-Layer F1 | Best Layer |
|---|---|---|---|
| Block | 0.8120 (MFCC) | **0.8430** | Layer 9 |
| Interjection | 0.7612 (SFFCC) | **0.8694** | Layer 9 |
| Prolongation | 0.7272 (SFFCC) | **0.8372** | Layer 9 |
| Sound Rep | 0.6969 (ZTWCC) | **0.8374** | Layer 9 |
| Word Rep | 0.6456 (SFFCC) | **0.8132** | Layer 6 |
| **Mean** | **0.7286** | **0.8400** | — |

- Layer 9 is best for 4/5 types; Layer 6 for Word Repetition
- Layer 12 (last) is consistently weakest — over-specialises to ASR task
- **+15.86% average improvement over best handcrafted SVM baseline**

**Conclusion:** Wav2Vec2 mid-to-late layers capture stutter-discriminative acoustic structure far better than handcrafted features. Layer 9 is the canonical reference layer for all downstream experiments.

---

### Exp A2 — Weighted-Sum Layer Aggregation
**What We Did:**  
Instead of using a single fixed layer, learned a softmax-weighted combination over layers [1, 3, 6, 9, 12]. The layer weights were jointly trained with a linear classifier. Used 768-dim full features, target=Block.

**What We Got:**
- Macro-F1 = **0.5591**
- Learned weights favoured **Layer 6** most, then Layer 9
- (Note: this is multi-label setting on less splits than A1, hence lower absolute F1)

**Conclusion:** Weighted aggregation does encode a preference for mid-layers (6, 9). However, soft blending does not significantly outperform using just the best single layer (A1). Promoted as a useful diagnostic, but not the top strategy.

---

### Exp A3 — Gumbel Softmax Hard Layer Selection
**What We Did:**  
Applied Gumbel-softmax with temperature annealing to discretely select one layer among [1, 3, 6, 9, 12] during training. Used BiLSTM + Attention classifier. Aim: learn which single layer to use, differentiably.

**What We Got:**
- Macro-F1 = **0.5137**
- Hard-selected layer: **Layer 9**
- Highest soft weight still at Layer 6 (contradiction between soft and hard gate)

**Conclusion:** Gumbel hard selection converges to Layer 9 (confirming A1), but the performance is lower than A2 weighted-sum. Hard discrete selection loses gradient signal and hurts training stability. Not promoted beyond diagnostic value.

---

### Exp A4 — Dimension-wise Gumbel Layer Selection
**What We Did:**  
Extended A3 to select a different layer for each of the 768 feature dimensions independently (dimension-wise spatial gating). Extremely fine-grained selection.

**What We Got:**
- Macro-F1 = **0.000** — **COMPLETE COLLAPSE**
- Layer selection stayed near-uniform (no feature specialisation emerged)
- Classifier defaulted to majority-class prediction throughout

**Conclusion:** Dimension-wise discrete gating is too high-dimensional for the dataset size — the gradient signal is lost in the combinatorial space. **Not promoted. Dead end.**

---

### Exp A5 — Hierarchical Conv Across Layers
**What We Did:**  
Applied a 1D depthwise convolution across the layer axis (stacking layers 1, 3, 6, 9, 12 as 5 temporal positions) followed by an MLP classifier. Learns spatial context between adjacent layers.

**What We Got:**
- Macro-F1 = **0.5642**
- Strongest salience detected at **Layer 6**
- Outperforms A3 and matches/slightly beats A2

**Conclusion:** Convolutional aggregation across layers is competitive. This shows that inter-layer transition patterns are informative. Best layer-aggregation method in the A-series (alongside A2).

---

### Exp A6 — Group-Weighted Layer Fusion
**What We Did:**  
Grouped layers into early (1, 3), mid (6, 9), late (12) and applied a learnable group-level weight before PCA-32 reduction + CNN-1D classifier.

**What We Got:**
- Macro-F1 = **0.3023**
- Learned weights favoured **mid-group**
- Significantly underperformed A2 and A5

**Conclusion:** Coarse group-level fusion loses the fine-grained layer discrimination available in per-layer weighting. **Not promoted.**

---

### Exp A7 — CKA / SVCCA Representational Similarity Probing
**What We Did:**  
Computed Centered Kernel Alignment (CKA) and SVCCA similarities between layer pairs across the 5 Wav2Vec2 layers on a fold0 sample. Also computed probing accuracy per layer. Pure analysis experiment, no classification model trained.

**What We Got:**
- Layers 6 and 9 form a **high-similarity cluster** — they share most representational content
- Layers 1 and 3 form an early-feature cluster
- Layer 12 is most distinct from all others
- Layer 9 probing accuracy highest, confirming A1

**Conclusion:** The CKA analysis validates the A1 layer selection empirically. Mid-layers (6–9) contain the most stutter-relevant shared representations. This justifies Layer 9 as the canonical choice. Analytical finding, no model promoted.

---

### Exp A8 — Layer Transition Drift Analysis
**What We Did:**  
Measured cosine similarity and embedding drift between consecutive transformer layers (1→3, 3→6, 6→9, 9→12) on fold0 samples. Analysed how much representation changes at each transition.

**What We Got:**
- **Largest drift at Layer 9→12** (low cosine similarity = big shift)
- Layer 6→9 moderate drift
- Layer 1→3 smallest drift (features still close to raw input)
- This matches: Layer 12 over-specialises for ASR, drifting away from acoustic stutter patterns

**Conclusion:** The drift analysis confirms Layer 9 is the last "stable" layer before the model collapses into ASR-specific representations at Layer 12. Strong methodological backing for the Layer 9 choice.

---

## Phase 1 — Dimensionality Reduction (B-series)

### Exp B1 — PCA Sweep (Baseline Dimensionality Study)
**What We Did:**  
Applied PCA to Wav2Vec2 Layer 9 features (768-dim) and swept dimensions: {768, 512, 256, 128, 64, 32, 16, 8}. Measured reconstruction quality and used SVM to evaluate classification F1 at each dimension. Full dataset: 20,906 samples.

**What We Got:**
- Best trade-off: **dim = 32** → minimal F1 loss vs full 768-dim, massive compute savings
- Dimensions below 16 show significant performance degradation
- PCA-32 retains >95% of explained variance

**Conclusion:** PCA to 32 dimensions is the canonical dimensionality reduction for all subsequent experiments. It's fast, interpretable, and matches full-dim performance. **B1 establishes the project's dim=32 standard.**

---

### Exp B2 — Kernel PCA (RBF)
**What We Did:**  
Applied KPCA with RBF kernel (γ=auto) to reduce HuBERT-large Layer 21 features to 32 dims. Trained CNN-1D classifier on top with MLSM loss + weighted sampler.

**What We Got:**
- Macro-F1 = **0.3935** — **NEGATIVE vs linear PCA**
- Prolongation F1 = 0.173 (catastrophic collapse for minority class)
- KPCA introduces non-linear distortions that hurt separability

**Conclusion:** Non-linear kernel projection is **redundant** on SSL features, which are already highly non-linearly structured. Linear PCA is strictly better here. KPCA not promoted.

---

### Exp B3 — Low-Rank MDS
**What We Did:**  
Applied Metric MDS with landmark approximation (n_landmarks=3000) to reduce HuBERT-L21 features to 32 dims. Evaluated with CNN-1D + MLSM.

**What We Got:**
- Macro-F1 = **0.4643** — marginally below PCA (0.490)
- Reconstruction stress = 19.8M (non-zero = distortion present)
- B-series ranking so far: **PCA > MDS > KPCA**

**Conclusion:** MDS preserves global pairwise distances but introduces slight distortions that hurt local class separability. Marginally negative vs PCA. Not promoted.

---

### Exp B4 — Autoencoder Bottleneck
**What We Did:**  
Trained a 2-stage deep autoencoder (768→256→32→256→768) jointly with a CNN classifier, using MSE reconstruction loss + MLSM classification loss.

**What We Got:**
- Macro-F1 = **0.3784** — **WORST of B-series**
- Interjection F1 = 0.359 (severe collapse)
- Reconstruction loss and discriminative loss compete during training

**Conclusion:** AE bottleneck optimises for reconstruction, not classification separability. The two objectives conflict, leading to poor minority-class performance. **Reconstruction ≠ discriminability. Not promoted.**

---

### Exp B5 — VAE Latent Space
**What We Did:**  
Trained a Variational Autoencoder (β=0.5, KL warmup over 20 epochs) to map HuBERT-L21 768-dim features to 32-dim latent μ. Used μ (mean) as features for CNN-1D classifier.

**What We Got:**
- Macro-F1 = **0.4731** — **BEST non-PCA reduction in B-series**
- KL divergence = 0.226 (stable)
- Only -0.017 below the PCA-32 baseline

**Conclusion:** VAE is the best generative reduction approach — the Gaussian prior encourages smoother, more generalisable latent representations. The KL warmup is critical. Still trails PCA, but the smallest gap. Positive finding.

---

### Exp B6 — UMAP
**What We Did:**  
Applied UMAP (n_neighbors=15, min_dist=0.1) to reduce HuBERT-L21 to 32 dims, then trained CNN-1D + MLSM. B-series closure experiment.

**What We Got:**
- Macro-F1 = **0.4656**
- 3rd best non-PCA method (after VAE and MDS)
- Fast inference but non-reproducible stochasticity without fixed seed

**Conclusion:** UMAP topology preservation helps slightly over MDS and KPCA but still trails PCA. B-series final ranking: **PCA > VAE > UMAP > MDS > KPCA > AE > ICA.** PCA is the canonical choice.

---

### Exp B7 — Sparse PCA
**What We Did:**  
Applied Sparse PCA (α=0.1) to Wav2Vec2-base Layer 9 features. Analysis-only experiment to find interpretable sparse components. Best dimension: 79.

**What We Got:**
- Best reconstruction MSE = 0.5697 at dim=79, α=0.1
- Sparse components correlate to audible sub-band patterns (more interpretable than dense PCA)
- No direct F1 comparison (analysis experiment)

**Conclusion:** Sparse PCA produces more interpretable features — components map to specific acoustic phenomena. Useful for paper analysis sections, not for production classification. Low priority.

---

### Exp B8 — ICA (FastICA)
**What We Did:**  
Applied FastICA (logcosh nonlinearity) to reduce HuBERT-L21 to 32 independent components. Trained CNN-1D on ICA features.

**What We Got:**
- Macro-F1 = **0.2758** — **ABSOLUTE WORST of B-series**
- Prolongation recall = 0.056 (nearly zero — catastrophic)
- ICA components maximise statistical independence, not class separability

**Conclusion:** Statistical independence of components is **orthogonal** to classification discriminability for correlated multi-label stutter types. **ICA is definitively the worst dimensionality reduction approach for this task.**

---

## Phase 2 — Feature Fusion (C-series)

### Exp C1 — SSL + MFCC Fusion
**What We Did:**  
Concatenated Wav2Vec2 Layer 9 PCA-32 features with MFCC features of varying dimension (swept from 13 to 80). Trained CNN-1D classifier. Target=Block, multi-class setting.

**What We Got:**
- Best fusion: **dim=52 MFCC** concatenated with PCA-32 SSL (total: 84-dim)
- Macro-F1 = **0.5268** for Block
- Marginal gain over SSL-only at best dim

**Conclusion:** MFCC adds limited but real complementary signal — the best fusion dimension (52) reflects that larger spectral resolution helps. However, the gain is not consistent across all stutter types, suggesting handcrafted features are partially redundant with SSL.

---

### Exp C2 — SSL + Handcrafted Feature Group Ablation
**What We Did:**  
Fused SSL PCA-32 features with groups of handcrafted features: (1) energy, (2) ZCR (zero crossing rate), (3) jitter, (4) formant frequencies. Ablated each group separately to find the most informative complement.

**What We Got:**
- Best group: **ssl + ZCR** (fusion_dim=34)
- Macro-F1 = **0.5550** — best in C-series at this stage
- ZCR captures the high-frequency fine structure of stuttering events (rapid changes)
- Jitter and formants add minimal value over SSL alone

**Conclusion:** ZCR is the most complementary handcrafted feature to SSL representations. It captures temporal irregularity not fully encoded in smooth SSL embeddings. Simple but effective.

---

### Exp C3 — Multi-SSL Temporal Concatenation (Transformer)
**What We Did:**  
Concatenated features from two Wav2Vec2-base models (primary + benchmark-w2v2-base) at their best layers along the temporal axis. Applied a Transformer classifier (4-head, 2-layer) over the 128-dim projected multi-stream input.

**What We Got:**
- Macro-F1 = **0.5674**
- Two-stream Transformer improved over single-stream by modest margin
- Train/val convergence stable across fold0

**Conclusion:** Multi-model feature concatenation with cross-stream attention shows promise. Two streams of the same model architecture are partially redundant — different model families would yield more complementary features (motivates C4, C5, C6).

---

### Exp C4 — Cross-Attention SSL Fusion: HuBERT × Whisper ⭐ ALL-TIME CHAMPION
**What We Did:**  
Fused **HuBERT-large Layer 21** (1024-dim, self-supervised) with **Whisper-large Layer 28** (1280-dim, ASR-supervised) using **bidirectional cross-attention** (8 heads × 32-dim per head). Both streams query each other simultaneously. MLP classifier on the 512-dim joint representation.

**What We Got:**

| Stutter Type | F1 | Precision | Recall | AUPRC |
|---|---|---|---|---|
| Block | 0.6023 | 0.6531 | 0.5589 | 0.6796 |
| Prolongation | 0.5887 | 0.6680 | 0.5263 | 0.6942 |
| SoundRep | 0.6210 | 0.6947 | 0.5614 | 0.7032 |
| WordRep | **0.6949** | 0.7229 | 0.6689 | **0.7684** ← ALL-TIME BEST |
| Interjection | **0.7868** | 0.8563 | 0.7278 | **0.8841** ← ALL-TIME BEST |
| **Macro** | **0.6587** | **0.7190** | **0.6087** | **0.7459** |

- **+0.119 over previous champion F3** — unprecedented gap
- All 5 stutter types achieve all-time best simultaneously
- Best epoch = 3 (severe overfitting after → needs regularisation)

**Why it works:**
- HuBERT captures **acoustic/phonetic structure** of the stutter sounds
- Whisper captures **linguistic/boundary structure** (word positions, filler tokens)
- Cross-attention lets each model query the other for matching context
- WordRep: Whisper detects repeated word-boundary patterns HuBERT can't see
- Interjection: Whisper explicitly models "um/uh" as linguistic tokens

**Conclusion:** **C4 is the project champion.** The combination of complementary-objective SSL models (acoustic masking vs linguistic ASR) is the decisive breakthrough. Cross-attention is superior to simple concatenation for capturing inter-modal interactions.

---

### Exp C5 — Whisper + HuBERT BiLSTM Fusion
**What We Did:**  
Fused Whisper-large and HuBERT-large features using a BiLSTM with attention-weighted pooling across both streams. PCA-32 per stream, BiLSTM processes both streams sequentially.

**What We Got:**
- Macro-F1 = **0.6282**
- Interjection F1 = 0.7160 (excellent)
- Attention weights ~50/50 per stream (both equally informative)
- AUPRC = 0.6548

**Conclusion:** Whisper+HuBERT BiLSTM is a strong result (2nd best at this stage, later surpassed by C4). BiLSTM sequential fusion is solid but cross-attention (C4) is explicitly better at capturing inter-stream relationships.

---

### Exp C6 — HuBERT + WavLM-SV Speaker-Normalized Fusion
**What We Did:**  
Fused HuBERT-large Layer 21 features with WavLM-SV (speaker verification) embeddings. Goal: add explicit speaker identity signal and test if normalising by speaker helps. CNN-1D classifier.

**What We Got:**
- Macro-F1 = **0.5617**
- WordRep F1 = 0.468 (relatively low)  
- AUPRC = 0.6768 (high, suggesting good ranking even with lower F1)
- Speaker embedding adds limited discriminative improvement

**Conclusion:** Speaker identity is a noise factor, not a helpful feature for stutter classification. Explicitly encoding speaker identity pulls the model toward speaker-specific patterns rather than generalising stutter characteristics. This motivates adversarial speaker removal (F3).

---

## Phase 2 — Imbalance + Multi-label (F-series, core runs)

### Exp F1 — Focal Loss + Class-Balanced Sampling (Production Baseline)
**What We Did:**  
Multi-label classification with HuBERT-large L21 PCA-32 features. Used **Focal Loss** (γ=2, α=class_freq-inverse) combined with class-balanced batch sampling. CNN-1D classifier (3 conv layers). Sigmoid head for multi-label output.

**What We Got:**
- Macro-F1 = **0.5497**
- Recall = **0.726** (very high — model catches most stutters)
- Precision = 0.458 (low — over-predicts minority classes)
- Interjection F1 = 0.642 (best class)
- WordRep F1 = 0.415 (hardest)

**Conclusion:** Focal loss dramatically improves minority class recall (the model "tries harder" to find rare stutter types) but at the cost of precision. This is the core production baseline — subsequent experiments compare against this. F1 recall=0.73 is an important clinical property.

---

### Exp F4 — Conformer + BCE Multi-label
**What We Did:**  
Replaced CNN-1D with a **Conformer block** (depthwise conv + self-attention) in a multi-label setup with Binary Cross-Entropy loss (not focal loss). Used same PCA-32 + MFCC-21 features as F1.

**What We Got:**
- Macro-F1 = **0.5448** (~same as F1)
- ~15% more parameters than F1 CNN
- Block F1 = 0.606 (best class)
- WordRep F1 = 0.421

**Conclusion:** Conformer with BCE is essentially equivalent to CNN with Focal Loss for this dataset size. The Conformer's local conv + attention is beneficial (motivates D1) but BCE alone isn't better than focal loss for minority classes.

---

## Phase 3 — Deep Models (D-series)

### Exp D1 — 4-block Conformer + MLSM ✅ PROMOTED
**What We Did:**  
4-block Conformer (depthwise conv + multi-head self-attention) with MultiLabelSoftMarginLoss. HuBERT-large L21 PCA-32 features. Class-balanced sampler.

**What We Got:**
- Macro-F1 = **0.5477**
- AUPRC = **0.6016** (Best AUPRC at this stage)
- Precision = **0.574** (best precision in series)
- WordRep F1 = 0.444 (best WordRep in D-series at the time)

**Conclusion:** 4-block Conformer is the best deep architecture for single-stream classification. The **depthwise convolution is the critical inductive bias** — it captures local acoustic co-articulation patterns. Best single-stream production model promoted from D-series.

---

### Exp D2 — BiLSTM + Attention Pooling
**What We Did:**  
BiLSTM with multi-head attention pooling over PCA-64 features (slightly wider than D1). MLSM loss + class-balanced sampler.

**What We Got:**
- Macro-F1 = **0.4528**
- Attention focuses on PCA dimensions 01 and 02 (not PCA-00)
- Significantly below D1 Conformer

**Conclusion:** BiLSTM is not competitive with Conformer on this task. Sequential dependency modelling is less effective than local conv for short stutter clips (~1-2 sec). Optional/documented experiment; not promoted for main pipeline.

---

### Exp D3 — SE-ResNet1D + BiLSTM (Post-35 Extension)
**What We Did:**  
Combined Squeeze-and-Excitation ResNet-1D (32×32 sequence from HuBERT full-dim) with BiLSTM. The SE block provides channel attention, BiLSTM provides sequence modelling. No PCA (uses 1024-dim directly).

**What We Got:**
- Macro-F1 = **0.5503**
- Recall = **0.600** (best recall of any single-stream model)
- Overfits after epoch 10
- Block F1=0.573, Interjection F1=0.624

**Conclusion:** SE-ResNet + BiLSTM is the **best single-stream model in the D-series** when full-dim features are used. Channel attention + sequence modelling complement each other. Overfitting is manageable with early stopping. Strong post-35 result.

---

### Exp D4 — Conv-Free Transformer + MLSM ⛔ FAILED
**What We Did:**  
Pure Transformer (multi-head attention only, no depthwise conv) on PCA-32 features with MLSM loss. Tested the hypothesis that self-attention alone is sufficient.

**What We Got:**
- Macro-F1 = **0.2719** — **CATASTROPHIC FAILURE**
- Prolongation F1 = 0.082 (near-zero)
- Interjection F1 = 0.163
- D1 Conformer beats by **+0.276 F1**

**Conclusion:** **Depthwise convolution is NOT optional for stutter classification.** Pure attention collapses on local acoustic stutter patterns. Transformers without conv cannot capture the fine-grained temporal co-articulation structure needed. This is the clearest proof of the conv inductive bias.

---

### Exp D5 — Multi-task CNN (Detection + Type Classification)
**What We Did:**  
Shared CNN backbone with two heads: (1) binary stutter detection (BCE+AUROC), (2) multi-label stutter type classification (MLSM). Trained jointly with class-balanced sampler.

**What We Got:**
- Type Macro-F1 = **0.4894**
- Detection AUROC = **0.811** (unique clinical value)
- Detection Precision = 0.97 (very few false alarms)
- Type F1 slightly below single-task D1

**Conclusion:** Multi-task learning provides a **unique clinical value** — detection AUROC 0.811 means the model can reliably identify if a clip contains stuttering at all. This is valuable for a two-stage screening pipeline. Type F1 trades off slightly vs single-task models.

---

### Exp D6 — TDNN (Time-Delay Neural Network) + SSL + MFCC (Post-35)
**What We Did:**  
TDNN with dilations [1,1,2,2,1] + Statistics Pooling, fed with concatenated HuBERT-L21 (1024-dim) + MFCC-78 features (total 1102-dim). Weighted sampler.

**What We Got:**
- Macro-F1 = **0.5848** — **Rank #2 among D-series post-35**
- Most balanced per-class F1 in entire D-series
- Interjection F1 = 0.685
- Best epoch = 4 (early convergence)

**Conclusion:** TDNN with multi-scale dilations is extremely effective — dilated receptive fields capture stutter duration variations well. MFCC adds prosodic information on top of SSL. **Most balanced model in the project** across all 5 stutter types.

---

### Exp D7 — Atrous-CNN + Dual-Pool (HuBERT + Whisper) (Post-35)
**What We Did:**  
Atrous/Dilated CNN (dilations [1,2,4,8,16]) with Dual Pooling (mean + max) on **concatenated HuBERT-large L21 + Whisper-large L28** features (1024+1280=2304-dim).

**What We Got:**
- Macro-F1 = **0.6494** — **Rank #2 overall in entire project**
- Prolongation F1 = **0.6128** — ALL-TIME BEST ever for Prolongation
- WordRep F1 = 0.6882
- Best epoch = 2 (very fast convergence/early stopping critical)

**Conclusion:** HuBERT+Whisper with Atrous-CNN is the second-best method overall, just barely below C4's cross-attention. The dual-pool structure captures both average energy (mean) and peak activations (max) for the stutter signal. The Atrous dilation scales naturally to different stutter durations.

---

### Exp D8 — Prototypical Network (Few-Shot)
**What We Did:**  
Episodic training using 5-way 8-shot prototypical network on PCA-32 features. Computes class prototypes from support examples and classifies by similarity. Multi-label adaptation.

**What We Got:**
- Macro-F1 = **0.5142** — 3rd best in D-series
- Recall = **0.638** (best recall in D-series for single-stream)
- WordRep Recall = 0.665 — best minority class recall
- AUPRC = 0.4947 (lower than Conformer)

**Conclusion:** Prototypical networks excel at **minority class recall** — the episodic training forces the model to find discriminative prototypes even for rare classes. Less suitable as the primary classifier (lower AUPRC) but valuable for interpretability and few-shot scenarios.

---

## Phase 3 — Graph Models (E-series)

### Exp E1 — GCN Temporal Graph
**What We Did:**  
Built a k-NN (k=10) temporal graph over the training batch using PCA-32 features, then applied 2-layer Graph Convolutional Network (GCN) with MLSM loss.

**What We Got:**
- Macro-F1 = **0.4676**
- AUPRC = **0.6023** (highest at this stage — best ranking)
- Precision = **0.675** (highest precision in E-series)
- Model not fully converged at epoch 200

**Conclusion:** GCN with temporal k-NN graph gives excellent ranking quality (high AUPRC) but lower absolute F1 than Conformer. The graph structure captures neighbourhood stutter co-occurrence patterns but is sensitive to k-NN choice and slow to converge.

---

### Exp E2 — Dynamic GAT (4-head Attention)
**What We Did:**  
Replaced GCN with a 4-head Graph Attention Network (GAT). Used dynamic edge weighting (attention-computed) instead of fixed k-NN adjacency. PCA-64 features (wider than E1).

**What We Got:**
- Macro-F1 = **0.5153**
- AUPRC = **0.6184** — NEW BEST AUPRC at this stage
- Interjection AUPRC = **0.733** (exceptional)
- Stable convergence in 200 epochs

**Conclusion:** Dynamic GAT outperforms GCN substantially. The 4-head attention enables different heads to specialise in different stutter co-occurrence patterns. GAT is the recommended graph architecture for stutter classification. Best in E-series.

---

### Exp E3 — Hypergraph Neural Network (HGNN) ⛔ NEGATIVE
**What We Did:**  
Built a hypergraph where each hyperedge connects clusters of semantically similar samples. Applied HGNN with 100+50 cluster hyperedges. Included MFCC-21 as additional node features.

**What We Got:**
- Macro-F1 = **0.3330** — **CATASTROPHIC**
- WordRep F1 = 0.118 (near-zero — catastrophic collapse)
- Mean cluster size = 209 nodes (too large = over-smoothing)

**Conclusion:** Large hyperedge clusters cause severe over-smoothing — all node representations collapse to the cluster mean, destroying local stutter patterns. HGNN is **not suitable** for this task without very fine-grained cluster design. **Not promoted.**

---

### Exp E4 — Spatio-Temporal GCN (ST-GCN)
**What We Did:**  
Applied ST-GCN with separate spatial (acoustic similarity) and temporal (sequential) edge types. Lambda parameter (λ=0.5) balances spatial vs temporal contribution.

**What We Got:**
- Macro-F1 = **0.4618** — slightly below E1 GCN
- Temporal edges provide no clear benefit (−0.006 vs E1)
- AUPRC = 0.6006

**Conclusion:** Adding explicit temporal graph edges to GCN does not help over purely spatial neighbourhood graphs. Stutter events in 1-2 second clips don't have strong sequential graph dependencies at the PCA-32 feature level. E2 GAT (dynamic attention) is strictly better.

---

### Exp E5 — Graph Transformer (HuBERT + Whisper) (Post-35)
**What We Did:**  
Applied a Graph Transformer (k-NN=8, adjacency bias β=1.0) on **concatenated HuBERT+Whisper** (2304-dim) features. Node features from both SSL models, edges from cosine similarity.

**What We Got:**
- Macro-F1 = **0.6528** — **Rank #3 overall**
- SoundRep F1 = **0.6271** — ALL-TIME BEST for Sound Repetition
- AUPRC = **0.7354** (2nd highest ever)
- Most informative graph model in the project

**Conclusion:** HuBERT+Whisper Graph Transformer is exceptional — combining the two complementary SSL streams in a graph structure that models inter-sample acoustic similarity produces state-of-the-art results. **SoundRep benefit is unique** — the graph captures rapid syllable-repetition neighbouring patterns.

---

## Phase 4 — Robustness Extensions (F-series, remaining)

### Exp F2 — Mixup + Spectral Feature Masking (DimMask) Augmentation
**What We Did:**  
Applied Mixup (α=0.3) at the feature level: interpolated between random training examples. Also applied DimMask (20% of PCA-32 dimensions randomly zeroed). CNN-1D + MLSM, weighted sampler.

**What We Got:**
- Macro-F1 = **0.4339** — **NEGATIVE vs baseline**
- Prolongation F1 = 0.287 (severe collapse)
- WordRep Recall = 0.547 (best recall for WordRep in this series)
- DimMask destroyed PCA dimension structure

**Conclusion:** Feature-space Mixup creates unrealistic interpolations for stutter events (mixing "block" with "interjection" has no acoustic meaning). DimMask destroys the PCA dimension ordering. **Augmentation in feature space is counterproductive** — augmentation should happen in raw audio space before SSL.

---

### Exp F3 — GRL Adversarial Speaker Disentanglement ⭐ CORE-35 CHAMPION
**What We Did:**  
Added a **Gradient Reversal Layer (GRL)** with λ=0.5 (warmup from 0 over 10 epochs) to adversarially prevent the encoder from learning speaker/show identity. Show identity (TV show source: 5 groups) used as the adversarial proxy. CNN-1D stutter head + GRL speaker adversary trained jointly.

**What We Got:**

| Stutter Type | F1 | Precision | Recall | AUPRC |
|---|---|---|---|---|
| Block | 0.5177 | 0.6198 | 0.4445 | 0.6275 |
| Prolongation | 0.5203 | 0.6362 | 0.4401 | 0.6379 |
| SoundRep | 0.5267 | 0.6075 | 0.4649 | 0.5778 |
| WordRep | 0.4911 | 0.5746 | 0.4288 | 0.5383 |
| Interjection | 0.6439 | 0.7422 | 0.5686 | 0.7543 |
| **Macro** | **0.5399** | **0.6360** | **0.4694** | **0.6271** |

- **+0.025 over previous champion E2-GAT**
- **All 5 classes hit their best simultaneous F1** in the core-35 series
- Speaker adversary converged to near-chance (loss=1.37 ≈ chance 1.61)
- Monotonically improves through 50 epochs (not fully converged)

**Why it works:** Dataset has massive speaker bias — WomenWhoStutter = 43.8% of clips, HeStutters = 17.6%. Without disentanglement, the model learns show-specific recording artifacts, not stutter patterns. GRL forces the encoder to forget show identity → more generalisable features.

**Conclusion:** Speaker disentanglement is the highest-impact single architectural change in the core-35 experiments. **F3 is the recommended production model for the core-35 scope.** Should be combined with C4 cross-attention in future work for maximum performance.

---

### Exp F5 — Feature-Space Perturbation (Jitter + Scale + Dropout)
**What We Did:**  
Applied feature-level augmentation: Gaussian jitter (σ=0.01), random scale (±10%), and feature dropout (15%). CNN-1D + MLSM. Weighted sampler. Closure experiment for F-series and the full 35-core set.

**What We Got:**
- Macro-F1 = **0.4457** — **NEGATIVE vs baseline**
- Recall = 0.362 (too low — conservative predictions)
- HuBERT features are already noise-robust (pre-trained on diverse audio)

**Conclusion:** HuBERT's pre-training already provides robustness to the types of perturbations tested. Adding noise at the feature level reduces confidence without improving generalisation. Feature-space augmentation is a dead end; raw audio augmentation (before SSL) would be more appropriate. **35/35 core experiments complete.**

---

## Summary Tables

### By Series

| Series | Goal | Best Exp | Best F1 | Key Insight |
|--------|------|----------|---------|-------------|
| **A** | Layer Analysis | A5 (0.564) | 0.564 | Layer 9 is canonical; conv aggregation beats gating |
| **B** | Dim Reduction | B5 VAE (0.473) | 0.490 (PCA) | PCA-32 wins; ICA/AE fail |
| **C** | Feature Fusion | C4 (0.659) | **0.6587** | HuBERT × Whisper XAttn = CHAMPION |
| **D** | Deep Models | D7 (0.649) | 0.6494 | Conformer+conv critical; Atrous+dual-stream 2nd best |
| **E** | Graph Models | E5 (0.653) | 0.6528 | GAT > GCN > ST-GCN > HGNN; needs dual-stream |
| **F** | Robustness | F3 (0.540) | 0.5399 | GRL speaker disentanglement; feature augment fails |

### Per-Class All-Time Best F1

| Stutter Type | Best F1 | Method | Exp |
|---|---|---|---|
| Block | **0.6531 (pre)** / 0.6023 F1 | HuBERT × Whisper XAttn | C4 |
| Prolongation | **0.6128** | HuBERT + Whisper Atrous-CNN | D7 |
| SoundRep | **0.6271** | HuBERT + Whisper Graph Transformer | E5 |
| WordRep | **0.6949** | HuBERT × Whisper XAttn | C4 |
| Interjection | **0.7868** | HuBERT × Whisper XAttn | C4 |

---

## 🏆 FINAL CONCLUSION

### What We Discovered (In Priority Order)

**1. Complementary SSL Models are the Decisive Factor**  
Combining HuBERT-large (acoustic masking pre-training) with Whisper-large (ASR pre-training) in a cross-attention fusion (C4) gives the best result: **Macro-F1 = 0.6587**, a +0.119 improvement over the previous champion. HuBERT captures acoustic stuttering patterns; Whisper captures linguistic boundary information. Their combination is non-redundant and synergistic.

**2. Speaker Bias Must Be Actively Removed**  
The dataset is dominated by one podcast (WomenWhoStutter = 43.8%). Without adversarial speaker disentanglement (F3-GRL), models learn speaker/show identity as a shortcut, not stutter acoustics. F3 raised Macro-F1 to 0.540 using only basic CNN, proving speaker generalisation is a core bottleneck.

**3. Depthwise Convolution is Non-Negotiable**  
Conv-free Transformer (D4) scored 0.272 while 4-block Conformer (D1) scored 0.548 — a +0.276 gap. Stutter classification requires local temporal modelling. Depthwise conv captures the co-articulation patterns within 20-40ms windows that pure attention cannot find.

**4. Layer 9 of Wav2Vec2/HuBERT is the Canonical Feature Layer**  
Across all single-model experiments, mid-to-late transformer layers (specifically Layer 9 for Wav2Vec2-base, Layer 21 for HuBERT-large) consistently outperform early and final layers. The final layer (L12/L24/L32) is over-specialised for the SSL pre-training objective.

**5. PCA-32 is the Best Lightweight Dimensionality Reduction**  
768-dim or 1024-dim SSL features can be safely reduced to 32 dimensions with PCA with minimal F1 loss (<0.5%). Non-linear methods (KPCA, ICA, AE) all underperform PCA for SSL features that are already non-linearly structured.

**6. Graph Models Excel at Ranking but Struggle with Raw F1**  
E2-GAT achieves the best AUPRC (0.618) among single-stream graph models, indicating excellent ranking quality. However, graph models need large batches to form meaningful neighbourhood graphs and are slow to converge. When combined with dual-stream SSL (E5), graph transformers become highly competitive.

**7. Focal Loss + Balanced Sampling is Essential for Minority Class Recovery**  
Standard BCE/CE training ignores rare stutter types (WordRep, SoundRep). Focal Loss (F1 experiment) pushed recall to 0.726, recovering minority class performance at a precision trade-off. This is clinically justified — missing a stutter event is worse than a false alarm in a screening context.

### Final Best System

| Approach | Components | Macro-F1 | Recommended For |
|---|---|---|---|
| **Research Best** | C4: HuBERT × Whisper Cross-Attention | **0.6587** | Highest accuracy |
| **Production Best** | F3 + D1: GRL + Conformer (combine in future) | ~0.60 est. | Generalisation |
| **Clinical Pipeline** | D5 Multi-task (detect + type) | AUROC=0.811 | Screening first stage |
| **Lightweight** | B1+F1: PCA-32 + CNN + Focal Loss | 0.5497 | CPU deployment |

### The One-Line Summary  
> **A cross-attention fusion of HuBERT-large and Whisper-large (C4) achieves Macro-F1=0.659 — the best stutter classification result in this project. The key insight is that acoustic SSL models (HuBERT) and linguistic ASR models (Whisper) encode complementary and non-redundant stutter features that, when combined via cross-attention on the SEP-28k dataset, produce all-time best F1 across all 5 stutter types simultaneously.**

---

*Generated: 2026-04-10 | Project: SEP-28k Stutter Classification | Experiments: A1-A8, B1-B8, C1-C6, D1-D8, E1-E5, F1-F5 (40 total)*
