# Stuttering Event Classification using SVM & Wav2Vec2

## Project Overview
This project implements **two approaches** for detecting stuttering events in speech audio:

1. **Classical ML (SVM)** - Traditional machine learning with handcrafted features
2. **Deep Learning (Wav2Vec2)** - Transfer learning with pre-trained transformer models

Both approaches classify five types of stuttering events:
- Block
- Interjection
- Prolongation
- Sound Repetition
- Word Repetition

---

## 🏆 Overall Results: Wav2Vec2 WINS (4-1)

| Stuttering Type | SVM Best | Wav2Vec2 | Winner | Improvement |
|-----------------|----------|----------|--------|-------------|
| Block | 0.8120 (MFCC) | 0.8127 | Wav2Vec2 ✅ | +0.09% |
| Interjection | 0.7612 (SFFCC) | 0.7865 | Wav2Vec2 ✅ | +3.32% |
| Prolongation | 0.7272 (SFFCC) | 0.7572 | Wav2Vec2 ✅ | +4.12% |
| Sound Repetition | 0.6969 (ZTWCC) | 0.6843 | SVM 🏆 | -1.81% |
| Word Repetition | 0.6456 (SFFCC) | 0.6499 | Wav2Vec2 ✅ | +0.67% |

**Average Improvement**: +1.27% | **Win Rate**: 80%

See [Final_op.txt](Final_op.txt) for SVM results and [wav2vec2_op.txt](wav2vec2_op.txt) for Wav2Vec2 results.

---

## Approach Comparison

### 1. SVM Approach (Classical ML)

**Features**: 4 different audio feature extraction methods
- **CQCC** (Constant-Q Cepstral Coefficients)
- **MFCC** (Mel-Frequency Cepstral Coefficients) - Best for Block
- **SFFCC** (Spectral Flux-based Frequency Cepstral Coefficients) - Best for 3 types
- **ZTWCC** (Zero-Time Windowing Cepstral Coefficients) - Best for Sound Rep

**Configuration**:
- Classifier: SVM with RBF kernel
- Sample Rate: 8 kHz
- Feature Dimensions: 56-78
- Balancing: SMOTE
- Training Time: Fast (~minutes per model)
- Hardware: CPU-friendly ✅

**Total Models**: 20 (5 types × 4 features)

### 2. Wav2Vec2 Approach (Deep Learning)

**Architecture**:
```
Audio (16kHz) → Wav2Vec2-base (Pre-trained) → Mean Pooling → 3-Layer Classifier → Prediction
```

**Model**: `facebook/wav2vec2-base`
- Pre-trained on 960 hours of LibriSpeech
- 768-dimensional contextualized representations
- Automatic feature learning (no manual engineering)

**Configuration**:
- Sample Rate: 16 kHz
- Batch Size: 16
- Learning Rate: 0.0001
- Epochs: 10
- Training Time: ~40-60 min per model (GPU)
- Hardware: GPU recommended (6-8 GB VRAM)

**Total Models**: 5 (5 types × 1 model)

## Project Structure
```
.
├── SVM_BLOCK/                  # SVM Block detection (4 models)
├── SVM_INTERJECTION/           # SVM Interjection detection (4 models)
├── SVM_PROLONGATION/           # SVM Prolongation detection (4 models)
├── SVM_SOUND_REPTITION/        # SVM Sound repetition detection (4 models)
├── SVM_WORD-REPETITION/        # SVM Word repetition detection (4 models)
├── WAV2VEC_BLOCK/              # Wav2Vec2 Block detection
├── WAV2VEC_INTERJECTION/       # Wav2Vec2 Interjection detection
├── WAV2VEC_PROLONGATION/       # Wav2Vec2 Prolongation detection
├── WAV2VEC_SOUND_REPETITION/   # Wav2Vec2 Sound repetition detection
├── WAV2VEC_WORD_REPETITION/    # Wav2Vec2 Word repetition detection
├── Final_op.txt                # SVM complete results (20 models)
├── wav2vec2_op.txt             # Wav2Vec2 complete results (5 models)
├── requirements.txt            # Dependencies for both approaches
└── README.md                   # This file
```

## Dataset
See [requirements.txt](requirements.txt) for complete dependencies.

**For SVM only** (CPU-friendly):
- Python 3.10+
- librosa, numpy, pandas, scikit-learn, imbalanced-learn, scipy, tqdm

**For Wav2Vec2** (GPU recommended):
- All SVM dependencies +
- PyTorch (with CUDA support)
- transformers, torchaudio, soundfile

## Installation

### Option 1: SVM Only (Quick Start)
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install SVM dependencies
pip install librosa numpy pandas scikit-learn imbalanced-learn scipy tqdm
```

### Option 2: Full Setup (SVM + Wav2Vec2)
```bash
# Create conda environment (recommended for deep learning)
conda create -n stuttering python=3.10
conda activate stuttering

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt
```

## Usage

### SVM Models
Each SVM folder contains 4 scripts for different features:
```bash
# Example: Run MFCC-based block detection
cd SVM_BLOCK
python svm_mfcc_sep28k_local.py
```

### SVM Insights
1. **MFCC** achieved highest SVM score (0.8120) for Block detection
2. **SFFCC** performed best for 3 out of 5 stuttering types
3. **ZTWCC** only wins for Sound Repetition (rapid syllable repetitions)
4. Block detection is the easiest (highest scores across both approaches)
5. Word Repetition is the most challenging (lowest scores)

### Wav2Vec2 Insights
1. **Beats SVM on 4 out of 5 types** (80% win rate)
2. **Best improvements**: Prolongation (+4.12%) and Interjection (+3.32%)
3. **Only loss**: Sound Repetition (-1.81%) - ZTWCC's temporal resolution better for rapid events
4. **16 kHz sampling** provides better quality than 8 kHz used in SVM
5. **Automatic feature learning** eliminates manual feature engineering

### Comparison
- **Average improvement**: +1.27% (Wav2Vec2 over best SVM)
- **SVM**: Fast training, CPU-friendly, good baseline
- **Wav2Vec2**: Slower training, GPU-required, better overall performance
- **Hybrid potential**: Combine both for ensemble predictions

## Total Models Trained
- **SVM**: 20 models ✅
- **Wav2Vec2**: 5 models ✅
- **Grand Total**: 25 models 🏆

## Authors
23071A05 (CSE-B) Mini Project

## References
- **Wav2Vec 2.0**: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- **SEP-28k Dataset**: [Lea et al., ICASSP 2021](https://github.com/apple/ml-stuttering-events-dataset)y
```

**Note**: Wav2Vec2 training requires GPU for reasonable time (~40-60 min vs 3-6 hours on CPU

## Model Configuration
- **Classifier**: SVM with RBF kernel
- **Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Preprocessing**: StandardScaler normalization
- **Train-Test Split**: 67%-33%
- **Random State**: 42 (for reproducibility)

## Key Findings
1. **MFCC** achieved the highest F1-score (0.8120) for Block detection
2. **SFFCC** performed best for 3 out of 5 stuttering types
3. Block detection is the easiest (highest scores)
4. Word Repetition is the most challenging (lowest scores)

## Authors
23071A05 (CSE-B) Mini Project

## License
See LICENSE file for details.
