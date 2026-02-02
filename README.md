# Stuttering Event Classification using SVM

## Project Overview
This project implements Support Vector Machine (SVM) classifiers for detecting different types of stuttering events in speech audio. The system classifies five types of stuttering events:
- Block
- Interjection
- Prolongation
- Sound Repetition
- Word Repetition

## Features
Each stuttering type is classified using 4 different audio feature extraction methods:
- **CQCC** (Constant-Q Cepstral Coefficients)
- **MFCC** (Mel-Frequency Cepstral Coefficients)
- **SFFCC** (Spectral Flux-based Frequency Cepstral Coefficients)
- **ZTWCC** (Zero-Time Windowing Cepstral Coefficients)

## Results Summary
Total Models Trained: **20/20** ✅

| Stuttering Type | Best Method | Test F1-Score |
|-----------------|-------------|---------------|
| Block | MFCC | 0.8120 ⭐ |
| Interjection | SFFCC | 0.7612 |
| Prolongation | SFFCC | 0.7272 |
| Sound Repetition | ZTWCC | 0.6969 |
| Word Repetition | SFFCC | 0.6456 |

⭐ Highest overall performance

See [Final_op.txt](Final_op.txt) for detailed results of all 20 models.

## Project Structure
```
.
├── SVM_BLOCK/                  # Block detection models
├── SVM_INTERJECTION/           # Interjection detection models
├── SVM_PROLONGATION/           # Prolongation detection models
├── SVM_SOUND_REPTITION/        # Sound repetition detection models
├── SVM_WORD-REPETITION/        # Word repetition detection models
├── Final_op.txt                # Complete results and analysis
└── .gitignore                  # Git ignore configuration
```

## Dataset
- **Source**: SEP-28k (Stuttering Events in Podcasts - 28k dataset)
- **Clean Samples**: 4,090 (consistent across all experiments)
- **Stuttering Samples**: Varies by event type (3,795 - 8,874)

**Note**: Dataset is not included in this repository due to size constraints. Download the SEP-28k dataset separately.

## Requirements
- Python 3.10+
- librosa
- numpy
- pandas
- scikit-learn
- imbalanced-learn (SMOTE)
- scipy
- tqdm

## Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install librosa numpy pandas scikit-learn imbalanced-learn scipy tqdm
```

## Usage
Each folder contains 4 Python scripts for different feature extraction methods:
```bash
# Example: Run MFCC-based block detection
python SVM_BLOCK/svm_mfcc_sep28k_local.py
```

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
