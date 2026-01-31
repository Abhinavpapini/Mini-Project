# ============================================================
# MFCC + SVM (Block vs PURE CLEAN)
# CONTROLLED VARIANT (LABEL PURITY TEST)
# SEP-28K | Python 3.10+
# ============================================================

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"c:\23071A05(cse-b) mini project"
AUDIO_DIR = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "clips")
LABELS_PATH = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "SEP-28k_labels.csv")

# ============================================================
# PARAMETERS (UNCHANGED)
# ============================================================

SR = 8000
PRE_EMPH = 0.97

WIN_LENGTH = 200     # 25 ms
HOP_LENGTH = 80      # 10 ms
N_FFT = 256

TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# UTIL
# ============================================================

def wav_name(r):
    return f"{r['Show']}_{int(r['EpId'])}_{int(r['ClipId'])}.wav"

def wav_path(r):
    return os.path.join(AUDIO_DIR, r['Show'], str(int(r['EpId'])), wav_name(r))

# ============================================================
# MFCC FEATURE EXTRACTION
# ============================================================

def extract_mfcc(path):
    try:
        y, _ = librosa.load(path, sr=SR)
    except Exception:
        return None

    if y is None or len(y) < 2:
        return None

    y = librosa.effects.preemphasis(y, coef=PRE_EMPH)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=14,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window="hann"
    )

    mfcc = mfcc[1:, :]  # drop C0 → 13 MFCCs

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.hstack([
        mfcc.mean(axis=1),  mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1)
    ])

    return feat

# ============================================================
# LOAD LABELS
# ============================================================

print("\n[1] Loading labels...")
df = pd.read_csv(LABELS_PATH)
print("Total samples:", len(df))

# ============================================================
# PURE CLEAN vs BLOCK
# ============================================================

print("[2] Creating Block vs PURE CLEAN dataset...")

block_df = df[df["Block"] > 0]

pure_clean_df = df[
    (df["Block"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Prolongation"] == 0) &
    (df["Interjection"] == 0)
]

# Ensure audio exists
block_df = block_df[block_df.apply(
    lambda r: os.path.exists(wav_path(r)), axis=1)]

pure_clean_df = pure_clean_df[pure_clean_df.apply(
    lambda r: os.path.exists(wav_path(r)), axis=1)]

print(f"Final PURE CLEAN : {len(pure_clean_df)}")
print(f"Final BLOCK      : {len(block_df)}")

# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n[3] Extracting MFCC features...")

X_clean = []
for _, r in tqdm(pure_clean_df.iterrows(), total=len(pure_clean_df), desc="Pure Clean"):
    feat = extract_mfcc(wav_path(r))
    if feat is not None:
        X_clean.append(feat)

X_block = []
for _, r in tqdm(block_df.iterrows(), total=len(block_df), desc="Block"):
    feat = extract_mfcc(wav_path(r))
    if feat is not None:
        X_block.append(feat)

X_clean = np.array(X_clean, dtype=np.float32)
X_block = np.array(X_block, dtype=np.float32)

print("Feature shapes:", X_clean.shape, X_block.shape)

# ============================================================
# LABELS
# ============================================================

X = np.vstack([X_clean, X_block])
y = np.hstack([
    np.zeros(len(X_clean)),  # PURE CLEAN
    np.ones(len(X_block))    # BLOCK
])

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train distribution:", np.bincount(y_train.astype(int)))
print("Test  distribution:", np.bincount(y_test.astype(int)))

# ============================================================
# NORMALIZATION
# ============================================================

print("[5] Normalizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# SVM TRAINING & EVALUATION
# ============================================================

print("[6] Training SVM...")
svm = SVC(kernel="rbf", gamma="scale", C=1.0)
svm.fit(X_train, y_train)

train_f1 = f1_score(y_train, svm.predict(X_train))
test_f1 = f1_score(y_test, svm.predict(X_test))

print("\n📊 F1 SCORES (MFCC + Δ + ΔΔ | PURE CLEAN)")
print(f"Train F1-score : {train_f1:.4f}")
print(f"Test  F1-score : {test_f1:.4f}")

print("\n=== DONE (PURE CLEAN VARIANT) ===")
