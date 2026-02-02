# ============================================================
# SVM + MFCC (PURE CLEAN vs SOUND REPETITION) — SEP-28K
# FINAL • FROZEN • THESIS-SAFE
# Python 3.10.11
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.stats import skew, kurtosis

from imblearn.over_sampling import SMOTE
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
# PARAMETERS (FROZEN)
# ============================================================

SR = 8000
N_MFCC = 14          # 0–13
N_FFT = 1024
HOP_LENGTH = 256

TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# UTILS
# ============================================================

def fname(r):
    return f"{r['Show']}_{int(r['EpId'])}_{int(r['ClipId'])}.wav"

def get_audio_path(r):
    """Construct full path with subdirectory structure"""
    show = r['Show']
    ep_id = int(r['EpId'])
    wav_file = fname(r)
    return os.path.join(AUDIO_DIR, show, str(ep_id), wav_file)

def exists(r):
    return os.path.exists(get_audio_path(r))

# ============================================================
# LOAD LABELS
# ============================================================

print("\n[1] Loading labels...")
df = pd.read_csv(LABELS_PATH)
print("Total rows:", len(df))

# ============================================================
# PURE CLEAN vs SOUND REPETITION
# ============================================================

print("\n[2] Creating PURE CLEAN vs SOUND REPETITION split...")

clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Interjection"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Prolongation"] == 0)
]

soundrep_df = df[df["SoundRep"] > 0]

clean_df = clean_df[clean_df.apply(exists, axis=1)]
soundrep_df = soundrep_df[soundrep_df.apply(exists, axis=1)]

print("Final Clean:", len(clean_df))
print("Final SoundRep:", len(soundrep_df))

# ============================================================
# MFCC EXTRACTION (ROBUST)
# ============================================================

print("\n[3] Extracting MFCC features...")

def extract_mfcc(path):
    y, _ = librosa.load(path, sr=SR)

    # Handle empty / corrupted audio
    if y is None or len(y) < N_FFT:
        return np.zeros(4 * N_MFCC)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    feats = np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        skew(mfcc, axis=1),
        kurtosis(mfcc, axis=1)
    ])

    # Remove NaN / Inf (CRITICAL for SMOTE)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats

X_clean = np.array([
    extract_mfcc(get_audio_path(r))
    for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="MFCC clean")
])

X_soundrep = np.array([
    extract_mfcc(get_audio_path(r))
    for _, r in tqdm(soundrep_df.iterrows(), total=len(soundrep_df), desc="MFCC soundrep")
])

# ============================================================
# DATASET
# ============================================================

X = np.vstack([X_clean, X_soundrep])
y = np.hstack([
    np.zeros(len(X_clean)),        # clean = 0
    np.ones(len(X_soundrep))       # soundrep = 1
])

# ============================================================
# TRAIN / TEST SPLIT (FIRST)
# ============================================================

print("\n[4] Train-test split...")
Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train distribution:", np.bincount(ytr.astype(int)))
print("Test distribution :", np.bincount(yte.astype(int)))

# ============================================================
# SMOTE (TRAIN ONLY)
# ============================================================

print("\n[5] Applying SMOTE (TRAIN only)...")
Xtr_bal, ytr_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr, ytr)

print("Balanced train distribution:", np.bincount(ytr_bal.astype(int)))

# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()
Xtr_bal = scaler.fit_transform(Xtr_bal)
Xte = scaler.transform(Xte)

# ============================================================
# SVM + EVALUATION
# ============================================================

print("\n[6] Training RBF-SVM...")

svm = SVC(kernel="rbf", gamma="scale")
svm.fit(Xtr_bal, ytr_bal)

train_f1 = f1_score(ytr_bal, svm.predict(Xtr_bal))
test_f1  = f1_score(yte, svm.predict(Xte))

print("\n===== FINAL RESULTS =====")
print("Train F1:", round(train_f1, 4))
print("Test  F1:", round(test_f1, 4))
print("Gap      :", round(train_f1 - test_f1, 4))

print("\n=== DONE : MFCC + SVM (PURE CLEAN vs SOUND REPETITION) ===")
