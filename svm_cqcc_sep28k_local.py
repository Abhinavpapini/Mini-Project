# ============================================================
# FINAL SVM + CQCC (Clean vs Block) — SEP-28K
# Robust, thesis-aligned, examiner-safe
# Python 3.10+
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.fftpack import dct
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
# PARAMETERS (FROM THESIS)
# ============================================================

SR = 8000
N_CQCC = 14          # 0th–13th coefficient
TEST_SIZE = 0.33
RANDOM_STATE = 42
PRE_EMPH = 0.97

# ============================================================
# UTILS
# ============================================================

def wav_name(r):
    return f"{r['Show']}_{int(r['EpId'])}_{int(r['ClipId'])}.wav"

def wav_path(r):
    return os.path.join(AUDIO_DIR, r['Show'], str(int(r['EpId'])), wav_name(r))

def elapsed(t0):
    return f"{(time.time() - t0)/60:.2f} min"

# ============================================================
# FEATURE EXTRACTION (ROBUST)
# ============================================================

def extract_cqcc(path):
    try:
        y, _ = librosa.load(path, sr=SR)
    except Exception:
        return None

    # ---- SAFETY CHECK ----
    if y is None or len(y) < 2:
        return None
    # ---------------------

    # 1. Pre-emphasis
    y = np.append(y[0], y[1:] - PRE_EMPH * y[:-1])

    # 2. Constant-Q Transform
    cqt = np.abs(librosa.cqt(
        y,
        sr=SR,
        hop_length=256,
        fmin=32.7,
        bins_per_octave=24,
        n_bins=144
    ))

    # 3. Log compression
    log_cqt = np.log(cqt + 1e-8)

    # 4. Cepstrum
    cep = dct(log_cqt, type=2, axis=0, norm="ortho")[:N_CQCC]

    # 5. Global statistical descriptors
    feat = np.hstack([
        cep.mean(axis=1),
        cep.std(axis=1),
        skew(cep, axis=1),
        kurtosis(cep, axis=1)
    ])

    return feat

# ============================================================
# LOAD & FILTER LABELS
# ============================================================

print("\n[1] Loading labels...")
df = pd.read_csv(LABELS_PATH)

# ============================================================
# CLEAN vs BLOCK (ONE-vs-ONE)
# ============================================================

print("[2] Creating Clean vs Block sets...")

clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Prolongation"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Interjection"] == 0)
]

block_df = df[df["Block"] > 0]

# Ensure audio exists
clean_df = clean_df[clean_df.apply(
    lambda r: os.path.exists(wav_path(r)), axis=1)]
block_df = block_df[block_df.apply(
    lambda r: os.path.exists(wav_path(r)), axis=1)]

print("Final Clean:", len(clean_df), "Final Block:", len(block_df))

# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n[3] Extracting CQCC features...")
t0 = time.time()

X_clean = []
for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Clean"):
    feat = extract_cqcc(wav_path(r))
    if feat is not None:
        X_clean.append(feat)

X_block = []
for _, r in tqdm(block_df.iterrows(), total=len(block_df), desc="Block"):
    feat = extract_cqcc(wav_path(r))
    if feat is not None:
        X_block.append(feat)

X_clean = np.array(X_clean)
X_block = np.array(X_block)

print("Feature shapes:", X_clean.shape, X_block.shape)
print("Extraction time:", elapsed(t0))

# ============================================================
# LABELS
# ============================================================

X = np.vstack([X_clean, X_block])
y = np.hstack([
    np.zeros(len(X_clean)),   # Clean = 0
    np.ones(len(X_block))     # Block = 1
])

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Train-test split...")
Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# ============================================================
# SMOTE (TRAIN ONLY)
# ============================================================

print("[5] Applying SMOTE on training data...")
smote = SMOTE(random_state=RANDOM_STATE)
Xtr_bal, ytr_bal = smote.fit_resample(Xtr, ytr)

# ============================================================
# FEATURE NORMALIZATION
# ============================================================

print("[6] Feature normalization...")
scaler = StandardScaler()
Xtr_bal = scaler.fit_transform(Xtr_bal)
Xte = scaler.transform(Xte)

# ============================================================
# SVM TRAINING & EVALUATION
# ============================================================

print("[7] Training SVM...")
svm = SVC(kernel="rbf", gamma="scale")
svm.fit(Xtr_bal, ytr_bal)

y_pred = svm.predict(Xte)
f1 = f1_score(yte, y_pred)

print("\n✅ FINAL CQCC + SVM F1-score:", round(f1, 4))
print("\n=== DONE (FINAL THESIS-SAFE BASELINE) ===")
