# ============================================================
# FINAL ZTWCC + SVM (Clean vs Block) — SEP-28K
# Thesis-aligned, robust, leakage-free
# ============================================================

import os
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
# PARAMETERS (MATCH OTHER BASELINES)
# ============================================================

SR = 8000
PRE_EMPH = 0.97
N_ZTWCC = 14
N_FFT = 256
HOP_LENGTH = 64
TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# UTIL
# ============================================================

def wav_name(r):
    return f"{r['Show']}_{int(r['EpId'])}_{int(r['ClipId'])}.wav"
def get_audio_path(r):
    """Construct full path to audio file including subdirectory structure"""
    show = r['Show']
    ep_id = int(r['EpId'])
    wav_file = wav_name(r)
    return os.path.join(AUDIO_DIR, show, str(ep_id), wav_file)
# ============================================================
# ZTWCC FEATURE EXTRACTION (ROBUST)
# ============================================================

def extract_ztwcc(path):
    try:
        y, _ = librosa.load(path, sr=SR)
    except Exception:
        return None

    # Safety check
    if y is None or len(y) < 2:
        return None

    # 1. Pre-emphasis
    y = librosa.effects.preemphasis(y, coef=PRE_EMPH)

    # 2. High time-resolution STFT
    S = np.abs(librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann"
    )) ** 2

    # 3. Log compression
    log_S = np.log(S + 1e-8)

    # 4. Cepstrum
    cep = dct(log_S, type=2, axis=0, norm="ortho")[:N_ZTWCC]

    # 5. Global statistics
    feat = np.hstack([
        cep.mean(axis=1),
        cep.std(axis=1),
        skew(cep, axis=1),
        kurtosis(cep, axis=1)
    ])

    return feat

# ============================================================
# LOAD LABELS
# ============================================================

print("\n[1] Loading labels...")
df = pd.read_csv(LABELS_PATH)

# Filter to shows that have clip directories
audio_shows = {d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))}
df = df[df["Show"].isin(audio_shows)]

# ============================================================
# CLEAN vs BLOCK
# ============================================================

print("[2] Creating Clean vs Block dataset...")

clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Prolongation"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Interjection"] == 0)
]

block_df = df[df["Block"] > 0]

clean_df = clean_df[clean_df.apply(
    lambda r: os.path.exists(get_audio_path(r)), axis=1)]
block_df = block_df[block_df.apply(
    lambda r: os.path.exists(get_audio_path(r)), axis=1)]

print(f"Final Clean: {len(clean_df)} | Final Block: {len(block_df)}")

# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n[3] Extracting ZTWCC features...")

X_clean, X_block = [], []

for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Clean"):
    feat = extract_ztwcc(get_audio_path(r))
    if feat is not None:
        X_clean.append(feat)

for _, r in tqdm(block_df.iterrows(), total=len(block_df), desc="Block"):
    feat = extract_ztwcc(get_audio_path(r))
    if feat is not None:
        X_block.append(feat)

X_clean = np.array(X_clean)
X_block = np.array(X_block)

print("Feature shapes:", X_clean.shape, X_block.shape)

# ============================================================
# LABELS
# ============================================================

X = np.vstack([X_clean, X_block])
y = np.hstack([
    np.zeros(len(X_clean)),
    np.ones(len(X_block))
])

# ============================================================
# SPLIT → SMOTE → SCALE
# ============================================================

print("\n[4] Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print("[5] Applying SMOTE on training data...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("[6] Normalizing features...")
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test)

# ============================================================
# SVM TRAINING & EVALUATION
# ============================================================

print("[7] Training SVM...")
svm = SVC(kernel="rbf", gamma="scale", C=1.0)
svm.fit(X_train_bal, y_train_bal)

train_f1 = f1_score(y_train_bal, svm.predict(X_train_bal))
test_f1 = f1_score(y_test, svm.predict(X_test))

print("\n📊 F1 SCORES (ZTWCC + SVM)")
print(f"Train F1-score : {train_f1:.4f}")
print(f"Test  F1-score : {test_f1:.4f}")

print("\n=== DONE (ZTWCC BASELINE FROZEN) ===")
