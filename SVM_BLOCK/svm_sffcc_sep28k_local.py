# ============================================================
# FINAL SFFCC + SVM (Clean vs Block) — SEP-28K
# Thesis-aligned, robust, leakage-free
# Python 3.10+
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
# PARAMETERS (MATCH MFCC & CQCC)
# ============================================================

SR = 8000
PRE_EMPH = 0.97
N_SFFCC = 14              # keep consistent with cepstral features
N_FFT = 1024
HOP_LENGTH = 256
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
# SFFCC FEATURE EXTRACTION (ROBUST)
# ============================================================

def extract_sffcc(path):
    try:
        y, _ = librosa.load(path, sr=SR)
    except Exception:
        return None

    # Safety check
    if y is None or len(y) < 2:
        return None

    # 1. Pre-emphasis (Section 3.2.2)
    y = librosa.effects.preemphasis(y, coef=PRE_EMPH)

    # 2. STFT magnitude spectrum
    S = np.abs(librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )) ** 2

    # 3. Spectral smoothing (moving average)
    kernel = np.ones((5,)) / 5.0
    S_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0,
        arr=S
    )

    # 4. Log compression
    log_S = np.log(S_smooth + 1e-8)

    # 5. Cepstrum
    cep = dct(log_S, type=2, axis=0, norm='ortho')[:N_SFFCC]

    # 6. Global statistical descriptors (Section 3.3)
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
# CLEAN vs BLOCK (ONE-vs-ONE)
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

# Ensure audio exists
clean_df = clean_df[clean_df.apply(
    lambda r: os.path.exists(get_audio_path(r)), axis=1)]
block_df = block_df[block_df.apply(
    lambda r: os.path.exists(get_audio_path(r)), axis=1)]

print(f"Final Clean: {len(clean_df)} | Final Block: {len(block_df)}")

# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n[3] Extracting SFFCC features...")

X_clean = []
for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Clean"):
    feat = extract_sffcc(get_audio_path(r))
    if feat is not None:
        X_clean.append(feat)

X_block = []
for _, r in tqdm(block_df.iterrows(), total=len(block_df), desc="Block"):
    feat = extract_sffcc(get_audio_path(r))
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
    np.zeros(len(X_clean)),   # Clean = 0
    np.ones(len(X_block))     # Block = 1
])

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
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
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ============================================================
# FEATURE NORMALIZATION
# ============================================================

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

# ---- TRAIN F1 ----
y_train_pred = svm.predict(X_train_bal)
train_f1 = f1_score(y_train_bal, y_train_pred)

# ---- TEST F1 ----
y_test_pred = svm.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred)

print("\n📊 F1 SCORES (SFFCC + SVM)")
print(f"Train F1-score : {train_f1:.4f}")
print(f"Test  F1-score : {test_f1:.4f}")

print("\n=== DONE (SFFCC BASELINE FROZEN) ===")
