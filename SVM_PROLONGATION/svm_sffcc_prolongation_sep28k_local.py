# ============================================================
# SVM + SFFCC (PURE CLEAN vs NON-PURE Prolongation)
# SEP-28K | 8 kHz | CPU | THEORY-CORRECT
# ============================================================

import os
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
PRE_EMPH = 0.97

N_SFFCC = 14          # 0–13
N_FFT = 1024
HOP_LENGTH = 256

SMOOTH_WIN = 5        # moving average smoothing

TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# LOAD LABELS
# ============================================================

df = pd.read_csv(LABELS_PATH)

# PURE CLEAN
clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Prolongation"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Interjection"] == 0)
]

# NON-PURE POSITIVE
prolong_df = df[df["Prolongation"] > 0]

def build_filename(row):
    return f"{row['Show']}_{int(row['EpId'])}_{int(row['ClipId'])}.wav"

def get_audio_path(row):
    """Construct full path with subdirectory structure"""
    show = row['Show']
    ep_id = int(row['EpId'])
    wav_file = build_filename(row)
    return os.path.join(AUDIO_DIR, show, str(ep_id), wav_file)

clean_df = clean_df[clean_df.apply(
    lambda r: os.path.exists(get_audio_path(r)), axis=1
)]
prolong_df = prolong_df[prolong_df.apply(
    lambda r: os.path.exists(get_audio_path(r)), axis=1
)]

print("PURE Clean:", len(clean_df))
print("Prolongation (NON-PURE):", len(prolong_df))

# ============================================================
# SFFCC EXTRACTION (FROZEN DEFINITION)
# ============================================================

def extract_sffcc(path):
    y, sr = librosa.load(path, sr=SR)

    if len(y) < N_FFT:
        return np.zeros(4 * N_SFFCC)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - PRE_EMPH * y[:-1])

    # STFT power spectrum
    S = np.abs(librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )) ** 2

    # Frequency-domain smoothing (moving average)
    kernel = np.ones((SMOOTH_WIN, 1)) / SMOOTH_WIN
    S_smooth = np.apply_along_axis(
        lambda x: np.convolve(x, kernel[:, 0], mode="same"),
        axis=0,
        arr=S
    )

    log_S = np.log(S_smooth + 1e-8)

    # DCT → SFFCC
    sffcc = librosa.feature.mfcc(
        S=log_S,
        sr=sr,
        n_mfcc=N_SFFCC
    )

    return np.hstack([
        sffcc.mean(axis=1),
        sffcc.std(axis=1),
        skew(sffcc, axis=1),
        kurtosis(sffcc, axis=1)
    ])

# ============================================================
# FEATURE EXTRACTION
# ============================================================

X_clean = np.array([
    extract_sffcc(get_audio_path(r))
    for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="SFFCC clean")
])

X_prolong = np.array([
    extract_sffcc(get_audio_path(r))
    for _, r in tqdm(prolong_df.iterrows(), total=len(prolong_df), desc="SFFCC prolong")
])

X = np.vstack([X_clean, X_prolong])
y = np.hstack([
    np.zeros(len(X_clean)),
    np.ones(len(X_prolong))
])

# ============================================================
# TRAIN / TEST SPLIT (BEFORE SMOTE)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# ============================================================
# SMOTE (TRAIN ONLY)
# ============================================================

X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

# ============================================================
# SCALING + SVM
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train, y_train)

# ============================================================
# EVALUATION (TRAIN vs TEST)
# ============================================================

y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)

f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)

print("\nSFFCC + SVM (PURE Clean vs Prolongation)")
print(f"Train F1-score : {f1_train:.4f}")
print(f"Test  F1-score : {f1_test:.4f}")
