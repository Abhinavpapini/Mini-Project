# ============================================================
# SVM + SFFCC (PURE CLEAN vs WORD REPETITION — NON-PURE)
# SEP-28K | Python 3.10.11 | CPU SAFE | FINAL (LOCKED)
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
# PARAMETERS (FROZEN)
# ============================================================

SR = 8000
PRE_EMPH = 0.97

N_SFFCC = 14        # 0–13 ONLY
N_FFT = 1024
HOP_LENGTH = 256

TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# UTILITIES
# ============================================================

def build_filename(row):
    return f"{row['Show']}_{int(row['EpId'])}_{int(row['ClipId'])}.wav"

def get_audio_path(row):
    """Construct full path with subdirectory structure"""
    show = row['Show']
    ep_id = int(row['EpId'])
    wav_file = build_filename(row)
    return os.path.join(AUDIO_DIR, show, str(ep_id), wav_file)

def audio_exists(row):
    return os.path.exists(get_audio_path(row))

# ============================================================
# SFFCC EXTRACTION (ROBUST & SAFE)
# ============================================================

def extract_sffcc(path):
    try:
        y, _ = librosa.load(path, sr=SR)

        if y is None or len(y) < N_FFT:
            return np.zeros(4 * N_SFFCC)

        # Pre-emphasis
        y = librosa.effects.preemphasis(y, coef=PRE_EMPH)

        # STFT power spectrum
        S = np.abs(
            librosa.stft(
                y,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                window="hann"
            )
        ) ** 2

        # Frequency smoothing
        kernel = np.ones(5) / 5.0
        S_smooth = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"),
            axis=0,
            arr=S
        )

        log_S = np.log(S_smooth + 1e-8)

        # SFFCC
        cep = dct(log_S, type=2, axis=0, norm="ortho")[:N_SFFCC]

        feats = np.hstack([
            cep.mean(axis=1),
            cep.std(axis=1),
            skew(cep, axis=1),
            kurtosis(cep, axis=1)
        ])

        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception:
        return np.zeros(4 * N_SFFCC)

# ============================================================
# LOAD & ALIGN LABELS WITH AUDIO (CRITICAL FIX)
# ============================================================

df = pd.read_csv(LABELS_PATH)

valid_shows = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
df = df[df["Show"].isin(valid_shows)]

# ============================================================
# CLASS DEFINITIONS (FINAL)
# ============================================================

# PURE CLEAN
clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Interjection"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Prolongation"] == 0)
]

# WORD REPETITION — NON-PURE
wordrep_df = df[df["WordRep"] > 0]

# Validate audio existence
clean_df = clean_df[clean_df.apply(audio_exists, axis=1)]
wordrep_df = wordrep_df[wordrep_df.apply(audio_exists, axis=1)]

print("Final Clean samples   :", len(clean_df))
print("Final WordRep samples :", len(wordrep_df))

# ============================================================
# FEATURE EXTRACTION
# ============================================================

X_clean, X_word = [], []

for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="SFFCC Clean"):
    X_clean.append(extract_sffcc(get_audio_path(r)))

for _, r in tqdm(wordrep_df.iterrows(), total=len(wordrep_df), desc="SFFCC WordRep"):
    X_word.append(extract_sffcc(get_audio_path(r)))

X = np.vstack([X_clean, X_word])
y = np.hstack([
    np.zeros(len(X_clean)),
    np.ones(len(X_word))
])

# ============================================================
# TRAIN / TEST SPLIT (NO LEAKAGE)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# ============================================================
# SMOTE (TRAIN ONLY)
# ============================================================

X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(
    X_train, y_train
)

# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ============================================================
# SVM TRAINING & EVALUATION
# ============================================================

svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train, y_train)

train_f1 = f1_score(y_train, svm.predict(X_train))
test_f1  = f1_score(y_test,  svm.predict(X_test))

print("\nSFFCC + SVM (Clean vs WordRep — NON-PURE)")
print("Train F1:", round(train_f1, 4))
print("Test  F1:", round(test_f1, 4))
print("Gap      :", round(train_f1 - test_f1, 4))

print("\n=== DONE (FINAL SFFCC WORDREP PIPELINE) ===")
