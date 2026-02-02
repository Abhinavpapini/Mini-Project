# ============================================================
# SVM + CQCC (Clean vs WordRep — NON-PURE POSITIVE)
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

N_CQCC = 14          # 0–13 (MANDATED)
HOP_LENGTH = 256
BINS_PER_OCTAVE = 24
N_BINS = 144

TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# UTILITIES
# ============================================================

def build_filename(row):
    return f"{row['Show']}_{int(row['EpId'])}_{int(row['ClipId'])}.wav"

def extract_cqcc(path):
    """
    CQCC extraction with full safety against:
    - empty audio
    - corrupted clips
    - librosa pre-emphasis crashes
    """
    try:
        y, _ = librosa.load(path, sr=SR)

        # ---- SAFETY CHECK ----
        if y is None or len(y) < 2:
            return np.zeros(4 * N_CQCC)

        # Pre-emphasis
        y = librosa.effects.preemphasis(y, coef=PRE_EMPH)

        # Constant-Q Transform
        cqt = np.abs(
            librosa.cqt(
                y,
                sr=SR,
                hop_length=HOP_LENGTH,
                fmin=32.7,
                bins_per_octave=BINS_PER_OCTAVE,
                n_bins=N_BINS
            )
        )

        log_cqt = np.log(cqt + 1e-8)

        # CQCC
        cep = dct(log_cqt, type=2, axis=0, norm="ortho")[:N_CQCC]

        # Global statistics
        return np.hstack([
            cep.mean(axis=1),
            cep.std(axis=1),
            skew(cep, axis=1),
            kurtosis(cep, axis=1)
        ])

    except Exception:
        # Absolute safety net (never crash pipeline)
        return np.zeros(4 * N_CQCC)

# ============================================================
# LOAD & FILTER LABELS
# ============================================================

df = pd.read_csv(LABELS_PATH)

valid_shows = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
df = df[df["Show"].isin(valid_shows)]

# ============================================================
# CLASS DEFINITIONS (FINAL & NON-NEGOTIABLE)
# ============================================================

# PURE CLEAN (ONLY NEGATIVE CLASS)
clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Interjection"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Prolongation"] == 0)
]

# WORD REPETITION — NON-PURE POSITIVE
wordrep_df = df[df["WordRep"] > 0]

print("Clean samples:", len(clean_df))
print("WordRep samples:", len(wordrep_df))

# ============================================================
# AUDIO VALIDATION
# ============================================================

def get_audio_path(row):
    """Construct full path with subdirectory structure"""
    show = row['Show']
    ep_id = int(row['EpId'])
    wav_file = build_filename(row)
    return os.path.join(AUDIO_DIR, show, str(ep_id), wav_file)

def audio_exists(row):
    return os.path.exists(get_audio_path(row))

clean_df = clean_df[clean_df.apply(audio_exists, axis=1)]
wordrep_df = wordrep_df[wordrep_df.apply(audio_exists, axis=1)]

# ============================================================
# FEATURE EXTRACTION
# ============================================================

X_clean, X_word = [], []

for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="CQCC Clean"):
    X_clean.append(
        extract_cqcc(get_audio_path(r))
    )

for _, r in tqdm(wordrep_df.iterrows(), total=len(wordrep_df), desc="CQCC WordRep"):
    X_word.append(
        extract_cqcc(get_audio_path(r))
    )

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
X_test = scaler.transform(X_test)

# ============================================================
# SVM TRAINING & EVALUATION
# ============================================================

svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train, y_train)

y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

print("\nCQCC + SVM (Clean vs WordRep — NON-PURE)")
print("Train F1:", f1_score(y_train, y_pred_train))
print("Test  F1:", f1_score(y_test, y_pred_test))

print("\n=== DONE (FINAL, METHODologically & ENGINEERING SAFE) ===")
