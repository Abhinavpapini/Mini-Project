# ============================================================
# SVM + CQCC (PURE CLEAN vs SOUND REPETITION) — SEP-28K
# FINAL • FROZEN • THESIS-SAFE
# Python 3.10.11
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.fftpack import dct

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"c:\23071A05(cse-b) mini project"

AUDIO_DIR   = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "clips")
LABELS_PATH = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "SEP-28k_labels.csv")

# ============================================================
# PARAMETERS (FROZEN)
# ============================================================

SR = 8000
N_CQCC = 14          # 0–13
TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# UTILS
# ============================================================

def elapsed(start):
    return f"{(time.time() - start) / 60:.2f} minutes"

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
t0 = time.time()
df = pd.read_csv(LABELS_PATH)
print("Total rows:", len(df), "| Time:", elapsed(t0))

# ============================================================
# ALIGN WITH AUDIO
# ============================================================

print("\n[2] Aligning labels with audio...")
t0 = time.time()

audio_shows = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
df = df[df["Show"].isin(audio_shows)]

print("Rows after filtering:", len(df))
print("Time:", elapsed(t0))

# ============================================================
# PURE CLEAN vs SOUND REPETITION
# ============================================================

print("\n[3] Creating PURE CLEAN vs SOUND REPETITION split...")
t0 = time.time()

clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Interjection"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Prolongation"] == 0)
]

soundrep_df = df[df["SoundRep"] > 0]

print("Initial Clean:", len(clean_df), "SoundRep:", len(soundrep_df))
print("Time:", elapsed(t0))

# ============================================================
# AUDIO VALIDATION
# ============================================================

print("\n[4] Validating audio files...")
t0 = time.time()

clean_df     = clean_df[clean_df.apply(exists, axis=1)]
soundrep_df = soundrep_df[soundrep_df.apply(exists, axis=1)]

print("Final Clean:", len(clean_df), "Final SoundRep:", len(soundrep_df))
print("Time:", elapsed(t0))

# ============================================================
# CQCC FEATURE EXTRACTION
# ============================================================

print("\n[5] Extracting CQCC features...")
t0 = time.time()

def extract_cqcc(path):
    y, sr = librosa.load(path, sr=SR)

    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=256,
            fmin=32.7,
            bins_per_octave=24,
            n_bins=144
        )
    )

    log_cqt = np.log(cqt + 1e-8)
    cep = dct(log_cqt, type=2, axis=0, norm="ortho")[:N_CQCC]

    stats = np.hstack([
        cep.mean(axis=1),
        cep.std(axis=1),
        np.mean((cep - cep.mean(axis=1, keepdims=True))**3, axis=1),
        np.mean((cep - cep.mean(axis=1, keepdims=True))**4, axis=1)
    ])

    return stats

X_clean = []
for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="CQCC clean"):
    X_clean.append(extract_cqcc(get_audio_path(r)))
X_clean = np.array(X_clean)

X_soundrep = []
for _, r in tqdm(soundrep_df.iterrows(), total=len(soundrep_df), desc="CQCC soundrep"):
    X_soundrep.append(extract_cqcc(get_audio_path(r)))
X_soundrep = np.array(X_soundrep)

print("Feature shapes:", X_clean.shape, X_soundrep.shape)
print("Time:", elapsed(t0))

# ============================================================
# DATASET
# ============================================================

X = np.vstack([X_clean, X_soundrep])
y = np.hstack([
    np.zeros(len(X_clean)),        # clean = 0
    np.ones(len(X_soundrep))       # soundrep = 1
])

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[6] Train-test split...")
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

print("\n[7] Applying SMOTE (TRAIN only)...")
t0 = time.time()

smote = SMOTE(random_state=RANDOM_STATE)
Xtr_bal, ytr_bal = smote.fit_resample(Xtr, ytr)

print("Balanced train distribution:", np.bincount(ytr_bal.astype(int)))
print("Time:", elapsed(t0))

# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()
Xtr_bal = scaler.fit_transform(Xtr_bal)
Xte     = scaler.transform(Xte)

# ============================================================
# SVM + EVALUATION
# ============================================================

print("\n[8] Training RBF-SVM...")
t0 = time.time()

svm = SVC(kernel="rbf", gamma="scale")
svm.fit(Xtr_bal, ytr_bal)

ytr_pred = svm.predict(Xtr_bal)
yte_pred = svm.predict(Xte)

train_f1 = f1_score(ytr_bal, ytr_pred)
test_f1  = f1_score(yte, yte_pred)

print("\n===== FINAL RESULTS =====")
print("Train F1:", round(train_f1, 4))
print("Test  F1:", round(test_f1, 4))
print("Gap      :", round(train_f1 - test_f1, 4))
print("Training time:", elapsed(t0))

print("\n=== DONE : CQCC + SVM (PURE CLEAN vs SOUND REPETITION) ===")
