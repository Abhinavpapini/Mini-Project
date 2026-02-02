# ============================================================
# SVM + ZTWCC (PURE Clean vs Interjection) — SEP-28K
# FINAL | THESIS-SAFE | TRAIN vs TEST F1
# Python 3.10.11 | CPU
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
# PARAMETERS (FROZEN – ZTW APPROXIMATION)
# ============================================================

SR = 8000
PRE_EMPH = 0.97

N_ZTWCC = 14          # 0–13
N_FFT = 256           # high time resolution
HOP_LENGTH = 64

TEST_SIZE = 0.33
RANDOM_STATE = 42

# ============================================================
# UTILS
# ============================================================

def elapsed(t0):
    return f"{(time.time() - t0) / 60:.2f} min"

def build_filename(row):
    return f"{row['Show']}_{int(row['EpId'])}_{int(row['ClipId'])}.wav"

def get_audio_path(r):
    """Construct full path to audio file including subdirectory structure"""
    show = r['Show']
    ep_id = int(r['EpId'])
    wav_file = build_filename(r)
    return os.path.join(AUDIO_DIR, show, str(ep_id), wav_file)

# ============================================================
# LOAD LABELS
# ============================================================

print("\n[1] Loading labels...")
df = pd.read_csv(LABELS_PATH)
print("Total rows:", len(df))

# ============================================================
# ALIGN WITH AUDIO FILES
# ============================================================

print("\n[2] Aligning labels with audio...")
# Get shows from clips directory structure  
audio_shows = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
df = df[df["Show"].isin(audio_shows)]
print("Rows after alignment:", len(df))

# ============================================================
# PURE CLEAN vs INTERJECTION
# ============================================================

print("\n[3] Creating PURE Clean vs Interjection split...")

clean_df = df[
    (df["NoStutteredWords"] > 0) &
    (df["Block"] == 0) &
    (df["Prolongation"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["Interjection"] == 0)
]

inter_df = df[df["Interjection"] > 0]

print("Clean candidates:", len(clean_df))
print("Interjection candidates:", len(inter_df))

# ============================================================
# AUDIO VALIDATION
# ============================================================

def audio_exists(row):
    return os.path.exists(get_audio_path(row))

clean_df = clean_df[clean_df.apply(audio_exists, axis=1)]
inter_df = inter_df[inter_df.apply(audio_exists, axis=1)]

print("Final Clean samples:", len(clean_df))
print("Final Interjection samples:", len(inter_df))

# ============================================================
# ZTWCC EXTRACTION (ROBUST + PRE-EMPHASIS)
# ============================================================

print("\n[4] Extracting ZTWCC features...")
t0 = time.time()

def extract_ztwcc(path):
    try:
        y, _ = librosa.load(path, sr=SR)

        if y is None or len(y) < 2:
            return None

        # Pre-emphasis
        y = np.append(y[0], y[1:] - PRE_EMPH * y[:-1])

        S = np.abs(librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window="hann"
        )) ** 2

        log_S = np.log(S + 1e-8)
        cep = dct(log_S, type=2, axis=0, norm="ortho")[:N_ZTWCC]

        return np.hstack([
            cep.mean(axis=1),
            cep.std(axis=1),
            skew(cep, axis=1),
            kurtosis(cep, axis=1)
        ])

    except Exception:
        return None

X_clean, X_inter = [], []

for _, r in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Clean"):
    feat = extract_ztwcc(get_audio_path(r))
    if feat is not None:
        X_clean.append(feat)

for _, r in tqdm(inter_df.iterrows(), total=len(inter_df), desc="Interjection"):
    feat = extract_ztwcc(get_audio_path(r))
    if feat is not None:
        X_inter.append(feat)

X_clean = np.array(X_clean)
X_inter = np.array(X_inter)

print("Usable Clean:", len(X_clean))
print("Usable Interjection:", len(X_inter))
print("Feature dim:", X_clean.shape[1])
print("Time:", elapsed(t0))

# ============================================================
# DATASET BUILD
# ============================================================

X = np.vstack([X_clean, X_inter])
y = np.hstack([
    np.zeros(len(X_clean)),
    np.ones(len(X_inter))
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

smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# SVM TRAINING
# ============================================================

svm = SVC(kernel="rbf", gamma="scale", random_state=RANDOM_STATE)
svm.fit(X_train, y_train)

# ============================================================
# TRAIN vs TEST F1
# ============================================================

train_f1 = f1_score(y_train, svm.predict(X_train))
test_f1 = f1_score(y_test, svm.predict(X_test))

print("\n📊 FINAL RESULTS")
print("ZTWCC + SVM | Clean vs Interjection")
print("Train F1-score:", round(train_f1, 4))
print("Test  F1-score:", round(test_f1, 4))
print("Generalization gap:", round(train_f1 - test_f1, 4))
print("=== DONE ===")
