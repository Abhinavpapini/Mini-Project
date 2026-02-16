# ============================================================
# Wav2Vec2 Features + SVM (Clean vs Block) — SEP-28K
# CORRECTED METHODOLOGY: Wav2Vec2 as Feature Extractor
# Layer-wise Analysis (Layers 1, 3, 6, 9, 12)
# Python 3.10+ | GPU Recommended for faster extraction
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"c:\23071A05(cse-b) mini project"
AUDIO_DIR = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "clips")
LABELS_PATH = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "SEP-28k_labels.csv")

# ============================================================
# PARAMETERS
# ============================================================

TARGET_SR = 16000  # Wav2Vec2 requires 16kHz
TEST_SIZE = 0.33
RANDOM_STATE = 42

# Layers to analyze
LAYERS_TO_TEST = [1, 3, 6, 9, 12]  # Different depths of Wav2Vec2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# UTILS
# ============================================================

def wav_name(r):
    return f"{r['Show']}_{int(r['EpId'])}_{int(r['ClipId'])}.wav"

def get_audio_path(r):
    """Construct full path to audio file"""
    show = r['Show']
    ep_id = int(r['EpId'])
    wav_file = wav_name(r)
    return os.path.join(AUDIO_DIR, show, str(ep_id), wav_file)

# ============================================================
# WAV2VEC2 FEATURE EXTRACTION (FROZEN MODEL)
# ============================================================

def extract_wav2vec2_features(audio_path, processor, model, layer_idx):
    """
    Extract features from a specific Wav2Vec2 layer (FROZEN model).
    
    Args:
        audio_path: Path to audio file
        processor: Wav2Vec2Processor
        model: Wav2Vec2Model (frozen)
        layer_idx: Which layer to extract features from (1-12)
    
    Returns:
        Feature vector (768-dim after mean pooling)
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=TARGET_SR)
    except Exception:
        return None
    
    # Process audio
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    
    # Extract features (no gradient computation)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
    
    # Get hidden states from specific layer
    # outputs.hidden_states is a tuple of (layers + 1) tensors
    # Index 0 is the embedding layer, 1-12 are the transformer layers
    hidden_state = outputs.hidden_states[layer_idx]  # (1, time, 768)
    
    # Mean pooling over time dimension
    features = hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)
    
    return features

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("WAV2VEC2 FEATURES + SVM (BLOCK CLASSIFICATION)")
    print("Layer-wise Analysis: Testing layers", LAYERS_TO_TEST)
    print("="*70)
    
    # Load labels
    print("\n[1] Loading labels...")
    df = pd.read_csv(LABELS_PATH)
    
    # Filter to shows with audio
    audio_shows = {d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))}
    df = df[df["Show"].isin(audio_shows)]
    
    # Create dataset (Clean vs Block)
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
    
    # Verify audio files exist
    clean_df = clean_df[clean_df.apply(lambda r: os.path.exists(get_audio_path(r)), axis=1)]
    block_df = block_df[block_df.apply(lambda r: os.path.exists(get_audio_path(r)), axis=1)]
    
    print(f"Clean samples: {len(clean_df)}")
    print(f"Block samples: {len(block_df)}")
    
    # Prepare file paths and labels
    file_paths = (
        [get_audio_path(r) for _, r in clean_df.iterrows()] +
        [get_audio_path(r) for _, r in block_df.iterrows()]
    )
    
    labels = np.array([0] * len(clean_df) + [1] * len(block_df))
    
    # Train-test split (do once, use same split for all layers)
    print("\n[3] Train-test split...")
    X_paths_train, X_paths_test, y_train, y_test = train_test_split(
        file_paths, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    
    print(f"Train samples: {len(X_paths_train)}")
    print(f"Test samples: {len(X_paths_test)}")
    
    # Load Wav2Vec2 model and processor
    print("\n[4] Loading Wav2Vec2 model (frozen)...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
    model.eval()  # Set to evaluation mode (frozen)
    
    # Freeze all parameters (no training)
    for param in model.parameters():
        param.requires_grad = False
    
    print("✅ Wav2Vec2 loaded and frozen (feature extraction mode)")
    
    # ============================================================
    # LAYER-WISE ANALYSIS
    # ============================================================
    
    results = []
    
    for layer_idx in LAYERS_TO_TEST:
        print("\n" + "="*70)
        print(f"TESTING LAYER {layer_idx}")
        print("="*70)
        
        # Extract features from this layer
        print(f"\n[5.{layer_idx}] Extracting features from Layer {layer_idx}...")
        
        print("Extracting training features...")
        X_train = []
        for path in tqdm(X_paths_train, desc=f"Train Layer {layer_idx}"):
            feat = extract_wav2vec2_features(path, processor, model, layer_idx)
            if feat is not None:
                X_train.append(feat)
            else:
                X_train.append(np.zeros(768))  # Fallback for failed extractions
        
        print("Extracting test features...")
        X_test = []
        for path in tqdm(X_paths_test, desc=f"Test Layer {layer_idx}"):
            feat = extract_wav2vec2_features(path, processor, model, layer_idx)
            if feat is not None:
                X_test.append(feat)
            else:
                X_test.append(np.zeros(768))
        
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        
        print(f"Feature shapes: Train={X_train.shape}, Test={X_test.shape}")
        
        # Normalize features
        print(f"[6.{layer_idx}] Normalizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM
        print(f"[7.{layer_idx}] Training SVM...")
        svm = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = svm.predict(X_train_scaled)
        y_test_pred = svm.predict(X_test_scaled)
        
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\n📊 RESULTS - LAYER {layer_idx}")
        print(f"Train F1-score: {train_f1:.4f}")
        print(f"Test F1-score:  {test_f1:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        
        results.append({
            'layer': layer_idx,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'test_acc': test_acc
        })
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print("\n" + "="*70)
    print("LAYER-WISE COMPARISON SUMMARY")
    print("="*70)
    print("\n{:<10} {:<15} {:<15} {:<15}".format("Layer", "Train F1", "Test F1", "Test Accuracy"))
    print("-" * 70)
    
    best_test_f1 = 0
    best_layer = None
    
    for r in results:
        print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            r['layer'], r['train_f1'], r['test_f1'], r['test_acc']
        ))
        if r['test_f1'] > best_test_f1:
            best_test_f1 = r['test_f1']
            best_layer = r['layer']
    
    print("-" * 70)
    print(f"\n🏆 BEST LAYER: Layer {best_layer} (Test F1 = {best_test_f1:.4f})")
    
    # Detailed report for best layer
    print(f"\n" + "="*70)
    print(f"DETAILED CLASSIFICATION REPORT - BEST LAYER {best_layer}")
    print("="*70)
    
    # Re-extract and train on best layer for detailed report
    print(f"\nRe-training with best layer ({best_layer}) for detailed metrics...")
    
    X_train_best = []
    for path in tqdm(X_paths_train, desc=f"Final Train Layer {best_layer}"):
        feat = extract_wav2vec2_features(path, processor, model, best_layer)
        if feat is not None:
            X_train_best.append(feat)
        else:
            X_train_best.append(np.zeros(768))
    
    X_test_best = []
    for path in tqdm(X_paths_test, desc=f"Final Test Layer {best_layer}"):
        feat = extract_wav2vec2_features(path, processor, model, best_layer)
        if feat is not None:
            X_test_best.append(feat)
        else:
            X_test_best.append(np.zeros(768))
    
    X_train_best = np.array(X_train_best, dtype=np.float32)
    X_test_best = np.array(X_test_best, dtype=np.float32)
    
    scaler_best = StandardScaler()
    X_train_best = scaler_best.fit_transform(X_train_best)
    X_test_best = scaler_best.transform(X_test_best)
    
    svm_best = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=RANDOM_STATE)
    svm_best.fit(X_train_best, y_train)
    y_test_pred_best = svm_best.predict(X_test_best)
    
    report_text = classification_report(y_test, y_test_pred_best, target_names=['Clean', 'Block'])
    print("\n" + report_text)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKEY FINDINGS:")
    print("- Wav2Vec2 used as FEATURE EXTRACTOR (frozen, no training)")
    print("- SVM classifier (same as MFCC/CQCC/SFFCC/ZTWCC)")
    print("- Best performing layer: Layer " + str(best_layer))
    print("- Best Test F1-Score: " + str(best_test_f1))
    print("- Feature dimension: 768 (from Wav2Vec2 hidden states)")
    print("\nThis is now a FAIR comparison with other feature methods!")

if __name__ == "__main__":
    main()
