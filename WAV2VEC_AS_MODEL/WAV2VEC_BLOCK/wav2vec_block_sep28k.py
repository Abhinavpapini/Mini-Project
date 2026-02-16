# ============================================================
# Wav2Vec2 + Classifier (Clean vs Block) — SEP-28K
# Deep Learning Approach with Pre-trained Wav2Vec2
# Python 3.10+ | GPU Recommended
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"c:\23071A05(cse-b) mini project"
AUDIO_DIR = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "clips")
LABELS_PATH = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "SEP-28k_labels.csv")

# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
TEST_SIZE = 0.33
RANDOM_STATE = 42
TARGET_SR = 16000  # Wav2Vec2 requires 16kHz

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

def elapsed(t0):
    return f"{(time.time() - t0)/60:.2f} min"

# ============================================================
# CUSTOM DATASET
# ============================================================

class StutteringDataset(Dataset):
    def __init__(self, file_paths, labels, processor, target_sr=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.target_sr = target_sr
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        try:
            audio, sr = librosa.load(self.file_paths[idx], sr=self.target_sr)
        except Exception as e:
            # Return zeros if file cannot be loaded
            audio = np.zeros(int(3 * self.target_sr))
        
        # Process audio
        inputs = self.processor(
            audio, 
            sampling_rate=self.target_sr, 
            return_tensors="pt",
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================================
# WAV2VEC2 CLASSIFIER MODEL
# ============================================================

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=2, freeze_base=False):
        super().__init__()
        
        # Load pre-trained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze Wav2Vec2 weights if specified (faster training, use as feature extractor)
        if freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Classification head
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input_values):
        # Extract features from Wav2Vec2
        outputs = self.wav2vec2(input_values)
        
        # Use mean pooling over time dimension
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_size)
        pooled = torch.mean(hidden_states, dim=1)  # (batch, hidden_size)
        
        # Classify
        logits = self.classifier(pooled)
        return logits

# ============================================================
# COLLATE FUNCTION (for variable length audio)
# ============================================================

def collate_fn(batch):
    # Find max length in batch
    max_len = max([item['input_values'].shape[0] for item in batch])
    
    # Pad all sequences to max length
    input_values = []
    labels = []
    
    for item in batch:
        input_val = item['input_values']
        
        # Pad if necessary
        if input_val.shape[0] < max_len:
            padding = torch.zeros(max_len - input_val.shape[0])
            input_val = torch.cat([input_val, padding])
        
        input_values.append(input_val)
        labels.append(item['label'])
    
    return {
        'input_values': torch.stack(input_values),
        'labels': torch.stack(labels)
    }

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_values = batch['input_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_values)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_values)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc, all_labels, all_preds

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("WAV2VEC2 BLOCK CLASSIFICATION")
    print("="*60)
    
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
    
    labels = [0] * len(clean_df) + [1] * len(block_df)
    
    # Train-test split
    print("\n[3] Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Load Wav2Vec2 processor
    print("\n[4] Loading Wav2Vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Create datasets
    print("[5] Creating PyTorch datasets...")
    train_dataset = StutteringDataset(X_train, y_train, processor, TARGET_SR)
    test_dataset = StutteringDataset(X_test, y_test, processor, TARGET_SR)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print(f"\n[6] Initializing Wav2Vec2 classifier on {DEVICE}...")
    model = Wav2Vec2Classifier(
        model_name="facebook/wav2vec2-base",
        num_classes=2,
        freeze_base=False  # Set True for faster training (feature extraction only)
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n[7] Training for {NUM_EPOCHS} epochs...")
    print("="*60)
    
    t0 = time.time()
    best_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Evaluate
        test_loss, test_f1, test_acc, _, _ = evaluate(
            model, test_loader, criterion, DEVICE
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test F1:  {test_f1:.4f} | Test Acc:  {test_acc:.4f}")
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), "best_wav2vec_block_model.pt")
            print(f"✅ Best model saved! (F1: {best_f1:.4f})")
    
    print("\n" + "="*60)
    print(f"Training completed in {elapsed(t0)}")
    print(f"Best Test F1-Score: {best_f1:.4f}")
    
    # Load best model and get final results
    print("\n[8] Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_wav2vec_block_model.pt"))
    _, final_f1, final_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS (WAV2VEC2 - BLOCK CLASSIFICATION)")
    print("="*60)
    print(f"Test F1-Score: {final_f1:.4f}")
    print(f"Test Accuracy: {final_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Clean', 'Block']))
    print("="*60)

if __name__ == "__main__":
    main()
