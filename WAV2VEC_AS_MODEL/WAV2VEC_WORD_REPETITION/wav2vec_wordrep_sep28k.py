# ============================================================
# Wav2Vec2 + Classifier (Clean vs Word Repetition) — SEP-28K
# Deep Learning Approach with Pre-trained Wav2Vec2
# Python 3.10+ | GPU Recommended
# ============================================================

import os
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

PROJECT_ROOT = r"c:\23071A05(cse-b) mini project"
AUDIO_DIR = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "clips")
LABELS_PATH = os.path.join(PROJECT_ROOT, "ml-stuttering-events-dataset", "SEP-28k_labels.csv")

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
TEST_SIZE = 0.33
RANDOM_STATE = 42
TARGET_SR = 16000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Using device: {DEVICE}")

def get_audio_path(r):
    return os.path.join(AUDIO_DIR, r['Show'], str(int(r['EpId'])), f"{r['Show']}_{int(r['EpId'])}_{int(r['ClipId'])}.wav")

class StutteringDataset(Dataset):
    def __init__(self, file_paths, labels, processor, target_sr=16000):
        self.file_paths, self.labels, self.processor, self.target_sr = file_paths, labels, processor, target_sr
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        try:
            audio, _ = librosa.load(self.file_paths[idx], sr=self.target_sr)
        except:
            audio = np.zeros(int(3 * self.target_sr))
        inputs = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        return {'input_values': inputs.input_values.squeeze(0), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=2, freeze_base=False):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, input_values):
        return self.classifier(torch.mean(self.wav2vec2(input_values).last_hidden_state, dim=1))

def collate_fn(batch):
    max_len = max([item['input_values'].shape[0] for item in batch])
    input_values, labels = [], []
    for item in batch:
        input_val = item['input_values']
        if input_val.shape[0] < max_len:
            input_val = torch.cat([input_val, torch.zeros(max_len - input_val.shape[0])])
        input_values.append(input_val)
        labels.append(item['label'])
    return {'input_values': torch.stack(input_values), 'labels': torch.stack(labels)}

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(dataloader, desc="Training"):
        input_values, labels = batch['input_values'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), f1_score(all_labels, all_preds), accuracy_score(all_labels, all_preds)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values, labels = batch['input_values'].to(device), batch['labels'].to(device)
            logits = model(input_values)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), f1_score(all_labels, all_preds), accuracy_score(all_labels, all_preds), all_labels, all_preds

def main():
    print("\n" + "="*60)
    print("WAV2VEC2 WORD REPETITION CLASSIFICATION")
    print("="*60)
    
    df = pd.read_csv(LABELS_PATH)
    audio_shows = {d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))}
    df = df[df["Show"].isin(audio_shows)]
    
    clean_df = df[(df["NoStutteredWords"] > 0) & (df["Block"] == 0) & (df["Prolongation"] == 0) &
                  (df["SoundRep"] == 0) & (df["WordRep"] == 0) & (df["Interjection"] == 0)]
    wordrep_df = df[df["WordRep"] > 0]
    
    clean_df = clean_df[clean_df.apply(lambda r: os.path.exists(get_audio_path(r)), axis=1)]
    wordrep_df = wordrep_df[wordrep_df.apply(lambda r: os.path.exists(get_audio_path(r)), axis=1)]
    
    print(f"Clean: {len(clean_df)} | Word Repetition: {len(wordrep_df)}")
    
    file_paths = [get_audio_path(r) for _, r in clean_df.iterrows()] + [get_audio_path(r) for _, r in wordrep_df.iterrows()]
    labels = [0] * len(clean_df) + [1] * len(wordrep_df)
    
    X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    train_loader = DataLoader(StutteringDataset(X_train, y_train, processor, TARGET_SR), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(StutteringDataset(X_test, y_test, processor, TARGET_SR), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model = Wav2Vec2Classifier(num_classes=2, freeze_base=False).to(DEVICE)
    criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_f1, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_f1, test_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f} | Test - Loss: {test_loss:.4f}, F1: {test_f1:.4f}")
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), "best_wav2vec_wordrep_model.pt")
    
    model.load_state_dict(torch.load("best_wav2vec_wordrep_model.pt"))
    _, final_f1, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n🎯 Best F1: {best_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Clean', 'WordRep']))

if __name__ == "__main__":
    main()
