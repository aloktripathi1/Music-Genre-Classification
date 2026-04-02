"""
EXP_002: Scratch CNN Baseline
Simple CNN from scratch (no pretrained weights) for genre classification.
This is the true baseline before moving to pretrained models.

Expected: ~0.50-0.65 LB (weak, but proves scratch CNN was attempted)
Runtime: ~30 min on T4
"""

import os, glob, random, warnings, time
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ─── Config ───
DATA_ROOT = os.path.expanduser("~/data/messy_mashup")
# DATA_ROOT = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup"
OUTPUT_DIR = "./outputs"
STEMS_DIR = os.path.join(DATA_ROOT, "genres_stems")
NOISE_DIR = os.path.join(DATA_ROOT, "ESC-50-master", "audio")
TEST_DIR  = os.path.join(DATA_ROOT, "mashups")
TEST_CSV  = os.path.join(DATA_ROOT, "test.csv")

SR = 22050
DURATION = 10.0
TARGET_LEN = int(SR * DURATION)
N_MELS = 128
N_FFT = 2048
HOP = 512

GENRES = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE = {i: g for g, i in GENRE2IDX.items()}
STEMS = ['drums', 'vocals', 'bass']

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── WandB ───
import wandb
wandb.login(key="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")
run = wandb.init(
    entity="23f3003225-indian-institute-of-technology-madras",
    project="23f3003225-dl-genai-project",
    name="exp_002_scratch_cnn",
    config={"backbone": "scratch_cnn", "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR},
    tags=["scratch", "cnn", "baseline"],
)

# ─── Data Index ───
stem_index = {g: {st: [] for st in STEMS} for g in GENRES}
song_index = {g: [] for g in GENRES}

for genre in GENRES:
    gp = os.path.join(STEMS_DIR, genre)
    for song in sorted(os.listdir(gp)):
        sd = os.path.join(gp, song)
        if not os.path.isdir(sd): continue
        avail = []
        for st in STEMS:
            fp = os.path.join(sd, f"{st}.wav")
            if os.path.exists(fp):
                stem_index[genre][st].append(fp)
                avail.append(st)
        if avail:
            song_index[genre].append({'dir': sd, 'stems': avail})

noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))

# train/val split
train_stems = {g: {st: [] for st in STEMS} for g in GENRES}
val_songs = {g: [] for g in GENRES}
for genre in GENRES:
    songs = song_index[genre].copy()
    random.shuffle(songs)
    sp = int(0.85 * len(songs))
    val_songs[genre] = songs[sp:]
    dirs = {s['dir'] for s in songs[:sp]}
    for st in STEMS:
        train_stems[genre][st] = [fp for fp in stem_index[genre][st] if os.path.dirname(fp) in dirs]

# ─── Audio ───
def load_wav(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        if len(y) < TARGET_LEN: y = np.pad(y, (0, TARGET_LEN - len(y)))
        elif len(y) > TARGET_LEN:
            s = random.randint(0, len(y) - TARGET_LEN)
            y = y[s:s+TARGET_LEN]
        return y.astype(np.float32)
    except:
        return np.zeros(TARGET_LEN, dtype=np.float32)

def wav_to_mel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    return S_db.astype(np.float32)

# ─── Dataset ───
class MashupDataset(Dataset):
    def __init__(self, stem_idx, noise_files, n=500):
        self.stem_idx = stem_idx
        self.noise_files = noise_files
        self.samples = []
        for g in GENRES:
            self.samples.extend([GENRE2IDX[g]] * n)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        gi = self.samples[idx]
        g = IDX2GENRE[gi]
        stems = []
        for st in STEMS:
            av = self.stem_idx[g][st]
            if av: stems.append(load_wav(random.choice(av)) * random.uniform(0.5, 1.5))
        if not stems:
            return torch.zeros(1, N_MELS, TARGET_LEN // HOP + 1), gi
        mix = np.sum(stems, axis=0)
        # add noise
        if self.noise_files:
            noise = load_wav(random.choice(self.noise_files))
            mix = mix + noise * 0.1
        peak = np.max(np.abs(mix))
        if peak > 1e-6: mix = mix / peak
        mel = wav_to_mel(mix)
        # simple normalization
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return torch.from_numpy(mel).unsqueeze(0), gi

class ValDataset(Dataset):
    def __init__(self, song_idx):
        self.items = []
        for g in GENRES:
            for s in song_idx[g]: self.items.append((s, GENRE2IDX[g]))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        si, lb = self.items[idx]
        stems = [load_wav(os.path.join(si['dir'], f"{st}.wav")) for st in si['stems']]
        mix = np.sum(stems, axis=0)
        peak = np.max(np.abs(mix))
        if peak > 1e-6: mix = mix / peak
        mel = wav_to_mel(mix)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return torch.from_numpy(mel).unsqueeze(0), lb

class TestDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(TEST_CSV, dtype={'id': str})
        self.paths = []
        for _, r in self.df.iterrows():
            p = None
            for pat in [f"song{str(r['id']).zfill(4)}.wav", f"{r['id']}.wav"]:
                fp = os.path.join(TEST_DIR, pat)
                if os.path.exists(fp): p = fp; break
            self.paths.append(p)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        p = self.paths[idx]
        if p:
            mel = wav_to_mel(load_wav(p))
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            return torch.from_numpy(mel).unsqueeze(0), str(self.df.iloc[idx]['id'])
        return torch.zeros(1, N_MELS, TARGET_LEN // HOP + 1), str(self.df.iloc[idx]['id'])

# ─── SCRATCH CNN MODEL (no pretrained weights) ───
class ScratchCNN(nn.Module):
    # simple CNN built from basic layers - no pretrained weights
    def __init__(self, num_classes=10):
        super().__init__()
        # 4 conv blocks with increasing channels
        self.features = nn.Sequential(
            # block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ScratchCNN().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"Scratch CNN: {params/1e6:.2f}M parameters (all randomly initialized)")

# ─── Training ───
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

val_ds = ValDataset(val_songs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

best_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_ds = MashupDataset(train_stems, noise_files, n=500)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    # train
    model.train()
    total_loss, n = 0, 0
    for mel, labels in tqdm(train_loader, desc=f"E{epoch}", leave=False):
        mel, labels = mel.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(mel), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    scheduler.step()

    # eval
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for mel, lb in val_loader:
            mel = mel.to(DEVICE)
            preds.extend(model(mel).argmax(1).cpu().numpy())
            labs.extend(lb.numpy())
    f1 = f1_score(labs, preds, average='macro')
    acc = np.mean(np.array(preds) == np.array(labs))

    tag = ""
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_scratch_cnn.pth'))
        tag = " ★"

    wandb.log({"epoch": epoch, "loss": total_loss/n, "val_f1": f1, "val_acc": acc})
    print(f"E{epoch:02d}/{EPOCHS} | loss={total_loss/n:.4f} | f1={f1:.4f} | acc={acc:.4f} | {time.time()-t0:.0f}s{tag}")

print(f"\nBest val F1: {best_f1:.4f}")

# ─── Inference ───
print("\n--- Inference ---")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_scratch_cnn.pth'), weights_only=True))
model.eval()

test_ds = TestDataset()
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

all_preds, all_ids = [], []
with torch.no_grad():
    for mel, ids in test_loader:
        mel = mel.to(DEVICE)
        all_preds.extend(model(mel).argmax(1).cpu().numpy())
        all_ids.extend(ids)

test_df = pd.read_csv(TEST_CSV, dtype={'id': str})
pred_dict = {str(id_): IDX2GENRE[p] for id_, p in zip(all_ids, all_preds)}
test_df['genre'] = test_df['id'].apply(lambda x: pred_dict.get(str(x), 'rock'))
test_df[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, 'submission_scratch_cnn.csv'), index=False)
print("Submission saved!")
print(test_df['genre'].value_counts().sort_index())

wandb.log({"best_f1": best_f1, "status": "complete"})
artifact = wandb.Artifact("scratch_cnn", type="model")
artifact.add_file(os.path.join(OUTPUT_DIR, 'best_scratch_cnn.pth'))
run.log_artifact(artifact)
wandb.finish()
print(f"\nDone! Scratch CNN baseline — best val F1 = {best_f1:.4f}")