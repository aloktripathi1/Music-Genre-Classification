"""
EXP_006 — ResNet-50 on Mel Spectrograms
Target: 0.93+ standalone, 0.95+ in 3-model ensemble
Platform: Lightning.ai L4 GPU

NO torchaudio dependency — uses librosa for all audio processing.

Architecture: Waveform → librosa MelSpec → InstanceNorm → ResNet-50 (ImageNet) → GeM → Linear(10)
Same mashup augmentation as CNN/AST experiments.

Run: pip install librosa timm wandb --quiet && python main.py
"""

import os, glob, random, warnings, time, gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import librosa
import timm

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# CONFIG
DATA_ROOT  = os.path.expanduser("~/data/messy_mashup")
OUTPUT_DIR = "./outputs"
STEMS_DIR  = os.path.join(DATA_ROOT, "genres_stems")
NOISE_DIR  = os.path.join(DATA_ROOT, "ESC-50-master", "audio")
TEST_DIR   = os.path.join(DATA_ROOT, "mashups")
TEST_CSV   = os.path.join(DATA_ROOT, "test.csv")

SR         = 22050
DURATION   = 10.0
TARGET_LEN = int(SR * DURATION)
N_MELS     = 128
N_FFT      = 2048
HOP_LENGTH = 512
FMIN, FMAX = 20, 8000

GENRES    = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE = {i: g for g, i in GENRE2IDX.items()}
STEMS     = ['drums', 'vocals', 'bass']

SAMPLES_PER_GENRE = 1200
BATCH_SIZE        = 24
EPOCHS            = 25
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.1
GRAD_CLIP         = 1.0
NUM_WORKERS       = 4
MIXUP_ALPHA       = 0.4
STEM_WEIGHTS      = {'drums': 0.45, 'vocals': 0.35, 'bass': 0.20}

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Config: {SAMPLES_PER_GENRE*10} mashups/epoch, bs={BATCH_SIZE}, epochs={EPOCHS}")

# WANDB
import wandb
wandb.login(key="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")
run = wandb.init(
    entity="23f3003225-indian-institute-of-technology-madras",
    project="23f3003225-dl-genai-project",
    name="exp_006_resnet50",
    config=dict(sr=SR, n_mels=N_MELS, backbone="resnet50", pooling="gem",
                samples_per_genre=SAMPLES_PER_GENRE, batch_size=BATCH_SIZE, epochs=EPOCHS,
                lr=LR, label_smoothing=LABEL_SMOOTHING, mixup=MIXUP_ALPHA),
    tags=["resnet50", "librosa", "gem", "mixup"],
    job_type="train",
)

# DATA INDEX
print("\n--- Building data index ---")
stem_index = {g: {st: [] for st in STEMS} for g in GENRES}
song_index = {g: [] for g in GENRES}

for genre in GENRES:
    gp = os.path.join(STEMS_DIR, genre)
    songs = sorted(s for s in os.listdir(gp) if os.path.isdir(os.path.join(gp, s)))
    for song in songs:
        song_dir = os.path.join(gp, song)
        avail = []
        for st in STEMS:
            fp = os.path.join(song_dir, f"{st}.wav")
            if os.path.exists(fp):
                stem_index[genre][st].append(fp)
                avail.append(st)
        if avail:
            song_index[genre].append({'dir': song_dir, 'stems': avail})

noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))
print(f"Noise clips: {len(noise_files)}")

train_stems = {g: {st: [] for st in STEMS} for g in GENRES}
val_songs   = {g: [] for g in GENRES}

for genre in GENRES:
    songs = song_index[genre].copy()
    random.shuffle(songs)
    split = int(0.85 * len(songs))
    train_list, val_list = songs[:split], songs[split:]
    val_songs[genre] = val_list
    train_dirs = {s['dir'] for s in train_list}
    for st in STEMS:
        train_stems[genre][st] = [fp for fp in stem_index[genre][st] if os.path.dirname(fp) in train_dirs]
    print(f"  {genre}: train={len(train_list)}, val={len(val_list)}")

# AUDIO — pure librosa, zero torchaudio
def load_wav(path, sr=SR, target_len=TARGET_LEN):
    """Load audio with librosa. Returns torch tensor."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        wav = torch.from_numpy(y).float()
        if len(wav) < target_len:
            wav = F.pad(wav, (0, target_len - len(wav)))
        elif len(wav) > target_len:
            start = random.randint(0, len(wav) - target_len)
            wav = wav[start:start + target_len]
        return wav
    except:
        return torch.zeros(target_len)


def wav_to_mel_tensor(wav_tensor):
    """Convert waveform tensor to log-mel spectrogram tensor using librosa.
    Returns: (1, n_mels, time) tensor ready for CNN input.
    """
    y = wav_tensor.numpy()
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    mel = torch.from_numpy(S_db).float()
    # Instance normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0)  # (1, n_mels, time)

print("Audio utils ready (librosa only).")

# DATASETS
class MashupDataset(Dataset):
    """On-the-fly mashup: drums from song A + vocals from song B + bass from song C."""
    def __init__(self, stem_idx, noise_files, samples_per_genre=1200, augment=True):
        self.stem_idx = stem_idx
        self.noise_files = noise_files
        self.augment = augment
        self.samples = []
        for genre in GENRES:
            for _ in range(samples_per_genre):
                self.samples.append(GENRE2IDX[genre])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        genre_idx = self.samples[idx]
        genre = IDX2GENRE[genre_idx]

        stems_wav = []
        for st in STEMS:
            available = self.stem_idx[genre][st]
            if not available: continue
            wav = load_wav(random.choice(available))
            gain = random.uniform(0.5, 1.5) * (STEM_WEIGHTS[st] / 0.33)
            stems_wav.append(wav * gain)

        if not stems_wav:
            return torch.zeros(1, N_MELS, TARGET_LEN // HOP_LENGTH + 1), genre_idx

        mix = torch.stack(stems_wav).sum(0)

        if self.augment:
            mix = torch.roll(mix, random.randint(-SR, SR))

            # ESC-50 noise (0-3 clips)
            for _ in range(random.randint(0, 3)):
                noise = load_wav(random.choice(self.noise_files))
                snr_db = random.uniform(3.0, 25.0)
                sig_pwr = mix.pow(2).mean() + 1e-10
                nse_pwr = noise.pow(2).mean() + 1e-10
                scale = (sig_pwr / (nse_pwr * 10 ** (snr_db / 10))).sqrt()
                mix = mix + noise * scale

            # Overdrive (30%)
            if random.random() < 0.3:
                mix = torch.clamp(mix * random.uniform(1.2, 3.0), -1, 1)

        # Normalize waveform
        peak = mix.abs().max()
        if peak > 1e-6:
            mix = mix / peak * random.uniform(0.7, 1.0)

        # Convert to mel spectrogram
        mel = wav_to_mel_tensor(mix)
        return mel, genre_idx


class ValDataset(Dataset):
    def __init__(self, song_index):
        self.items = []
        for genre in GENRES:
            for s in song_index[genre]:
                self.items.append((s, GENRE2IDX[genre]))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        song_info, label = self.items[idx]
        stems = [load_wav(os.path.join(song_info['dir'], f"{st}.wav")) for st in song_info['stems']]
        mix = torch.stack(stems).sum(0)
        peak = mix.abs().max()
        if peak > 1e-6: mix = mix / peak
        mel = wav_to_mel_tensor(mix)
        return mel, label


class TestDataset(Dataset):
    def __init__(self, test_dir, test_csv):
        self.df = pd.read_csv(test_csv, dtype={'id': str})
        self.paths = []
        for _, row in self.df.iterrows():
            path = None
            for pat in [f"song{str(row['id']).zfill(4)}.wav", f"{row['id']}.wav", f"song{row['id']}.wav"]:
                p = os.path.join(test_dir, pat)
                if os.path.exists(p): path = p; break
            self.paths.append(path)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        path = self.paths[idx]
        if path:
            wav = load_wav(path)
        else:
            wav = torch.zeros(TARGET_LEN)
        mel = wav_to_mel_tensor(wav)
        return mel, str(self.df.iloc[idx]['id'])

print("Datasets ready.")

# MODEL: ResNet-50 + GeM
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class GenreResNet(nn.Module):
    """
    Mel Spectrogram (precomputed) → ResNet-50 (ImageNet) → GeM → LayerNorm → Dropout → Linear(10)
    Input: (B, 1, 128, T) log-mel spectrogram
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True,
                                           in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features  # 2048
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(
            nn.LayerNorm(nf),
            nn.Dropout(0.4),
            nn.Linear(nf, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.gem(feat)
        return self.head(pooled)

model = GenreResNet().to(DEVICE)
dummy = torch.randn(2, 1, N_MELS, 432).to(DEVICE)
with torch.no_grad(): out = model(dummy)
print(f"Model output: {out.shape}, Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
del dummy, out; gc.collect(); torch.cuda.empty_cache()

# MIXUP
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# TRAINING
def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss, n = 0, 0
    for mel, labels in tqdm(loader, desc="Train", leave=False):
        mel, labels = mel.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if random.random() < 0.5:
            mel, y_a, y_b, lam = mixup_data(mel, labels, MIXUP_ALPHA)
            with autocast():
                logits = model(mel)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            with autocast():
                logits = model(mel)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for mel, labels in loader:
        mel = mel.to(DEVICE)
        with autocast():
            logits = model(mel)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, acc, np.array(all_preds), np.array(all_labels)

# TRAIN LOOP
print("\n--- Starting training ---")

val_ds = ValDataset(val_songs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
print(f"Val: {len(val_ds)} samples")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = GradScaler()

best_f1 = 0.0
patience = 0
history = {'loss': [], 'val_f1': [], 'val_acc': [], 'lr': []}

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_ds = MashupDataset(train_stems, noise_files, SAMPLES_PER_GENRE, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
    scheduler.step()
    val_f1, val_acc, vp, vl = evaluate(model, val_loader)
    lr = scheduler.get_last_lr()[0]
    elapsed = time.time() - t0

    history['loss'].append(loss)
    history['val_f1'].append(val_f1)
    history['val_acc'].append(val_acc)
    history['lr'].append(lr)
    wandb.log({"epoch": epoch, "loss": loss, "val_f1": val_f1, "val_acc": val_acc, "lr": lr})

    tag = ""
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_resnet50.pth'))
        tag = " ★"
        patience = 0
    else:
        patience += 1

    print(f"E{epoch:02d}/{EPOCHS} | loss={loss:.4f} | f1={val_f1:.4f} | acc={val_acc:.4f} | lr={lr:.6f} | {elapsed:.0f}s{tag}")

    if patience >= 7:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"\nBest val F1: {best_f1:.4f}")

# RESULTS
print("\n--- Results ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history['loss']); axes[0].set_title('Loss')
axes[1].plot(history['val_f1'], label='F1'); axes[1].plot(history['val_acc'], label='Acc', alpha=0.7)
axes[1].set_title('Validation'); axes[1].legend()
axes[2].plot(history['lr']); axes[2].set_title('LR')
plt.suptitle(f'ResNet-50 — Best F1: {best_f1:.4f}'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'resnet50_curves.png'), dpi=150)
wandb.log({"plots/resnet50_curves": wandb.Image(fig)}); plt.close()

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_resnet50.pth'), weights_only=True))
val_f1, val_acc, preds, labels = evaluate(model, val_loader)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=GENRES).plot(ax=ax, cmap='Blues', xticks_rotation=45)
ax.set_title(f'ResNet-50 — F1={val_f1:.4f}'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'resnet50_confusion.png'), dpi=150)
wandb.log({"plots/resnet50_confusion": wandb.Image(fig)}); plt.close()
print(classification_report(labels, preds, target_names=GENRES))

# INFERENCE + 5x TTA
print("\n--- Inference with TTA ---")

test_ds = TestDataset(TEST_DIR, TEST_CSV)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
print(f"Test: {len(test_ds)} samples")

@torch.no_grad()
def predict_tta(model, loader, n_tta=5):
    model.eval()
    all_ids = []
    for _, ids in loader: all_ids.extend(ids)

    all_probs = []
    for _ in range(n_tta):
        round_probs = []
        for mel, _ in loader:
            mel = mel.to(DEVICE)
            with autocast():
                logits = model(mel)
            round_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(round_probs))
    avg = np.mean(all_probs, axis=0)
    return avg.argmax(1), all_ids, avg

preds, ids, probs = predict_tta(model, test_loader, n_tta=5)
print(f"Predictions: {Counter(preds)}")

# SUBMISSION + SAVE PROBS FOR ENSEMBLE
test_df = pd.read_csv(TEST_CSV, dtype={'id': str})
pred_dict = {str(id_): IDX2GENRE[p] for id_, p in zip(ids, preds)}
test_df['genre'] = test_df['id'].apply(lambda x: pred_dict.get(str(x), 'rock'))
test_df[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, 'submission_resnet50.csv'), index=False)
print(f"\nSubmission saved: {os.path.join(OUTPUT_DIR, 'submission_resnet50.csv')}")
print(test_df['genre'].value_counts().sort_index())

# Save probs for 3-model ensemble
np.save(os.path.join(OUTPUT_DIR, 'test_probs_resnet50.npy'), probs)
print(f"Probs saved: {os.path.join(OUTPUT_DIR, 'test_probs_resnet50.npy')}")

# QUICK 3-MODEL ENSEMBLE (if CNN + AST probs exist)
print("\n--- 3-Model Ensemble ---")

cnn_path = os.path.expanduser('~/cnn/test_probs_efficientnet.npy')
ast_path = os.path.expanduser('~/ast/test_probs_ast.npy')

if os.path.exists(cnn_path) and os.path.exists(ast_path):
    cnn_probs = np.load(cnn_path)
    ast_probs = np.load(ast_path)
    resnet_probs = probs

    # Sweep weights
    weight_combos = [
        (0.15, 0.55, 0.30),  # AST heavy
        (0.15, 0.50, 0.35),
        (0.20, 0.50, 0.30),
        (0.20, 0.45, 0.35),
        (0.15, 0.45, 0.40),  # balanced AST + ResNet
        (0.20, 0.40, 0.40),
        (0.10, 0.50, 0.40),
        (0.10, 0.45, 0.45),  # minimal CNN
    ]

    for w_c, w_a, w_r in weight_combos:
        ens = w_c * cnn_probs + w_a * ast_probs + w_r * resnet_probs
        ens_preds = ens.argmax(1)
        fname = f"submission_3way_{int(w_c*100)}_{int(w_a*100)}_{int(w_r*100)}.csv"
        sub = test_df.copy()
        sub['genre'] = [IDX2GENRE[p] for p in ens_preds]
        sub[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print(f"  CNN={w_c} AST={w_a} ResNet={w_r} → {fname}")

    print("\nRecommended: submit submission_3way_15_50_35.csv first")
else:
    print("CNN/AST probs not found — submit ResNet standalone first")

# WANDB FINISH
wandb.log({"best_f1": best_f1, "status": "complete"})
art = wandb.Artifact("resnet50_model", type="model")
art.add_file(os.path.join(OUTPUT_DIR, 'best_resnet50.pth'))
run.log_artifact(art)
wandb.finish()
print(f"\n✅ Done — best val_f1={best_f1:.4f}")
