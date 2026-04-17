# Architecture: Mel Spectrogram (librosa) → InstanceNorm → EfficientNet-B0 → GeM → Linear(10)
# Key: On-the-fly mashup augmentation + SpecAugment + Mixup
# No torchaudio dependency - uses librosa for all audio processing.


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

SR, DURATION   = 22050, 10.0
TARGET_LEN     = int(SR * DURATION)
N_MELS, N_FFT  = 128, 2048
HOP_LENGTH     = 512
FMIN, FMAX     = 20, 8000

GENRES    = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE = {i: g for g, i in GENRE2IDX.items()}
STEMS     = ['drums', 'vocals', 'bass']

SAMPLES_PER_GENRE = 1000
BATCH_SIZE        = 32
EPOCHS            = 35
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.1
GRAD_CLIP         = 1.0
NUM_WORKERS       = 4
MIXUP_ALPHA       = 0.4
FREQ_MASK, TIME_MASK = 27, 80
STEM_WEIGHTS = {'drums': 0.45, 'vocals': 0.35, 'bass': 0.20}

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Config: {SAMPLES_PER_GENRE*10} mashups/epoch, bs={BATCH_SIZE}, epochs={EPOCHS}")


# WANDB
os.system('pip install wandb timm --quiet')
import wandb
wandb.login(key="wandb-api-key")

run = wandb.init(
    entity="23f3003225-indian-institute-of-technology-madras",
    project="23f3003225-dl-genai-project",
    name="exp_003_cnn_v2",
    config=dict(sr=SR, n_mels=N_MELS, backbone="efficientnet_b0", pooling="gem",
                samples_per_genre=SAMPLES_PER_GENRE, batch_size=BATCH_SIZE, epochs=EPOCHS,
                lr=LR, label_smoothing=LABEL_SMOOTHING, mixup=MIXUP_ALPHA),
    tags=["cnn", "efficientnet", "specaugment", "mixup"],
    job_type="train",
)


# DATA INDEX & TRAIN/VAL SPLIT
print("\nBuilding data index")
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


# AUDIO UTILITIES (librosa)
def load_wav(path, sr=SR, target_len=TARGET_LEN):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start + target_len]
        return y.astype(np.float32)
    except:
        return np.zeros(target_len, dtype=np.float32)


def wav_to_mel(y, sr=SR):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    return S_db.astype(np.float32)


print("Audio utils ready.")


# SPECAUGMENT (pure torch on GPU)
def spec_augment(spec, freq_mask=FREQ_MASK, time_mask=TIME_MASK, n_freq=2, n_time=2):
    _, _, n_mels, n_frames = spec.shape
    aug = spec.clone()
    for _ in range(n_freq):
        f = random.randint(0, freq_mask)
        f0 = random.randint(0, max(0, n_mels - f))
        aug[:, :, f0:f0+f, :] = 0
    for _ in range(n_time):
        t = random.randint(0, time_mask)
        t0 = random.randint(0, max(0, n_frames - t))
        aug[:, :, :, t0:t0+t] = 0
    return aug


# DATASETS
class MashupDataset(Dataset):
    def __init__(self, stem_idx, noise_files, samples_per_genre=1000, augment=True):
        self.stem_idx = stem_idx
        self.noise_files = noise_files
        self.augment = augment
        self.samples = []
        for genre in GENRES:
            for _ in range(samples_per_genre):
                self.samples.append(GENRE2IDX[genre])

    def __len__(self):
        return len(self.samples)

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
            mel = np.zeros((N_MELS, TARGET_LEN // HOP_LENGTH + 1), dtype=np.float32)
            return torch.from_numpy(mel).unsqueeze(0), genre_idx

        mix = np.sum(stems_wav, axis=0)

        if self.augment:
            mix = np.roll(mix, random.randint(-SR, SR))
            for _ in range(random.randint(0, 2)):
                noise = load_wav(random.choice(self.noise_files))
                snr_db = random.uniform(5.0, 25.0)
                sig_pwr = np.mean(mix ** 2) + 1e-10
                nse_pwr = np.mean(noise ** 2) + 1e-10
                scale = np.sqrt(sig_pwr / (nse_pwr * 10 ** (snr_db / 10)))
                mix = mix + noise * scale
            if random.random() < 0.3:
                mix = np.clip(mix * random.uniform(1.2, 3.0), -1, 1)

        peak = np.max(np.abs(mix))
        if peak > 1e-6:
            mix = mix / peak * random.uniform(0.7, 1.0)

        mel = wav_to_mel(mix)
        return torch.from_numpy(mel).unsqueeze(0), genre_idx


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
        mix = np.sum(stems, axis=0)
        peak = np.max(np.abs(mix))
        if peak > 1e-6: mix = mix / peak
        mel = wav_to_mel(mix)
        return torch.from_numpy(mel).unsqueeze(0), label


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
            mel = wav_to_mel(load_wav(path))
            mel_t = torch.from_numpy(mel).unsqueeze(0)
        else:
            mel_t = torch.zeros(1, N_MELS, TARGET_LEN // HOP_LENGTH + 1)
        return mel_t, str(self.df.iloc[idx]['id'])

print("Datasets ready.")


# MODEL
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class GenreClassifier(nn.Module):
    def __init__(self, num_classes=10, backbone='efficientnet_b0', pretrained=True):
        super().__init__()
        self.inst_norm = nn.InstanceNorm2d(1)
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                           in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(nn.LayerNorm(nf), nn.Dropout(0.5), nn.Linear(nf, num_classes))

    def forward(self, x, augment=False):
        x = self.inst_norm(x)
        if augment:
            x = spec_augment(x)
        feat = self.backbone(x)
        pooled = self.gem(feat)
        return self.head(pooled)


model = GenreClassifier().to(DEVICE)
dummy = torch.randn(2, 1, N_MELS, TARGET_LEN // HOP_LENGTH + 1).to(DEVICE)
with torch.no_grad(): out = model(dummy)
print(f"Model output: {out.shape}, Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
del dummy, out; gc.collect(); torch.cuda.empty_cache()


# MIXUP
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

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
                logits = model(mel, augment=True)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            with autocast():
                logits = model(mel, augment=True)
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
            logits = model(mel, augment=False)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, acc, np.array(all_preds), np.array(all_labels)


# RUN TRAINING
print("\nStarting training")

val_ds = ValDataset(val_songs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
print(f"Val: {len(val_ds)} samples")

model = GenreClassifier().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = GradScaler()

best_f1 = 0.0
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
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_cnn.pth'))
        tag = " (best)"

    print(f"E{epoch:02d}/{EPOCHS} | loss={loss:.4f} | f1={val_f1:.4f} | acc={val_acc:.4f} | lr={lr:.6f} | {elapsed:.0f}s{tag}")

print(f"\nBest val F1: {best_f1:.4f}")


# RESULTS
print("\nResults")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history['loss']); axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
axes[1].plot(history['val_f1'], label='F1'); axes[1].plot(history['val_acc'], label='Acc', alpha=0.7)
axes[1].set_title('Validation'); axes[1].legend()
axes[2].plot(history['lr']); axes[2].set_title('LR')
plt.suptitle(f'Best F1: {best_f1:.4f}'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
wandb.log({"plots/curves": wandb.Image(fig)}); plt.close()

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_cnn.pth'), weights_only=True))
val_f1, val_acc, preds, labels = evaluate(model, val_loader)

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=GENRES).plot(ax=ax, cmap='Blues', xticks_rotation=45)
ax.set_title(f'F1={val_f1:.4f}'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
wandb.log({"plots/confusion": wandb.Image(fig)}); plt.close()

print(classification_report(labels, preds, target_names=GENRES))


# INFERENCE + 5x TTA
print("\nInference")

@torch.no_grad()
def predict_tta(model, loader, n_tta=5):
    model.eval()
    all_ids = []
    for _, ids in loader:
        all_ids.extend(ids)

    all_probs = []
    for _ in range(n_tta):
        round_probs = []
        for mel, _ in loader:
            mel = mel.to(DEVICE)
            with autocast():
                logits = model(mel, augment=False)
            round_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(round_probs))

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs.argmax(1), all_ids, avg_probs


test_ds = TestDataset(TEST_DIR, TEST_CSV)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
print(f"Test: {len(test_ds)} samples")

preds, ids, probs = predict_tta(model, test_loader, n_tta=5)
print(f"Predictions: {Counter(preds)}")


# SUBMISSION
test_df = pd.read_csv(TEST_CSV, dtype={'id': str})
pred_dict = {str(id_): IDX2GENRE[p] for id_, p in zip(ids, preds)}
test_df['genre'] = test_df['id'].apply(lambda x: pred_dict.get(str(x), 'rock'))
test_df[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
print(f"\nSubmission saved: {os.path.join(OUTPUT_DIR, 'submission.csv')}")
print(test_df['genre'].value_counts().sort_index())

np.save(os.path.join(OUTPUT_DIR, 'test_probs_efficientnet.npy'), probs)


# WANDB FINISH
wandb.log({"best_f1": best_f1, "status": "complete"})
art = wandb.Artifact("cnn_efficientnet_v2", type="model")
art.add_file(os.path.join(OUTPUT_DIR, 'best_cnn.pth'))
run.log_artifact(art)
wandb.finish()
print(f"\nTraining complete. best val_f1={best_f1:.4f}")