"""
EXP_004 — AST v2 (Audio Spectrogram Transformer)
Fixes from v1:
  - FIXED: differential LR was broken (head got 0 params → trained at 1e-5 instead of 1e-3)
  - More samples: 1200/genre (12k/epoch vs 8k)
  - Mixup α=0.3 on waveforms
  - 30 epochs with proper warmup
  - Stronger augmentation (more noise, tempo-like stretch via resampling)
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

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import ASTFeatureExtractor, ASTForAudioClassification

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

SR         = 16000
DURATION   = 10.0
TARGET_LEN = int(SR * DURATION)

GENRES    = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE = {i: g for g, i in GENRE2IDX.items()}
STEMS     = ['drums', 'vocals', 'bass']

SAMPLES_PER_GENRE = 1200
BATCH_SIZE        = 8
ACCUM_STEPS       = 4
EPOCHS            = 30
LR_BACKBONE       = 5e-6
LR_HEAD           = 5e-4
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.1
GRAD_CLIP         = 1.0
NUM_WORKERS       = 4
WARMUP_EPOCHS     = 3
MIXUP_ALPHA       = 0.3
STEM_WEIGHTS      = {'drums': 0.45, 'vocals': 0.35, 'bass': 0.20}

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Config: {SAMPLES_PER_GENRE*10} mashups/epoch, bs={BATCH_SIZE}x{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS}, epochs={EPOCHS}")

# WANDB
import wandb
wandb.login(key="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")
run = wandb.init(
    entity="23f3003225-indian-institute-of-technology-madras",
    project="23f3003225-dl-genai-project",
    name="exp_004_ast_v2",
    config=dict(sr=SR, backbone="MIT/ast-finetuned-audioset-10-10-0.4593",
                samples_per_genre=SAMPLES_PER_GENRE, batch_size=BATCH_SIZE*ACCUM_STEPS,
                epochs=EPOCHS, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD,
                label_smoothing=LABEL_SMOOTHING, warmup=WARMUP_EPOCHS, mixup=MIXUP_ALPHA,
                fix="differential_lr_head_params"),
    tags=["ast", "v2", "fixed_lr", "mixup"], job_type="train",
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

# AUDIO
def load_wav(path, sr=SR, target_len=TARGET_LEN):
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

print("Audio utils ready.")

# DATASETS
class MashupDataset(Dataset):
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
            return torch.zeros(TARGET_LEN), genre_idx

        mix = torch.stack(stems_wav).sum(0)

        if self.augment:
            # Time shift
            mix = torch.roll(mix, random.randint(-SR, SR))

            # Pitch-like augmentation via resampling (simulates tempo shift)
            if random.random() < 0.3:
                rate = random.uniform(0.9, 1.1)
                mix_np = mix.numpy()
                mix_np = librosa.effects.time_stretch(mix_np, rate=rate)
                mix = torch.from_numpy(mix_np).float()
                if len(mix) < TARGET_LEN:
                    mix = F.pad(mix, (0, TARGET_LEN - len(mix)))
                elif len(mix) > TARGET_LEN:
                    mix = mix[:TARGET_LEN]

            # ESC-50 noise (0-3 clips for stronger augmentation)
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

            # Random low-pass filter (20% — simulates muffled audio)
            if random.random() < 0.2:
                mix_np = mix.numpy()
                cutoff = random.randint(2000, 6000)
                from scipy.signal import butter, sosfilt
                sos = butter(5, cutoff, btype='low', fs=SR, output='sos')
                mix_np = sosfilt(sos, mix_np).astype(np.float32)
                mix = torch.from_numpy(mix_np)

        peak = mix.abs().max()
        if peak > 1e-6: mix = mix / peak * random.uniform(0.7, 1.0)
        return mix, genre_idx


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
        return mix, label


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
        wav = load_wav(path) if path else torch.zeros(TARGET_LEN)
        return wav, str(self.df.iloc[idx]['id'])

print("Datasets ready.")

# MODEL
print("\n--- Loading AST ---")
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL)

class ASTGenreClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            AST_MODEL, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    def forward(self, input_values):
        return self.ast(input_values=input_values).logits

model = ASTGenreClassifier(num_classes=10).to(DEVICE)

# FIX: Proper differential LR
# In AST, the classifier is at model.ast.classifier.dense
# v1 bug: checked for 'classifier' which matched nothing → 0 head params
head_params = []
backbone_params = []
for name, param in model.named_parameters():
    if 'classifier' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

head_count = sum(p.numel() for p in head_params)
backbone_count = sum(p.numel() for p in backbone_params)
print(f"Backbone params: {backbone_count/1e6:.1f}M (lr={LR_BACKBONE})")
print(f"Head params:     {head_count/1e6:.3f}M (lr={LR_HEAD})")
assert head_count > 0, "BUG: head params is 0 — classifier name matching failed!"

# COLLATORS
class TrainCollator:
    def __init__(self, fe, sr=16000):
        self.fe = fe; self.sr = sr
    def __call__(self, batch):
        waveforms, labels = zip(*batch)
        inputs = self.fe([w.numpy() for w in waveforms], sampling_rate=self.sr,
                         return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        return inputs["input_values"], torch.tensor(labels, dtype=torch.long)

class TestCollator:
    def __init__(self, fe, sr=16000):
        self.fe = fe; self.sr = sr
    def __call__(self, batch):
        waveforms, ids = zip(*batch)
        inputs = self.fe([w.numpy() for w in waveforms], sampling_rate=self.sr,
                         return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        return inputs["input_values"], list(ids)

collator = TrainCollator(feature_extractor, SR)
test_collator = TestCollator(feature_extractor, SR)

# MIXUP
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# TRAINING
def train_one_epoch(model, loader, optimizer, scaler, criterion, accum=ACCUM_STEPS):
    model.train()
    total_loss, n = 0, 0
    optimizer.zero_grad()
    for step, (inp, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        inp, labels = inp.to(DEVICE), labels.to(DEVICE)

        # Mixup 50% of the time
        use_mix = random.random() < 0.5
        if use_mix:
            inp, y_a, y_b, lam = mixup_data(inp, labels, MIXUP_ALPHA)

        with autocast():
            logits = model(inp)
            if use_mix:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / accum
            else:
                loss = criterion(logits, labels) / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum * len(labels)
        n += len(labels)
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for inp, labels in loader:
        inp = inp.to(DEVICE)
        with autocast():
            logits = model(inp)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, acc, np.array(all_preds), np.array(all_labels)

# SETUP & TRAIN
print("\n--- Training ---")
val_ds = ValDataset(val_songs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collator)
print(f"Val: {len(val_ds)} samples")

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LR_BACKBONE},
    {'params': head_params,     'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = GradScaler()

best_f1 = 0.0
patience_counter = 0
history = {'loss': [], 'val_f1': [], 'val_acc': [], 'lr_bb': [], 'lr_hd': []}

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_ds = MashupDataset(train_stems, noise_files, SAMPLES_PER_GENRE, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, collate_fn=collator)

    loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
    scheduler.step()
    val_f1, val_acc, vp, vl = evaluate(model, val_loader)
    lr_bb = optimizer.param_groups[0]['lr']
    lr_hd = optimizer.param_groups[1]['lr']
    elapsed = time.time() - t0

    history['loss'].append(loss)
    history['val_f1'].append(val_f1)
    history['val_acc'].append(val_acc)
    history['lr_bb'].append(lr_bb)
    history['lr_hd'].append(lr_hd)
    wandb.log({"epoch": epoch, "loss": loss, "val_f1": val_f1, "val_acc": val_acc,
               "lr_backbone": lr_bb, "lr_head": lr_hd})

    tag = ""
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_ast.pth'))
        tag = " ★"
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"E{epoch:02d}/{EPOCHS} | loss={loss:.4f} | f1={val_f1:.4f} | acc={val_acc:.4f} | "
          f"lr_bb={lr_bb:.7f} lr_hd={lr_hd:.5f} | {elapsed:.0f}s{tag}")

    if patience_counter >= 8:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"\nBest val F1: {best_f1:.4f}")

# RESULTS
print("\n--- Results ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history['loss']); axes[0].set_title('Loss')
axes[1].plot(history['val_f1'], label='F1'); axes[1].plot(history['val_acc'], label='Acc', alpha=0.7)
axes[1].set_title('Validation'); axes[1].legend()
axes[2].plot(history['lr_bb'], label='Backbone'); axes[2].plot(history['lr_hd'], label='Head')
axes[2].set_title('LR'); axes[2].legend()
plt.suptitle(f'AST v2 — Best F1: {best_f1:.4f}'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ast_v2_curves.png'), dpi=150)
wandb.log({"plots/ast_v2_curves": wandb.Image(fig)}); plt.close()

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_ast.pth'), weights_only=True))
val_f1, val_acc, preds, labels = evaluate(model, val_loader)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=GENRES).plot(ax=ax, cmap='Blues', xticks_rotation=45)
ax.set_title(f'AST v2 — F1={val_f1:.4f}'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ast_v2_confusion.png'), dpi=150)
wandb.log({"plots/ast_v2_confusion": wandb.Image(fig)}); plt.close()
print(classification_report(labels, preds, target_names=GENRES))

# INFERENCE + TTA
print("\n--- Inference ---")

@torch.no_grad()
def predict_tta(model, loader, n_tta=5):
    model.eval()
    all_ids = []
    for _, ids in loader: all_ids.extend(ids)
    all_probs = []
    for _ in range(n_tta):
        round_probs = []
        for inp, _ in loader:
            inp = inp.to(DEVICE)
            with autocast():
                logits = model(inp)
            round_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(round_probs))
    avg = np.mean(all_probs, axis=0)
    return avg.argmax(1), all_ids, avg

test_ds = TestDataset(TEST_DIR, TEST_CSV)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, collate_fn=test_collator)
print(f"Test: {len(test_ds)} samples")

preds, ids, probs = predict_tta(model, test_loader, n_tta=5)
print(f"Predictions: {Counter(preds)}")

# SUBMISSION
test_df = pd.read_csv(TEST_CSV, dtype={'id': str})
pred_dict = {str(id_): IDX2GENRE[p] for id_, p in zip(ids, preds)}
test_df['genre'] = test_df['id'].apply(lambda x: pred_dict.get(str(x), 'rock'))
test_df[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, 'submission_ast_v2.csv'), index=False)
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'submission_ast_v2.csv')}")
print(test_df['genre'].value_counts().sort_index())

np.save(os.path.join(OUTPUT_DIR, 'test_probs_ast_v2.npy'), probs)

wandb.log({"best_f1": best_f1, "status": "complete"})
art = wandb.Artifact("ast_v2", type="model")
art.add_file(os.path.join(OUTPUT_DIR, 'best_ast.pth'))
run.log_artifact(art)
wandb.finish()
print(f"\n✅ Done — best val_f1={best_f1:.4f}")
