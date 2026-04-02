# 3× AST with different seeds + full ensemble
# Replicates EXACT v1 config with 3 seeds for diversity.
# Then ensembles with existing CNN + ResNet probs.

import os, glob, random, warnings, time, gc
import numpy as np, pandas as pd
from collections import Counter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import librosa

from sklearn.metrics import f1_score, classification_report
from transformers import ASTFeatureExtractor, ASTForAudioClassification

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════
# CONFIG — exact v1 replica
# ═══════════════════════════════════════════
DATA_ROOT  = os.path.expanduser("~/data/messy_mashup")
OUTPUT_DIR = "./outputs_final"
STEMS_DIR  = os.path.join(DATA_ROOT, "genres_stems")
NOISE_DIR  = os.path.join(DATA_ROOT, "ESC-50-master", "audio")
TEST_DIR   = os.path.join(DATA_ROOT, "mashups")
TEST_CSV   = os.path.join(DATA_ROOT, "test.csv")

# Paths to existing probs (from previous runs)
CNN_PROBS  = os.path.expanduser("~/cnn/test_probs_efficientnet.npy")
AST_V1_PROBS = os.path.expanduser("~/ast/test_probs_ast.npy")
RESNET_PROBS = os.path.expanduser("~/resnet50/test_probs_resnet50.npy")

SR          = 16000
DURATION    = 10.0
TARGET_LEN  = int(SR * DURATION)
GENRES      = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX   = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE   = {i: g for g, i in GENRE2IDX.items()}
STEMS       = ['drums', 'vocals', 'bass']
STEM_WEIGHTS = {'drums': 0.45, 'vocals': 0.35, 'bass': 0.20}

# v1 exact config
SAMPLES_PER_GENRE = 800
BATCH_SIZE        = 8
ACCUM_STEPS       = 4
EPOCHS            = 20
LR                = 1e-5   # uniform LR — this is what v1 actually used (the "bug" that worked)
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.1
GRAD_CLIP         = 1.0
NUM_WORKERS       = 4
WARMUP_EPOCHS     = 2

# 3 different seeds for diversity
SEEDS = [42, 123, 777]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Plan: Train 3 ASTs with seeds {SEEDS}, {SAMPLES_PER_GENRE*10} mashups/epoch, {EPOCHS} epochs each")
print(f"Estimated: ~80 min/model × 3 = ~4 hours total")

# ═══════════════════════════════════════════
# DATA INDEX (fixed across all seeds)
# ═══════════════════════════════════════════
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
print(f"Noise: {len(noise_files)}")
for g in GENRES:
    print(f"  {g}: {len(song_index[g])} songs")

# ═══════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════
def load_wav(path, sr=SR, target_len=TARGET_LEN):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        wav = torch.from_numpy(y).float()
        if len(wav) < target_len: wav = F.pad(wav, (0, target_len - len(wav)))
        elif len(wav) > target_len:
            start = random.randint(0, len(wav) - target_len)
            wav = wav[start:start + target_len]
        return wav
    except:
        return torch.zeros(target_len)

# ═══════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════
class MashupDataset(Dataset):
    def __init__(self, stem_idx, noise_files, samples_per_genre=800, augment=True):
        self.stem_idx = stem_idx
        self.noise_files = noise_files
        self.augment = augment
        self.samples = []
        for genre in GENRES:
            for _ in range(samples_per_genre):
                self.samples.append(GENRE2IDX[genre])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        gi = self.samples[idx]
        g = IDX2GENRE[gi]
        stems_wav = []
        for st in STEMS:
            avail = self.stem_idx[g][st]
            if not avail: continue
            wav = load_wav(random.choice(avail))
            gain = random.uniform(0.5, 1.5) * (STEM_WEIGHTS[st] / 0.33)
            stems_wav.append(wav * gain)
        if not stems_wav: return torch.zeros(TARGET_LEN), gi
        mix = torch.stack(stems_wav).sum(0)
        if self.augment:
            mix = torch.roll(mix, random.randint(-SR, SR))
            for _ in range(random.randint(0, 2)):
                noise = load_wav(random.choice(self.noise_files))
                snr = random.uniform(5.0, 25.0)
                sp = mix.pow(2).mean() + 1e-10
                np_ = noise.pow(2).mean() + 1e-10
                mix = mix + noise * (sp / (np_ * 10**(snr/10))).sqrt()
            if random.random() < 0.3:
                mix = torch.clamp(mix * random.uniform(1.2, 3.0), -1, 1)
        peak = mix.abs().max()
        if peak > 1e-6: mix = mix / peak * random.uniform(0.7, 1.0)
        return mix, gi

class ValDataset(Dataset):
    def __init__(self, song_index):
        self.items = []
        for g in GENRES:
            for s in song_index[g]: self.items.append((s, GENRE2IDX[g]))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        si, label = self.items[idx]
        stems = [load_wav(os.path.join(si['dir'], f"{st}.wav")) for st in si['stems']]
        mix = torch.stack(stems).sum(0)
        pk = mix.abs().max()
        if pk > 1e-6: mix = mix / pk
        return mix, label

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
        wav = load_wav(p) if p else torch.zeros(TARGET_LEN)
        return wav, str(self.df.iloc[idx]['id'])

# ═══════════════════════════════════════════
# COLLATORS
# ═══════════════════════════════════════════
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL)

class Collator:
    def __init__(self, fe, sr=16000):
        self.fe = fe; self.sr = sr
    def __call__(self, batch):
        waveforms, labels = zip(*batch)
        inputs = self.fe([w.numpy() for w in waveforms], sampling_rate=self.sr,
                         return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        if isinstance(labels[0], (int, np.integer)):
            return inputs["input_values"], torch.tensor(labels, dtype=torch.long)
        return inputs["input_values"], list(labels)

collator = Collator(feature_extractor, SR)

# ═══════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════
class ASTGenreClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            AST_MODEL, num_labels=10, ignore_mismatched_sizes=True)
    def forward(self, x):
        return self.ast(input_values=x).logits

def train_one_epoch(model, loader, optimizer, scaler, criterion, accum=ACCUM_STEPS):
    model.train()
    total_loss, n = 0, 0
    optimizer.zero_grad()
    for step, (inp, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        inp, labels = inp.to(DEVICE), labels.to(DEVICE)
        with autocast():
            loss = criterion(model(inp), labels) / accum
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
    preds, labels = [], []
    for inp, lab in loader:
        inp = inp.to(DEVICE)
        with autocast():
            logits = model(inp)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(lab.numpy())
    return f1_score(labels, preds, average='macro')

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs, all_ids = [], []
    for inp, ids in loader:
        inp = inp.to(DEVICE)
        with autocast():
            logits = model(inp)
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_ids.extend(ids)
    return np.vstack(all_probs), all_ids

def train_ast_with_seed(seed):
    """Train one AST model with a specific seed. Returns test probs."""
    print(f"\n{'='*60}")
    print(f"TRAINING AST — seed={seed}")
    print(f"{'='*60}")

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # Different train/val split per seed (key for diversity!)
    train_stems_local = {g: {st: [] for st in STEMS} for g in GENRES}
    val_songs_local = {g: [] for g in GENRES}
    for genre in GENRES:
        songs = song_index[genre].copy()
        random.shuffle(songs)
        split = int(0.85 * len(songs))
        train_list, val_list = songs[:split], songs[split:]
        val_songs_local[genre] = val_list
        train_dirs = {s['dir'] for s in train_list}
        for st in STEMS:
            train_stems_local[genre][st] = [fp for fp in stem_index[genre][st]
                                             if os.path.dirname(fp) in train_dirs]

    val_ds = ValDataset(val_songs_local)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collator)

    model = ASTGenreClassifier().to(DEVICE)

    # v1 config: uniform LR for all params (the "bug" that scored 0.927)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler()

    best_f1 = 0.0
    patience = 0
    save_path = os.path.join(OUTPUT_DIR, f'best_ast_seed{seed}.pth')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_ds = MashupDataset(train_stems_local, noise_files, SAMPLES_PER_GENRE, augment=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  drop_last=True, collate_fn=collator)
        loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        scheduler.step()
        val_f1 = evaluate(model, val_loader)
        elapsed = time.time() - t0

        tag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            tag = " ★"
            patience = 0
        else:
            patience += 1

        print(f"  E{epoch:02d}/{EPOCHS} | loss={loss:.4f} | f1={val_f1:.4f} | {elapsed:.0f}s{tag}")

        if patience >= 6:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"  Best val F1: {best_f1:.4f}")

    # Predict test
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_ds = TestDataset()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collator)
    probs, ids = predict(model, test_loader)
    np.save(os.path.join(OUTPUT_DIR, f'test_probs_ast_seed{seed}.npy'), probs)
    print(f"  Saved probs: test_probs_ast_seed{seed}.npy")

    del model; gc.collect(); torch.cuda.empty_cache()
    return probs, ids, best_f1

# ═══════════════════════════════════════════
# TRAIN 3 ASTs
# ═══════════════════════════════════════════
all_ast_probs = {}
all_ast_f1s = {}
test_ids = None

for seed in SEEDS:
    probs, ids, f1 = train_ast_with_seed(seed)
    all_ast_probs[seed] = probs
    all_ast_f1s[seed] = f1
    if test_ids is None:
        test_ids = ids

print(f"\n{'='*60}")
print(f"ALL AST MODELS TRAINED")
print(f"{'='*60}")
for seed, f1 in all_ast_f1s.items():
    print(f"  Seed {seed}: val F1 = {f1:.4f}")

# Average of 3 ASTs
avg_3ast = np.mean(list(all_ast_probs.values()), axis=0)
print(f"  3-AST avg predictions: {Counter(avg_3ast.argmax(1))}")

# ═══════════════════════════════════════════
# ENSEMBLE WITH EXISTING PROBS
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print(f"ENSEMBLE")
print(f"{'='*60}")

test_df = pd.read_csv(TEST_CSV, dtype={'id': str})

# Load existing probs
existing = {}
for name, path in [('ast_v1', AST_V1_PROBS), ('cnn', CNN_PROBS), ('resnet', RESNET_PROBS)]:
    if os.path.exists(path):
        existing[name] = np.load(path)
        print(f"  Loaded {name}: {path}")
    else:
        print(f"  NOT FOUND: {name} ({path})")

def save_submission(probs, fname):
    preds = probs.argmax(1)
    sub = test_df.copy()
    pred_dict = {str(id_): IDX2GENRE[p] for id_, p in zip(test_ids, preds)}
    sub['genre'] = sub['id'].apply(lambda x: pred_dict.get(str(x), 'rock'))
    sub[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

# ─── 3-AST standalone ───
print("\n--- 3-AST Ensemble (new models only) ---")
save_submission(avg_3ast, "submission_3ast_avg.csv")
print("  submission_3ast_avg.csv")

# ─── 3-AST + AST v1 (4 ASTs total) ───
if 'ast_v1' in existing:
    print("\n--- 4-AST Ensemble (3 new + v1) ---")
    for w_new, w_v1 in [(0.75, 0.25), (0.60, 0.40), (0.50, 0.50), (0.40, 0.60)]:
        ens = w_new * avg_3ast + w_v1 * existing['ast_v1']
        fname = f"submission_4ast_new{int(w_new*100)}_v1{int(w_v1*100)}.csv"
        save_submission(ens, fname)
        print(f"  new={w_new} v1={w_v1} → {fname}")

# ─── Full ensemble: 4 ASTs + CNN + ResNet ───
if all(k in existing for k in ['ast_v1', 'cnn', 'resnet']):
    print("\n--- Full Ensemble (4 ASTs + CNN + ResNet) ---")
    combos = [
        # (3ast_new, ast_v1, cnn, resnet)
        (0.30, 0.40, 0.10, 0.20),  # v1 heavy
        (0.35, 0.35, 0.10, 0.20),  # balanced ASTs
        (0.25, 0.45, 0.10, 0.20),  # v1 dominant
        (0.30, 0.40, 0.05, 0.25),  # more resnet
        (0.35, 0.35, 0.05, 0.25),
        (0.20, 0.50, 0.10, 0.20),  # max v1
        (0.30, 0.35, 0.15, 0.20),  # more cnn
        (0.25, 0.40, 0.10, 0.25),
        (0.35, 0.30, 0.10, 0.25),
        (0.30, 0.30, 0.10, 0.30),  # more resnet
    ]
    for w_new, w_v1, w_c, w_r in combos:
        ens = w_new * avg_3ast + w_v1 * existing['ast_v1'] + w_c * existing['cnn'] + w_r * existing['resnet']
        fname = f"submission_full_{int(w_new*100)}_{int(w_v1*100)}_{int(w_c*100)}_{int(w_r*100)}.csv"
        save_submission(ens, fname)
        print(f"  3ast={w_new} v1={w_v1} cnn={w_c} res={w_r} → {fname}")

    # Also try individual new ASTs + v1 + CNN + ResNet
    print("\n--- Per-seed AST + v1 + CNN + ResNet ---")
    for seed in SEEDS:
        for w_s, w_v1, w_c, w_r in [(0.30, 0.40, 0.10, 0.20), (0.35, 0.35, 0.10, 0.20)]:
            ens = w_s * all_ast_probs[seed] + w_v1 * existing['ast_v1'] + w_c * existing['cnn'] + w_r * existing['resnet']
            fname = f"submission_s{seed}_{int(w_s*100)}_{int(w_v1*100)}_{int(w_c*100)}_{int(w_r*100)}.csv"
            save_submission(ens, fname)
            print(f"  seed{seed}={w_s} v1={w_v1} cnn={w_c} res={w_r} → {fname}")

# ═══════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
for seed, f1 in all_ast_f1s.items():
    print(f"  AST seed {seed}: val F1 = {f1:.4f}")
print(f"\nAll submissions:")
for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'submission_*.csv'))):
    print(f"  {os.path.basename(f)}")
