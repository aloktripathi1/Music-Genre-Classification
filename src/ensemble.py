# Strategy:
#   1. Weighted ensemble (sweep weights)
#   2. Temperature-scaled pseudo-labeling on test set
#   3. Fine-tune AST on high-confidence pseudo-labeled test samples
#   4. Re-predict with adapted model
#   5. Final asymmetric ensemble

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

from sklearn.metrics import f1_score
from transformers import ASTFeatureExtractor, ASTForAudioClassification

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# CONFIG
DATA_ROOT  = os.path.expanduser("~/data/messy_mashup")
OUTPUT_DIR = "./outputs"
TEST_DIR   = os.path.join(DATA_ROOT, "mashups")
TEST_CSV   = os.path.join(DATA_ROOT, "test.csv")

SR         = 16000
DURATION   = 10.0
TARGET_LEN = int(SR * DURATION)

GENRES    = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE = {i: g for g, i in GENRE2IDX.items()}

# Pseudo-label config
PSEUDO_CONFIDENCE = 0.95   # only use test samples where ensemble is >95% confident
PSEUDO_TEMPERATURE = 0.5   # sharpen logits before thresholding
PSEUDO_EPOCHS = 5          # fine-tune AST on pseudo-labeled data
PSEUDO_LR = 1e-5
PSEUDO_BATCH = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# WANDB
import wandb
wandb.login(key="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")
run = wandb.init(
    entity="23f3003225-indian-institute-of-technology-madras",
    project="23f3003225-dl-genai-project",
    name="exp_005_ensemble_pseudo",
    config=dict(models=["efficientnet_b0", "ast_v1"],
                pseudo_confidence=PSEUDO_CONFIDENCE,
                pseudo_temperature=PSEUDO_TEMPERATURE,
                pseudo_epochs=PSEUDO_EPOCHS),
    tags=["ensemble", "pseudo_label", "domain_adaptation"],
    job_type="ensemble",
)

# STEP 1: Load saved probabilities
print("\n=== STEP 1: Load model probabilities ===")

cnn_path = os.path.expanduser('~/cnn/test_probs_efficientnet.npy')
ast_path = os.path.expanduser('~/ast/test_probs_ast.npy')  # v1 probs (0.927 LB)

cnn_probs = np.load(cnn_path)
ast_probs = np.load(ast_path)
print(f"CNN probs: {cnn_probs.shape}")
print(f"AST probs: {ast_probs.shape}")

test_df = pd.read_csv(TEST_CSV, dtype={'id': str})
print(f"Test samples: {len(test_df)}")

# STEP 2: Weighted ensemble sweep
print("\n=== STEP 2: Weighted ensemble sweep ===")

weights = [
    (0.2, 0.8),  # heavy AST
    (0.25, 0.75),
    (0.3, 0.7),  # recommended from top solutions
    (0.35, 0.65),
    (0.4, 0.6),
    (0.5, 0.5),  # equal
]

for w_cnn, w_ast in weights:
    ens_probs = w_cnn * cnn_probs + w_ast * ast_probs
    ens_preds = ens_probs.argmax(1)
    ens_labels = [IDX2GENRE[p] for p in ens_preds]

    # Save each
    fname = f"submission_ens_{int(w_cnn*100)}_{int(w_ast*100)}.csv"
    sub = test_df.copy()
    sub['genre'] = ens_labels
    sub[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    # Agreement rate between models
    cnn_preds = cnn_probs.argmax(1)
    ast_preds = ast_probs.argmax(1)
    agree = np.mean(cnn_preds == ast_preds)
    print(f"  w_cnn={w_cnn:.2f}, w_ast={w_ast:.2f} → {fname} | agree={agree:.3f}")

wandb.log({"ensemble/model_agreement": float(agree)})

# STEP 3: Temperature-scaled pseudo-labeling
print("\n=== STEP 3: Pseudo-labeling ===")

# Use 70/30 AST-heavy ensemble for pseudo labels
base_probs = 0.3 * cnn_probs + 0.7 * ast_probs

# Temperature scaling: sharpen predictions
# logits = log(probs) / T → softmax
base_logits = np.log(base_probs + 1e-10)
scaled_logits = base_logits / PSEUDO_TEMPERATURE
# Softmax
exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
sharp_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

# Filter: only keep samples where sharpened confidence > threshold
max_confidence = sharp_probs.max(axis=1)
pseudo_mask = max_confidence >= PSEUDO_CONFIDENCE
pseudo_labels = sharp_probs[pseudo_mask].argmax(axis=1)
pseudo_ids = test_df['id'].values[pseudo_mask]

print(f"Total test samples: {len(test_df)}")
print(f"Pseudo-labeled (conf >= {PSEUDO_CONFIDENCE}): {pseudo_mask.sum()} ({100*pseudo_mask.mean():.1f}%)")
print(f"Pseudo label distribution: {Counter(pseudo_labels)}")

# Per-class breakdown
print("\nPer-class pseudo-label counts:")
for i, g in enumerate(GENRES):
    count = (pseudo_labels == i).sum()
    avg_conf = max_confidence[pseudo_mask][pseudo_labels == i].mean() if count > 0 else 0
    print(f"  {g}: {count} samples, avg confidence: {avg_conf:.4f}")

wandb.log({
    "pseudo/total": int(pseudo_mask.sum()),
    "pseudo/percentage": float(100 * pseudo_mask.mean()),
    "pseudo/min_confidence": float(max_confidence[pseudo_mask].min()),
    "pseudo/mean_confidence": float(max_confidence[pseudo_mask].mean()),
})

# STEP 4: Fine-tune AST on pseudo-labeled test data
print("\n=== STEP 4: Fine-tune AST on pseudo-labeled data ===")

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


class PseudoDataset(Dataset):
    """Test samples with pseudo-labels for domain adaptation."""
    def __init__(self, test_dir, ids, labels):
        self.ids = ids
        self.labels = labels
        self.test_dir = test_dir
        self.paths = []
        for id_ in ids:
            path = None
            for pat in [f"song{str(id_).zfill(4)}.wav", f"{id_}.wav", f"song{id_}.wav"]:
                p = os.path.join(test_dir, pat)
                if os.path.exists(p): path = p; break
            self.paths.append(path)

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wav = load_wav(path) if path else torch.zeros(TARGET_LEN)
        return wav, int(self.labels[idx])


class PseudoCollator:
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


# Load AST v1 (the better model)
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

# Load best AST v1 weights
ast_weights_path = os.path.expanduser('~/ast/best_ast.pth')
if os.path.exists(ast_weights_path):
    model.load_state_dict(torch.load(ast_weights_path, weights_only=True))
    print(f"Loaded AST v1 weights from {ast_weights_path}")
else:
    print("WARNING: AST weights not found — using fresh model")

# Create pseudo dataset
pseudo_ds = PseudoDataset(TEST_DIR, pseudo_ids, pseudo_labels)
pseudo_collator = PseudoCollator(feature_extractor, SR)
pseudo_loader = DataLoader(pseudo_ds, batch_size=PSEUDO_BATCH, shuffle=True,
                           num_workers=4, pin_memory=True, collate_fn=pseudo_collator,
                           drop_last=True)
print(f"Pseudo dataset: {len(pseudo_ds)} samples")

# Fine-tune with low LR (domain adaptation)
optimizer = torch.optim.AdamW(model.parameters(), lr=PSEUDO_LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler = GradScaler()

print(f"\nFine-tuning AST on pseudo-labeled data ({PSEUDO_EPOCHS} epochs)...")
for epoch in range(1, PSEUDO_EPOCHS + 1):
    model.train()
    total_loss, n = 0, 0
    for inp, labels in tqdm(pseudo_loader, desc=f"Pseudo E{epoch}", leave=False):
        inp, labels = inp.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(inp)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    print(f"  Pseudo E{epoch}: loss={total_loss/n:.4f}")

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'ast_pseudo_adapted.pth'))
print("AST pseudo-adapted model saved.")

# STEP 5: Re-predict with adapted AST + TTA
print("\n=== STEP 5: Re-predict with adapted AST (3-way TTA) ===")

test_collator = TestCollator(feature_extractor, SR)
test_ds = TestDataset(TEST_DIR, TEST_CSV)
test_loader = DataLoader(test_ds, batch_size=PSEUDO_BATCH, shuffle=False,
                         num_workers=4, pin_memory=True, collate_fn=test_collator)

@torch.no_grad()
def predict_with_shift_tta(model, loader, shifts=[0, 80, -80]):
    # 3-way TTA: center + forward shift + backward shift
    model.eval()
    all_ids = []
    for _, ids in loader:
        all_ids.extend(ids)

    all_probs = []
    for shift in shifts:
        round_probs = []
        for inp, _ in loader:
            inp = inp.to(DEVICE)
            if shift != 0:
                inp = torch.roll(inp, shifts=shift, dims=-1)
            with autocast():
                logits = model(inp)
            round_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(round_probs))

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs.argmax(1), all_ids, avg_probs

preds_adapted, ids, ast_adapted_probs = predict_with_shift_tta(model, test_loader)
print(f"Adapted AST predictions: {Counter(preds_adapted)}")

np.save(os.path.join(OUTPUT_DIR, 'test_probs_ast_adapted.npy'), ast_adapted_probs)

# STEP 6: Final ensemble (CNN + AST_adapted)
print("\n=== STEP 6: Final ensemble ===")

# Try multiple weights with adapted AST
final_weights = [
    (0.2, 0.8),
    (0.25, 0.75),
    (0.3, 0.7),   # recommended
    (0.35, 0.65),
    (0.4, 0.6),
]

for w_cnn, w_ast in final_weights:
    final_probs = w_cnn * cnn_probs + w_ast * ast_adapted_probs
    final_preds = final_probs.argmax(1)

    fname = f"submission_final_{int(w_cnn*100)}_{int(w_ast*100)}.csv"
    sub = test_df.copy()
    sub['genre'] = [IDX2GENRE[p] for p in final_preds]
    sub[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  w_cnn={w_cnn}, w_ast_adapted={w_ast} → {fname}")

# Also try 3-model ensemble: CNN + AST_v1 + AST_adapted
print("\n3-model ensemble (CNN + AST_v1 + AST_adapted):")
three_weights = [
    (0.2, 0.4, 0.4),
    (0.25, 0.35, 0.4),
    (0.3, 0.35, 0.35),
    (0.2, 0.3, 0.5),
]

for w_c, w_a1, w_a2 in three_weights:
    final3 = w_c * cnn_probs + w_a1 * ast_probs + w_a2 * ast_adapted_probs
    preds3 = final3.argmax(1)

    fname = f"submission_3model_{int(w_c*100)}_{int(w_a1*100)}_{int(w_a2*100)}.csv"
    sub = test_df.copy()
    sub['genre'] = [IDX2GENRE[p] for p in preds3]
    sub[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  w_cnn={w_c}, w_ast_v1={w_a1}, w_ast_adapted={w_a2} → {fname}")

# STEP 7: Summary
print("\nBaseline ensembles (CNN + AST v1):")
for w_cnn, w_ast in weights:
    print(f"  submission_ens_{int(w_cnn*100)}_{int(w_ast*100)}.csv")

print("\nAdapted ensembles (CNN + AST pseudo-adapted):")
for w_cnn, w_ast in final_weights:
    print(f"  submission_final_{int(w_cnn*100)}_{int(w_ast*100)}.csv")

print("\n3-model ensembles:")
for w_c, w_a1, w_a2 in three_weights:
    print(f"  submission_3model_{int(w_c*100)}_{int(w_a1*100)}_{int(w_a2*100)}.csv")

wandb.log({"ensemble/status": "complete",
           "pseudo/samples_used": int(pseudo_mask.sum())})
wandb.finish()
