# Train + Ensemble
# Loads precomputed .pt spectrograms, trains ResNet50 + EfficientNet-B0, ensembles with AST.

# Input datasets:
#   1. Competition data: /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/
#   2. Precomputed specs: /kaggle/input/mashup-specs-25k/mashup_specs/ (from Notebook 1)
#   3. AST probs: /kaggle/input/ast-probs/test_probs_ast.npy (upload from Lightning.ai)


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
import timm

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# CONFIG
# UPDATE THESE PATHS to match your Kaggle dataset names
SPECS_DIR  = '/kaggle/input/mashup-specs-25k/mashup_specs'   # from Notebook 1
AST_PROBS  = '/kaggle/input/ast-probs/test_probs_ast.npy'    # uploaded from Lightning.ai
TEST_CSV   = '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv'
OUTPUT_DIR = '/kaggle/working'

GENRES    = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE = {i: g for g, i in GENRE2IDX.items()}

BATCH_SIZE      = 128     # precomputed = fast I/O → large batches
EPOCHS          = 40
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP       = 1.0
NUM_WORKERS     = 2
MIXUP_ALPHA     = 0.4

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Config: bs={BATCH_SIZE}, epochs={EPOCHS}")

# WANDB
os.system('pip install wandb --quiet')
import wandb
wandb.login(key="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")

# DATASET — loads precomputed .pt files (FAST)
class MelDataset(Dataset):
    def __init__(self, split='train'):
        self.files = sorted(glob.glob(os.path.join(SPECS_DIR, split, '*.pt')))
        print(f"  {split}: {len(self.files)} files")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        mel = data['mel']  # already normalized during precompute
        mel = mel.unsqueeze(0)  # (1, n_mels, time)
        return mel, data['label']


class TestMelDataset(Dataset):
    def __init__(self):
        self.files = sorted(glob.glob(os.path.join(SPECS_DIR, 'test', '*.pt')))
        print(f"  test: {len(self.files)} files")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        mel = data['mel']
        mel = mel.unsqueeze(0)
        return mel, str(data['id'])


train_ds = MelDataset('train')
val_ds   = MelDataset('val')
test_ds  = TestMelDataset()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

# MODEL COMPONENTS
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


def build_model(backbone_name, num_classes=10):
    backbone = timm.create_model(backbone_name, pretrained=True,
                                  in_chans=1, num_classes=0, global_pool='')
    nf = backbone.num_features
    model = nn.Sequential(
        backbone,
        GeM(p=3.0),
        nn.LayerNorm(nf),
        nn.Dropout(0.4),
        nn.Linear(nf, num_classes)
    )
    return model

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

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs, all_ids = [], []
    for mel, ids in loader:
        mel = mel.to(DEVICE)
        with autocast():
            logits = model(mel)
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_ids.extend(ids)
    return np.vstack(all_probs), all_ids


def train_model(backbone_name, run_name):
    """Full training pipeline for one backbone. Returns test probabilities."""
    print(f"\n{'='*60}")
    print(f"Training {backbone_name}")
    print(f"{'='*60}")

    wandb_run = wandb.init(
        entity="23f3003225-indian-institute-of-technology-madras",
        project="23f3003225-dl-genai-project",
        name=run_name,
        config=dict(backbone=backbone_name, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    lr=LR, mixup=MIXUP_ALPHA, label_smoothing=LABEL_SMOOTHING),
        tags=[backbone_name.replace('_', ''), "precomputed", "kaggle"],
        job_type="train",
        reinit=True,
    )

    model = build_model(backbone_name).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler()

    best_f1 = 0.0
    patience = 0
    save_path = os.path.join(OUTPUT_DIR, f'best_{backbone_name}.pth')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        scheduler.step()
        val_f1, val_acc, vp, vl = evaluate(model, val_loader)
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        wandb.log({"epoch": epoch, "loss": loss, "val_f1": val_f1, "val_acc": val_acc, "lr": lr})

        tag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            tag = " ★"
            patience = 0
        else:
            patience += 1

        print(f"  E{epoch:02d}/{EPOCHS} | loss={loss:.4f} | f1={val_f1:.4f} | acc={val_acc:.4f} | {elapsed:.0f}s{tag}")

        if patience >= 8:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"  Best val F1: {best_f1:.4f}")

    # Load best and evaluate
    model.load_state_dict(torch.load(save_path, weights_only=True))
    val_f1, val_acc, preds, labels = evaluate(model, val_loader)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=GENRES).plot(
        ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title(f'{backbone_name} — F1={val_f1:.4f}')
    plt.tight_layout()
    wandb.log({"plots/confusion": wandb.Image(fig)})
    plt.savefig(os.path.join(OUTPUT_DIR, f'{backbone_name}_confusion.png'), dpi=150)
    plt.close()

    print(classification_report(labels, preds, target_names=GENRES))

    # Predict test
    test_probs, test_ids = predict(model, test_loader)
    np.save(os.path.join(OUTPUT_DIR, f'test_probs_{backbone_name}.npy'), test_probs)

    # Standalone submission
    test_df = pd.read_csv(TEST_CSV, dtype={'id': str})
    pred_dict = {str(id_): IDX2GENRE[p] for id_, p in zip(test_ids, test_probs.argmax(1))}
    test_df['genre'] = test_df['id'].apply(lambda x: pred_dict.get(str(x), 'rock'))
    test_df[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, f'submission_{backbone_name}.csv'), index=False)
    print(f"  Saved: submission_{backbone_name}.csv + test_probs_{backbone_name}.npy")

    wandb.log({"best_f1": best_f1})
    wandb.finish()

    del model; gc.collect(); torch.cuda.empty_cache()
    return test_probs, test_ids, best_f1

# TRAIN BOTH MODELS
resnet_probs, test_ids, resnet_f1 = train_model('resnet50', 'exp_007_resnet50_precomputed')
cnn_probs, _, cnn_f1 = train_model('efficientnet_b0', 'exp_008_effnet_precomputed')

# ENSEMBLE
print(f"\n{'='*60}")
print(f"ENSEMBLE")
print(f"{'='*60}")

wandb_run = wandb.init(
    entity="23f3003225-indian-institute-of-technology-madras",
    project="23f3003225-dl-genai-project",
    name="exp_009_final_ensemble",
    config=dict(models=["resnet50", "efficientnet_b0", "ast_v1"]),
    tags=["ensemble", "final"], job_type="ensemble", reinit=True,
)

# Load AST probs if available
ast_probs = None
if os.path.exists(AST_PROBS):
    ast_probs = np.load(AST_PROBS)
    print(f"AST probs loaded: {ast_probs.shape}")
else:
    print(f"AST probs NOT found at {AST_PROBS}")
    print("Available files:")
    for d in glob.glob('/kaggle/input/*/'):
        print(f"  {d}: {os.listdir(d)[:5]}")

test_df = pd.read_csv(TEST_CSV, dtype={'id': str})

# ─── 2-model ensemble (ResNet + EfficientNet) ───
print("\n--- 2-Model Ensemble (ResNet + EfficientNet) ---")
for w_r, w_c in [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6)]:
    ens = w_r * resnet_probs + w_c * cnn_probs
    preds = ens.argmax(1)
    fname = f"submission_2way_R{int(w_r*100)}_C{int(w_c*100)}.csv"
    sub = test_df.copy()
    sub['genre'] = [IDX2GENRE[p] for p in preds]
    sub[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  ResNet={w_r} CNN={w_c} → {fname}")

# ─── 3-model ensemble (ResNet + EfficientNet + AST) ───
if ast_probs is not None:
    print("\n--- 3-Model Ensemble (ResNet + EfficientNet + AST) ---")
    weight_combos = [
        # (cnn, ast, resnet)
        (0.10, 0.60, 0.30),  # current best-ish from Lightning.ai
        (0.10, 0.55, 0.35),
        (0.15, 0.55, 0.30),
        (0.15, 0.50, 0.35),
        (0.10, 0.50, 0.40),
        (0.20, 0.50, 0.30),
        (0.10, 0.45, 0.45),
        (0.05, 0.55, 0.40),
        (0.05, 0.50, 0.45),
        (0.10, 0.40, 0.50),  # ResNet heavy
        (0.15, 0.45, 0.40),
        (0.05, 0.60, 0.35),
    ]

    for w_c, w_a, w_r in weight_combos:
        ens = w_c * cnn_probs + w_a * ast_probs + w_r * resnet_probs
        preds = ens.argmax(1)
        fname = f"submission_3way_C{int(w_c*100)}_A{int(w_a*100)}_R{int(w_r*100)}.csv"
        sub = test_df.copy()
        sub['genre'] = [IDX2GENRE[p] for p in preds]
        sub[['id', 'genre']].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print(f"  CNN={w_c} AST={w_a} ResNet={w_r} → {fname}")

    print(f"\nRecommended first submissions:")
    print(f"  1. submission_3way_C10_A55_R35.csv")
    print(f"  2. submission_3way_C10_A50_R40.csv")
    print(f"  3. submission_3way_C05_A50_R45.csv")

# ─── Summary ───
print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"ResNet-50 val F1:      {resnet_f1:.4f}")
print(f"EfficientNet-B0 val F1: {cnn_f1:.4f}")
print(f"\nAll submissions in: {OUTPUT_DIR}")
for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'submission_*.csv'))):
    print(f"  {os.path.basename(f)}")

wandb.log({"resnet_f1": resnet_f1, "cnn_f1": cnn_f1, "status": "complete"})
wandb.finish()
