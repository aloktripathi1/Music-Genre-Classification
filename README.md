# 🎵 Messy Mashup — Music Genre Classification from Noisy Mashups

> **Competition**: Jan 2026 DLGenAI Project — Messy Mashup  
> **Task**: Predict the correct genre from noisy music mashups  
> **Metric**: Macro F1 Score across 10 genres  
> **Best Score**: 0.92350 (Public Leaderboard)

---

## 📋 Table of Contents

- [Problem Statement](repo-zip/messy-mashup-repo/README.md#problem-statement)
- [Approach Overview](repo-zip/messy-mashup-repo/README.md#approach-overview)
- [Repository Structure](repo-zip/messy-mashup-repo/README.md#repository-structure)
- [Setup & Installation](repo-zip/messy-mashup-repo/README.md#setup--installation)
- [Data Description](repo-zip/messy-mashup-repo/README.md#data-description)
- [Methodology](repo-zip/messy-mashup-repo/README.md#methodology)
- [Experiments & Results](repo-zip/messy-mashup-repo/README.md#experiments--results)
- [Key Insights](repo-zip/messy-mashup-repo/README.md#key-insights)
- [How to Reproduce](repo-zip/messy-mashup-repo/README.md#how-to-reproduce)
- [References](repo-zip/messy-mashup-repo/README.md#references)

---

## 🎯 Problem Statement

The Messy Mashup competition challenges participants to classify music genre from **noisy mashups** — audio created by:
1. Selecting instrument stems (drums, vocals, bass, others) from **different songs** of the **same genre**
2. **Tempo-adjusting** stems for rhythmic synchronization
3. **Mixing** the stems together
4. Adding **ESC-50 environmental noise** at random positions and intensities

The core challenge: training data consists of **clean, separated stems** (100 songs × 4 stems × 10 genres), but test data consists of **3,020 noisy mashups**. This creates a significant **train-test distribution mismatch**.

---

## 🏗️ Approach Overview

```
┌─────────────────────────────────────────────────────┐
│                TRAINING PIPELINE                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ Clean Stems  │───▶│  Synthetic   │               │
│  │ (4 per song) │    │  Mashup Gen  │               │
│  └──────────────┘    └──────┬───────┘               │
│                             │                        │
│  ┌──────────────┐           ▼                        │
│  │  ESC-50      │───▶ Noisy Mashup                  │
│  │  Noise Clips │    (matches test distribution)     │
│  └──────────────┘           │                        │
│                             ▼                        │
│              ┌──────────────────────────┐            │
│              │   EfficientNet-B0 (CNN)  │            │
│              │   AST (Transformer)      │            │
│              └──────────┬───────────────┘            │
│                         │                            │
│                         ▼                            │
│              ┌──────────────────────────┐            │
│              │  Pseudo-Labeling on      │            │
│              │  Test Mashups (PL)       │            │
│              └──────────┬───────────────┘            │
│                         │                            │
│                         ▼                            │
│              ┌──────────────────────────┐            │
│              │  AST Domain Adaptation   │            │
│              │  (Synthetic + PL Data)   │            │
│              └──────────────────────────┘            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                INFERENCE PIPELINE                     │
│                                                      │
│  Test Mashup ──▶ CNN (1 pass) ──┐                   │
│                                  ├──▶ 0.3/0.7 ──▶ Genre
│  Test Mashup ──▶ AST (5-way TTA)┘    Ensemble       │
└─────────────────────────────────────────────────────┘
```

**Key Insight**: The competition is fundamentally a **domain adaptation problem** disguised as classification. Architecture matters less than **how well synthetic training data matches real test distribution**.

---

## 📁 Repository Structure

```
messy-mashup/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_baseline_cnn_ast.ipynb      # V4 baseline (0.923 LB)
│   ├── 03_experiments.ipynb           # Ablation studies & experiments
│   └── 04_final_submission.ipynb      # Final submission notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # All hyperparameters & paths
│   ├── audio_utils.py                 # Audio loading, resampling
│   ├── data_indexer.py                # Stem & noise file indexing
│   ├── datasets.py                    # MashupDataset, InferenceDataset, PseudoLabelDataset
│   ├── models.py                      # SOTAAudioCNN (EfficientNet-B0 + GeM), AST wrapper
│   ├── training.py                    # Training loop with AMP, scheduling
│   ├── pseudo_labeling.py             # Temperature-scaled ensemble PL extraction
│   ├── inference.py                   # TTA inference pipeline
│   └── utils.py                       # Seeding, metrics, helpers
│
├── scripts/
│   ├── train_kaggle.py                # Single-file Kaggle notebook script
│   ├── train_lightning.py             # Lightning.ai optimized script
│   └── generate_submission.py         # Inference-only from saved checkpoints
│
├── configs/
│   ├── v4_baseline.yaml               # V4 config (0.923)
│   ├── v9_large_data.yaml             # 1500 samples/genre config
│   └── v11_distribution_aligned.yaml  # Distribution-aligned augmentation
│
├── docs/
│   ├── approach.md                    # Detailed methodology writeup
│   ├── experiments.md                 # All experiment logs & analysis
│   ├── competition_insights.md        # Topper insights & strategies
│   └── figures/
│       ├── pipeline_diagram.png
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       └── genre_distribution.png
│
├── results/
│   ├── submissions/
│   │   ├── v4_baseline_0.923.csv
│   │   └── best_submission.csv
│   └── logs/
│       ├── v4_training.log
│       └── experiment_tracker.csv
│
└── analysis/
    ├── error_analysis.ipynb           # Per-genre error breakdown
    └── leaderboard_tracker.md         # Score progression
```

---

## ⚙️ Setup & Installation

### Kaggle (T4 x2)
```bash
# Dependencies are auto-installed in the notebook
pip install -q transformers librosa torchaudio timm
```

### Lightning.ai (L40S)
```bash
pip install -r requirements.txt
python scripts/train_lightning.py
```

### Local
```bash
git clone https://github.com/YOUR_USERNAME/messy-mashup.git
cd messy-mashup
pip install -r requirements.txt
```

---

## 📊 Data Description

| Directory | Contents | Size |
|-----------|----------|------|
| `genres_stems/` | 10 genres × 100 songs × 4 stems | ~25 GB |
| `ESC-50-master/` | 2000 environmental noise clips (50 classes) | ~600 MB |
| `mashups/` | 3020 unlabeled test mashups | ~2 GB |
| `test.csv` | Test sample IDs | 78 KB |

### Genres (10 classes, balanced)
`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

### Stems per Song
`drums.wav`, `vocals.wav`, `bass.wav`, `others.wav`

---

## 🔬 Methodology

### 1. Synthetic Mashup Generation (Most Important)

The single most impactful strategy. We replicate the test data creation process during training:

```python
# For each training sample:
# 1. Pick 4 DIFFERENT songs from the same genre
# 2. Take one stem type from each (drums from song A, vocals from song B, etc.)
# 3. Apply random gain [0.5, 1.5] per stem
# 4. Sum all 4 stems
# 5. Add ESC-50 noise (1-3 clips, SNR 5-25 dB)
```

This directly closes the train-test distribution gap because the model sees mashup-like inputs during training.

### 2. Noise Augmentation (Topper-Informed)

Based on leaderboard topper's strategy:
- **Noise count**: 1-3 random ESC-50 clips per mashup
- **Intensity**: Random SNR between 5-25 dB per clip
- **Position**: 40% chance noise placed at random temporal position (not full overlay)
- Uses all 2000 ESC-50 clips across 50 environmental sound categories

### 3. Model Architecture

**CNN Branch: EfficientNet-B0 + GeM Pooling**
- Input: 128-band log-mel spectrogram (16kHz, n_fft=1024, hop=320)
- Backbone: `tf_efficientnet_b0.ns_jft_in1k` (ImageNet pretrained)
- Pooling: Generalized Mean (GeM) pooling (learnable p parameter)
- Head: LayerNorm → Dropout(0.5) → Linear(1280, 10)

**AST Branch: Audio Spectrogram Transformer**
- Pretrained: `MIT/ast-finetuned-audioset-10-10-0.4593` (AudioSet, 95.6% ESC-50)
- Architecture: ViT-B/16 on 128-dim mel-spectrogram patches
- Fine-tuned classifier head: 768 → 10 classes

### 4. Pseudo-Labeling for Domain Adaptation

```
Step 1: Train CNN + AST on synthetic mashups
Step 2: Ensemble predicts test mashups (0.3 CNN + 0.7 AST)
Step 3: Apply temperature scaling (T=0.5) to sharpen probabilities
Step 4: Keep predictions with confidence > 0.95
Step 5: Fine-tune AST on synthetic + pseudo-labeled real test data
```

This bridges the remaining synthetic→real distribution gap by letting the model see actual test-domain audio.

### 5. Inference with TTA

- **AST**: 5-way Test-Time Augmentation (temporal shifts: [0, ±80, ±160])
- **CNN**: Single forward pass
- **Ensemble**: 0.3 × CNN + 0.7 × AST (softmax-level averaging)

### 6. Training Details

| Hyperparameter | CNN | AST Base | AST Final |
|---------------|-----|----------|-----------|
| Learning Rate | 1e-3 | 2e-5 | 1e-5 |
| Optimizer | AdamW | AdamW | AdamW |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealing | Cosine Warmup | Cosine Warmup |
| Label Smoothing | 0.1 | 0.1 | 0.1 |
| Batch Size | 16 | 16 | 16 |
| Epochs | 18 | 8 | 5 |
| Mixed Precision | ✅ | ✅ | ✅ |
| Grad Clipping | 1.0 | 1.0 | 1.0 |

---

## 📈 Experiments & Results

### Leaderboard Progression

| Version | Architecture | Key Changes | Val F1 | LB Score |
|---------|-------------|-------------|--------|----------|
| V1 | CNN only | Basic mel + CE loss | 0.85 | 0.78 |
| V2 | CNN + AST | Added AST, basic ensemble | 0.95 | 0.88 |
| V3 | CNN + AST | Added PL, temperature scaling | 0.98 | 0.91 |
| **V4** | **CNN + AST** | **5-way TTA, label smoothing** | **0.995** | **0.923** |
| V7 | CNN + AST | +Batch32, +TempoAug, +HigherLR | 0.997 | 0.906 ↓ |
| V11 | CNN + AST | +SWA, +ThresholdTuning, +NoOverNorm | 0.975 | 0.882 ↓ |

### Key Ablation Findings

| Ablation | LB Impact | Conclusion |
|----------|-----------|------------|
| Remove TTA (5-way → 1-way) | -0.01 | TTA helps modestly |
| Remove PL | -0.015 | PL is important |
| Batch 16 → 32 | -0.017 | Larger batch hurts |
| Add tempo augmentation | -0.017 | Pitch change = wrong distribution |
| Add per-class thresholds | -0.041 | Overfits to val, catastrophic on test |
| Change SNR 5-25 → 0-20 | -0.02 | 0dB noise too aggressive |

### Confusion Matrix Insights (from V4)

Most confused genre pairs:
- **Rock ↔ Metal**: Shared distorted guitar timbres
- **Blues ↔ Jazz**: Similar chord progressions and instrumentation
- **Country ↔ Rock**: Acoustic guitar overlap
- **Disco ↔ Pop**: Similar production style and tempo

---

## 💡 Key Insights

### What Actually Matters (Tier 1)
1. **Synthetic mashup generation** — replicating test distribution is the #1 lever
2. **Pretrained backbone** — AST (AudioSet) provides massive head start
3. **Pseudo-labeling** — bridges remaining synthetic→real gap
4. **Don't change what works** — V4's augmentation is proven; every modification degraded score

### What Helps Marginally (Tier 2)
5. **More samples/genre** — more diversity, needs more compute
6. **Noise position randomization** — topper-confirmed strategy
7. **5-way TTA** — modest but free improvement

### What Hurts (Tier 3 — Avoid)
8. **Tempo/pitch augmentation** — changes genre character
9. **Per-class threshold tuning** — overfits to non-representative val set
10. **Aggressive normalization changes** — real test isn't perfectly normalized but model expects it
11. **Batch size increases** — fewer gradient steps, sharper minima

### Topper Insights (from competition forum)
> "We cannot perfectly make samples like they have made in test data."
> "Just randomly choose stems and then add noise on random positions with random intensities."
> "Huge dataset I have prepared."

**The gap between 0.923 and 0.98+ is compute, not techniques.** The topper uses the same AST model with massively more data diversity.

---

## 🔄 How to Reproduce

### Best Score (0.923) on Kaggle
```bash
# Upload kaggle_v4_safe.py as a Kaggle notebook
# Select GPU T4 x2 accelerator
# Run all cells
# Submit submission.csv
```

### Expected Output
```
PHASE 1: CNN — 15 epochs, ~3 hours
PHASE 2: AST — 8 epochs, ~2.5 hours
PHASE 3: Pseudo-labeling — ~5 min
PHASE 4: AST Domain Adaptation — 5 epochs, ~2 hours
PHASE 5: TTA Inference — ~15 min
Total: ~8 hours (fits Kaggle 9hr limit)
```

---

## 📚 References

1. Gong, Y. et al. (2021). **AST: Audio Spectrogram Transformer**. Interspeech.
2. Kong, Q. et al. (2020). **PANNs: Large-Scale Pretrained Audio Neural Networks**. IEEE/ACM TASLP.
3. Chen, K. et al. (2022). **HTS-AT: A Hierarchical Token-Semantic Audio Transformer**. ICASSP.
4. Koutini, K. et al. (2022). **PaSST: Efficient Training of Audio Transformers with Patchout**. Interspeech.
5. Piczak, K. (2015). **ESC: Dataset for Environmental Sound Classification**. ACM MM.
6. Tan, M. & Le, Q. (2019). **EfficientNet: Rethinking Model Scaling for CNNs**. ICML.
7. Lee, D.H. (2013). **Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method**. ICML Workshop.

---

## 📄 License

This project is for educational purposes as part of the Jan 2026 DLGenAI course.
