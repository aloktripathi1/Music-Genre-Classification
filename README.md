# Jan 2026 DLGenAI Project - Messy Mashup

**Predicting Music Genre from Noisy Mashups**

Part of the **Jan 2026 Deep Learning & Generative AI (DLGenAI) Project**.

The goal is to predict the music genre of a noisy audio mashup. Each mashup is created by mixing instrument stems (drums, vocals, bass, others) from different songs of the same genre, with tempo changes and added noise.

The task focuses on building models that are robust to noise and variations, similar to real-world music classification problems.

---

## Task

Given a noisy audio mashup, predict one of 10 genres:

```
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

Evaluation metric: **Macro F1 Score**

---

## Dataset

| Component | Description |
|-----------|-------------|
| `genres_stems/` | 10 genres × 100 songs × 3 stems (drums, vocals, bass) — `others` stem missing for all songs |
| `ESC-50-master/` | 2000 environmental noise clips (50 categories) for augmentation |
| `mashups/` | 3020 unlabeled test mashups (stems mixed across songs + tempo adjusted + ESC-50 noise added) |

---

## Repo Structure

```
messy-mashup/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb                     # Deep EDA + audio statistics
│   ├── 02_classic_ml.ipynb              # Classical ML baselines (RF, GBM, SVM, KNN, LR)
│   ├── 03_cnn_efficientnet.ipynb        # EfficientNet-B0 CNN with mashup augmentation
│   ├── 04a_data_generation.ipynb        # Synthetic mashup generation (25k samples)
│   └── 04b_resnet50_training.ipynb      # ResNet50 on precomputed spectrograms
├── milestones/
│   ├── milestone-1.ipynb                # Data exploration & preprocessing
│   ├── milestone-2.ipynb                # Classical ML baselines & analysis
│   └── milestone-3.ipynb                # Deep learning pipeline & final submission
└── submissions/
    └── .gitkeep
```

---

## Approach

### EXP_001 — EDA + Preprocessing
- Dataset inspection: 10 genres × 100 songs × 3 available stems (1000 missing `others` stems)
- Audio statistics per stem per genre (RMS, energy, ZCR, spectral centroid, SNR)
- Stem importance analysis: **drums** carry most genre signal (ANOVA F=76.8), then **vocals** (60.1), **bass** weakest (18.8)
- Train vs test distribution comparison — significant domain shift quantified
- ESC-50 noise profile and spectral overlap analysis

### EXP_002 — Classical ML Baselines
- Feature extraction: ~90 features (MFCCs, delta MFCCs, chroma, spectral contrast, tonnetz, tempo)
- Models: Random Forest, Gradient Boosting, SVM (RBF), KNN, Logistic Regression
- 5-fold stratified CV with Macro F1
- Voting ensemble of top 3 models

### EXP_003 — EfficientNet-B0 CNN
- Architecture: waveform → MelSpectrogram (GPU) → InstanceNorm → EfficientNet-B0 (pretrained) → GeM pooling → Linear(10)
- On-the-fly mashup augmentation: mix 2–4 stems from different songs, ESC-50 noise injection, overdrive/clipping
- Stem volume weighted by EDA importance (drums=0.45, vocals=0.35, bass=0.20)
- CosineAnnealing LR, label smoothing, mixed precision, 5× TTA for inference

### EXP_004 — Synthetic Mashup + ResNet50 Pipeline
- **Notebook 4a:** Generate 25,000 synthetic mashups (2500/genre) matching test distribution
  - Cross-song stem mixing, random gain (0.5–1.5), ESC-50 noise at random SNR
  - Precompute mel spectrograms → save as `.pt` tensors
- **Notebook 4b:** Train ResNet50 (ImageNet pretrained, fully unfrozen) on precomputed spectrograms
  - Differential LR: backbone 1e-4, head 1e-3
  - Batch size 64, 40 epochs, CosineAnnealing

---

## Results

| Experiment | Model | Val Macro F1 | LB Score |
|-----------|-------|-------------|----------|
| EXP_002 | Classical ML Ensemble | TBD | TBD |
| EXP_003 | EfficientNet-B0 CNN | TBD | TBD |
| EXP_004 | ResNet50 (25k mashups) | TBD | TBD |

---

## Key Findings from EDA

- **`others` stem is missing** for all 1000 songs — only drums, vocals, bass available
- **Drums** are most genre-discriminative, then vocals; bass carries least signal
- **Classical/jazz** are very quiet (RMS ~0.003) — model must not rely on volume
- **Significant distribution shift** between clean training stems and noisy test mashups
- Synthetic mashup augmentation is the single most important factor for LB performance

---

## How to Run

All notebooks are designed to run on **Kaggle** with GPU (T4).

1. Upload the competition dataset
2. Run notebooks in order: `01_eda` → `02_classic_ml` → `03_cnn_efficientnet` → `04a` → `04b`
3. Each experiment logs to **WandB** automatically
4. For EXP_004: run 04a first, save output as Kaggle dataset, then run 04b with that as input

---

## Tools

- PyTorch, torchaudio, timm
- librosa, scikit-learn
- WandB for experiment tracking