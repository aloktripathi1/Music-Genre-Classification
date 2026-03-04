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
| `genres_stems/` | 10 genres × 100 songs × 4 stems (drums, vocals, bass, others) |
| `ESC-50-master/` | 2000 environmental noise clips for augmentation |
| `mashups/` | 3020 unlabeled test mashups (stems mixed + tempo adjusted + noise added) |

---

## Repo Structure

```
messy-mashup/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── exp_001_eda.ipynb                # Deep EDA + feature extraction
│   ├── exp_002_classical_ml.ipynb       # Classical ML baselines (RF, LGB, SVM, LR)
│   └── exp_003_scratch_cnn.ipynb        # Scratch CNN pipeline
│   └── exp_004_cnn_ast.ipynb            # CNN + AST deep learning pipeline
├── milestones/
│   ├── milestone-1.ipynb             # Data exploration & preprocessing
│   ├── milestone-2.ipynb             # Classical ML baselines & analysis
│   └── milestone-3.ipynb             # Deep learning pipeline & final submission
└── submissions/
    └── .gitkeep
```

---

## Approach

**EXP_001** — EDA + Preprocessing
- Dataset inspection, audio statistics (RMS, SNR, spectral features)
- Train vs test distribution comparison (domain gap analysis)
- Feature extraction (~92 features: MFCCs, chroma, spectral, tempo, tonnetz)

**EXP_002** — Classical ML Baselines
- Logistic Regression, Random Forest, LightGBM, SVM
- 5-fold stratified CV with per-class F1 breakdown
- Feature importance analysis, confusion diagnostics

**EXP_003** — Deep Learning Pipeline
- Scratch CNN (EfficientNet-B0 + SpecAugment + GeM pooling)
- Pretrained AST (Audio Spectrogram Transformer)
- Tempo & pitch augmentation, noise injection, waveform mixup
- Pseudo-label domain adaptation on test data
- Calibrated CNN + AST ensemble with TTA

---

## Results

| Experiment | Model | Macro F1 |
|-----------|-------|----------|
| EXP_002 | Classical ML (best single) | TBD |
| EXP_003 | CNN + AST Ensemble | 0.923 (baseline) |
| EXP_003 | CNN + AST v5.1 (improved) | TBD |

---

## How to Run

All notebooks are designed to run on **Kaggle** with GPU (T4 x2).

1. Upload the competition dataset
2. Run notebooks in order: `exp_001` → `exp_002` → `exp_003`
3. Each experiment logs to **WandB** automatically

---

## Tools

- PyTorch, torchaudio, timm, transformers (HuggingFace AST)
- librosa, scikit-learn, LightGBM
- WandB for experiment tracking