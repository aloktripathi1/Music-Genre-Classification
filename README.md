# Messy Mashup - Music Genre Classification

**Predicting Music Genre from Noisy Mashups**

Part of the **Jan 2026 Deep Learning & Generative AI (DLGenAI) Project** at IIT Madras.

🔗 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/aloktripathi/music-genre-classification)
📊 **Kaggle Score:** 0.9614 Macro F1

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
| `genres_stems/` | 10 genres × 100 songs × 3 stems (drums, vocals, bass) — `others` stem missing |
| `ESC-50-master/` | 2000 environmental noise clips (50 categories) for augmentation |
| `mashups/` | 3020 unlabeled test mashups (stems mixed + tempo adjusted + noise) |

---

## Results

| Experiment | Model | Val F1 | LB Score |
|-----------|-------|--------|----------|
| EXP_002 | Scratch CNN (no pretraining) | 0.75 | 0.5293 |
| EXP_003 | EfficientNet-B0 (ImageNet pretrained) | 0.82 | 0.8504 |
| EXP_004 | AST v1 (AudioSet pretrained) | 0.88 | 0.9279 |
| EXP_004 | AST v2 (stronger aug — worse) | 0.88 | 0.8973 |
| EXP_006 | ResNet-50 (on-the-fly) | 0.86 | ~0.86 |
| — | CNN + AST (20/80) | — | 0.9349 |
| — | CNN + AST + ResNet (10/60/30) | — | 0.9504 |
| — | **3-AST + v1 + CNN + ResNet** | — | **0.9614** |

---

## Approach

### EDA
- Discovered `others` stem missing for all 1000 songs
- Drums carry most genre signal (ANOVA F=76.8), then vocals (60.1)
- Classical/jazz 20× quieter than hiphop — needs volume-invariant normalization
- Significant train↔test distribution shift quantified

### Model Training
- **On-the-fly mashup augmentation** — drums from song A + vocals from song B + bass from song C
- ESC-50 noise injection (SNR 5-25 dB), overdrive, SpecAugment, Mixup
- Instance Normalization for volume invariance
- GeM pooling for discriminative focus

### Ensemble
- Weighted probability averaging with exhaustive weight sweep
- Multi-seed AST (3 seeds) averaged for variance reduction
- Final: 3-AST avg (42%) + AST v1 (25%) + CNN (3%) + ResNet (30%)

### Key Learnings
- On-the-fly augmentation provides infinite diversity from 1000 songs
- AudioSet pretraining transfers well to music genre classification
- More augmentation ≠ better (AST v2 was worse than v1)
- Multi-seed training reduces variance (+0.011 improvement)
- ResNet most diverse from AST (528/3020 predictions differ) — gets 30% weight

---

## Repo Structure

```
messy-mashup/
├── notebooks/          # Experiment notebooks (.ipynb)
│   ├── 01_eda.ipynb
│   ├── 02_scratch_cnn.ipynb
│   ├── 03_cnn_efficientnet.ipynb
│   ├── 04_ast.ipynb
│   ├── 05_ensemble.ipynb
│   ├── 06_resnet50.ipynb
│   └── 07_multi_seed_ast.ipynb
├── scripts/            # Training scripts (.py)
├── configs/            # Shared configuration
├── deployment/         # HuggingFace Spaces (Streamlit + Docker)
├── reports/            # LaTeX report + figures
├── eda/                # EDA notebook
├── milestones/         # Course milestone submissions
├── docs/               # Setup guides
└── wandb_screenshots/  # W&B dashboard screenshots
```

---

## Deployment

Deployed as a Streamlit app on HuggingFace Spaces (Docker SDK, CPU).

Upload a music clip → 3 models process independently → weighted ensemble predicts genre.

🔗 [https://huggingface.co/spaces/aloktripathi/music-genre-classifier](https://huggingface.co/spaces/aloktripathi/music-genre-classification)

---

## Tools

- PyTorch, torchaudio, timm, HuggingFace Transformers
- librosa, scikit-learn
- WandB for experiment tracking
- Streamlit + Docker for deployment
- Trained on Lightning.ai (L4 GPU) and Kaggle (T4 GPU)