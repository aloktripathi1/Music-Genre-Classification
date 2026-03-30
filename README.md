# Messy Mashup - Music Genre Classification

**Predicting Music Genre from Noisy Mashups**

Part of the **Jan 2026 Deep Learning & Generative AI (DLGenAI) Project** at IIT Madras.

🔗 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/aloktripathi/music-genre-classifier)

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
| EXP_003 | EfficientNet-B0 CNN | 0.82 | 0.8504 |
| EXP_004 | AST v1 (AudioSet pretrained) | 0.88 | 0.9279 |
| EXP_004 | AST v2 (stronger aug — worse) | 0.88 | 0.8973 |
| EXP_006 | ResNet-50 | 0.86 | ~0.86 |
| EXP_008 | ResNet-50 (precomputed, overfit) | 0.9948 | 0.88 |
| EXP_008 | EfficientNet (precomputed, overfit) | 0.9952 | 0.88 |
| — | CNN + AST (20/80) | — | 0.9349 |
| — | **CNN + AST + ResNet (10/60/30)** | — | **0.9504** |

---

## Approach

### EDA
- Discovered `others` stem missing for all 1000 songs
- Drums carry most genre signal (ANOVA F=76.8), then vocals (60.1)
- Significant train↔test distribution shift quantified

### Model Training
- **On-the-fly mashup augmentation** — drums from song A + vocals from song B + bass from song C
- ESC-50 noise injection (SNR 5-25 dB), overdrive, SpecAugment, Mixup
- Instance Normalization for volume invariance
- GeM pooling for discriminative focus

### Ensemble
- Weighted probability averaging with exhaustive weight sweep
- AST dominates (60%) due to AudioSet pretraining robustness

### Key Learnings
- On-the-fly augmentation >> precomputed (avoids synthetic overfitting)
- AudioSet pretraining transfers well to music genre classification
- More augmentation ≠ better (AST v2 was worse than v1)
- Pseudo-labeling on test data degraded performance

---

## Repo Structure

```
messy-mashup/
├── notebooks/          # Kaggle notebooks (.ipynb)
├── scripts/            # Lightning.ai training scripts (.py)
├── configs/            # Shared configuration
├── deployment/         # HuggingFace Spaces (Streamlit + Docker)
├── reports/            # LaTeX report + figures
├── eda/                # EDA notebook for report charts
├── milestones/         # Course milestone submissions
├── docs/               # Setup guides
├── submissions/        # CSV submissions
└── wandb_screenshots/  # W&B dashboard screenshots
```

---

## Deployment

Deployed as a Streamlit app on HuggingFace Spaces (Docker SDK, CPU).

Upload a music clip → 3 models process independently → weighted ensemble predicts genre.

🔗 [https://huggingface.co/spaces/aloktripathi/music-genre-classification](https://huggingface.co/spaces/aloktripathi/music-genre-classification)

---

## Tools

- PyTorch, torchaudio, timm, HuggingFace Transformers
- librosa, scikit-learn
- WandB for experiment tracking
- Streamlit + Docker for deployment
- Trained on Lightning.ai (L4 GPU) and Kaggle (T4 GPU)
