# Messy Mashup — Music Genre Classifier (Deployment)

3-model ensemble for music genre classification from noisy audio mashups.
Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/aloktripathi/music-genre-classifier).

## Live Demo

👉 **[Try it here](https://huggingface.co/spaces/aloktripathi/music-genre-classification)**

## Architecture

| Model | Params | Weight | Role |
|-------|--------|--------|------|
| EfficientNet-B0 | 4M | 10% | Local spectral patterns |
| AST (AudioSet pretrained) | 86M | 60% | Global temporal structure |
| ResNet-50 | 23.5M | 30% | Architectural diversity |

**Ensemble:** Weighted softmax averaging → **0.9504 Macro F1** on Kaggle.

## Files

```
├── app.py              # Streamlit app
├── Dockerfile          # HF Spaces Docker config
├── best_cnn.pth        # EfficientNet-B0 weights
├── best_ast.pth        # AST weights
└── best_resnet50.pth   # ResNet-50 weights
```

## Run Locally

```bash
pip install streamlit torch torchaudio timm transformers librosa
streamlit run app.py
```

## Competition

Built for the **Messy Mashup** Kaggle competition — Jan 2026 DL & GenAI Project, IIT Madras.

Training code: [messy-mashup](https://github.com/aloktripathi/messy-mashup)

