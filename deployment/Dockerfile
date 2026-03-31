FROM python:3.12-slim

RUN pip install --no-cache-dir streamlit torch torchaudio timm transformers librosa numpy

WORKDIR /app
COPY . .

# Pre-download AST model during build (avoids 403 at runtime)
RUN python -c "from transformers import ASTFeatureExtractor, ASTForAudioClassification; \
    ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593'); \
    ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593', num_labels=10, ignore_mismatched_sizes=True)"

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.maxUploadSize=200"]