# Model loading, inference, and submission CSV generation

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import IDX2GENRE, OUTPUT_DIR, TEST_CSV
from dataset import TestDataset


@torch.no_grad()
def predict(model, test_loader, device):
    # Run inference on test data, return predictions and IDs
    model.eval()
    all_preds, all_ids, all_probs = [], [], []

    for mel, ids in test_loader:
        mel = mel.to(device)
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_ids.extend(ids)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_preds, all_ids, all_probs


def generate_submission(preds, ids, filename='submission.csv'):
    # Generate Kaggle submission CSV
    test_df = pd.read_csv(TEST_CSV, dtype={'id': str})
    pred_dict = {str(id_): IDX2GENRE[p] for id_, p in zip(ids, preds)}
    test_df['genre'] = test_df['id'].apply(lambda x: pred_dict.get(str(x), 'rock'))
    out_path = os.path.join(OUTPUT_DIR, filename)
    test_df[['id', 'genre']].to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
    print(test_df['genre'].value_counts().sort_index())
    return out_path


def load_model(model, weights_path, device):
    # Load saved weights into model
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded weights from {weights_path}")
    return model


def run_inference(model, device, batch_size=32, num_workers=2,
                  submission_name='submission.csv'):
    # Full inference pipeline — load test data, predict, save submission
    test_ds = TestDataset()
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    preds, ids, probs = predict(model, test_loader, device)
    out_path = generate_submission(preds, ids, submission_name)
    return preds, ids, probs, out_path