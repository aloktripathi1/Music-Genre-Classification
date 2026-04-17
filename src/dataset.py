# Dataset classes for music genre classification
# Handles mashup generation, validation, and test data loading

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    GENRES, GENRE2IDX, IDX2GENRE, STEMS,
    SR, TARGET_LEN, N_MELS, HOP, TEST_DIR, TEST_CSV
)
from augmentation import load_wav, wav_to_mel, add_noise_snr


class MashupDataset(Dataset):
    # On-the-fly mashup generation for training
    # Mixes stems from different songs of same genre, adds noise, converts to mel
    def __init__(self, stem_idx, noise_files, samples_per_genre=500,
                 snr_range=(5, 25), overdrive_prob=0.3):
        self.stem_idx = stem_idx
        self.noise_files = noise_files
        self.snr_range = snr_range
        self.overdrive_prob = overdrive_prob
        self.samples = []
        for g in GENRES:
            self.samples.extend([GENRE2IDX[g]] * samples_per_genre)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gi = self.samples[idx]
        g = IDX2GENRE[gi]

        # mix stems from different songs (same genre)
        stems = []
        weights = {'drums': 0.45, 'vocals': 0.35, 'bass': 0.20}
        for st in STEMS:
            av = self.stem_idx[g][st]
            if av:
                wav = load_wav(random.choice(av))
                wav = wav * weights[st] * random.uniform(0.7, 1.3)
                stems.append(wav)

        if not stems:
            return torch.zeros(1, N_MELS, TARGET_LEN // HOP + 1), gi

        mix = np.sum(stems, axis=0)

        # add ESC-50 noise at random SNR
        if self.noise_files and random.random() < 0.8:
            noise = load_wav(random.choice(self.noise_files))
            snr = random.uniform(*self.snr_range)
            mix = add_noise_snr(mix, noise, snr)

        # overdrive distortion
        if random.random() < self.overdrive_prob:
            mix = np.clip(mix * 2.5, -1.0, 1.0)

        # random time shift
        shift = random.randint(-SR, SR)
        mix = np.roll(mix, shift)

        # peak normalize
        peak = np.max(np.abs(mix))
        if peak > 1e-6:
            mix = mix / peak

        mel = wav_to_mel(mix)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return torch.from_numpy(mel).unsqueeze(0), gi


class ValDataset(Dataset):
    # Validation dataset — mixes all stems from one song (no augmentation)
    def __init__(self, song_idx):
        self.items = []
        for g in GENRES:
            for s in song_idx[g]:
                self.items.append((s, GENRE2IDX[g]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        si, lb = self.items[idx]
        stems = [load_wav(os.path.join(si['dir'], f"{st}.wav")) for st in si['stems']]
        mix = np.sum(stems, axis=0)
        peak = np.max(np.abs(mix))
        if peak > 1e-6:
            mix = mix / peak
        mel = wav_to_mel(mix)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return torch.from_numpy(mel).unsqueeze(0), lb


class TestDataset(Dataset):
    # Test dataset — loads mashup audio files for inference
    def __init__(self):
        self.df = pd.read_csv(TEST_CSV, dtype={'id': str})
        self.paths = []
        for _, r in self.df.iterrows():
            p = None
            for pat in [f"song{str(r['id']).zfill(4)}.wav", f"{r['id']}.wav"]:
                fp = os.path.join(TEST_DIR, pat)
                if os.path.exists(fp):
                    p = fp
                    break
            self.paths.append(p)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if p:
            mel = wav_to_mel(load_wav(p))
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            return torch.from_numpy(mel).unsqueeze(0), str(self.df.iloc[idx]['id'])
        return torch.zeros(1, N_MELS, TARGET_LEN // HOP + 1), str(self.df.iloc[idx]['id'])