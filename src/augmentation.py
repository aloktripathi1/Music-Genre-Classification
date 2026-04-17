# Audio loading and augmentation utilities
# Handles waveform loading, mel conversion, noise injection, and augmentations

import random
import numpy as np
import librosa

from config import SR, TARGET_LEN, N_MELS, N_FFT, HOP


def load_wav(path):
    # Load audio file, resample to SR, pad/crop to TARGET_LEN.
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        if len(y) < TARGET_LEN:
            y = np.pad(y, (0, TARGET_LEN - len(y)))
        elif len(y) > TARGET_LEN:
            s = random.randint(0, len(y) - TARGET_LEN)
            y = y[s:s + TARGET_LEN]
        return y.astype(np.float32)
    except Exception:
        return np.zeros(TARGET_LEN, dtype=np.float32)


def wav_to_mel(y):
    # Convert waveform to log-mel spectrogram.
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=20, fmax=8000
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    return S_db.astype(np.float32)


def add_noise_snr(signal, noise, snr_db):
    # Add noise to signal at a specified SNR in dB.
    sig_power = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return signal + noise * scale


def apply_overdrive(signal, gain=2.5):
    # Simulate overdrive distortion by amplifying and clipping.
    return np.clip(signal * gain, -1.0, 1.0)


def apply_time_shift(signal, max_shift):
    # Circular time shift by random amount.
    shift = random.randint(-max_shift, max_shift)
    return np.roll(signal, shift)


def spec_augment(mel, num_freq_masks=2, freq_mask_size=27,
                 num_time_masks=2, time_mask_size=80):
    # Apply SpecAugment — mask random frequency and time bands.
    mel = mel.copy()
    n_mels, n_frames = mel.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_size)
        f0 = random.randint(0, max(0, n_mels - f))
        mel[f0:f0 + f, :] = 0

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_size)
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, t0:t0 + t] = 0

    return mel


def apply_mixup(mel1, label1, mel2, label2, alpha=0.4):
    # Blend two samples and labels for mixup augmentation.
    lam = np.random.beta(alpha, alpha)
    mel_mix = lam * mel1 + (1 - lam) * mel2
    return mel_mix, lam, label1, label2