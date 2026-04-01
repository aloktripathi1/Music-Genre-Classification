# Precompute Spectrograms
# Generates 25,000 synthetic mashups → mel spectrograms → .pt tensors
# Save output as Kaggle dataset, use as input for Notebook 2.


import os, glob, random, warnings, time, gc
import numpy as np, pandas as pd
from collections import Counter
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# CONFIG
DATA_ROOT  = '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup'
OUTPUT_DIR = '/kaggle/working/mashup_specs'
STEMS_DIR  = os.path.join(DATA_ROOT, 'genres_stems')
NOISE_DIR  = os.path.join(DATA_ROOT, 'ESC-50-master', 'audio')
TEST_DIR   = os.path.join(DATA_ROOT, 'mashups')
TEST_CSV   = os.path.join(DATA_ROOT, 'test.csv')

SR         = 22050
DURATION   = 10.0
TARGET_LEN = int(SR * DURATION)
N_MELS     = 128
N_FFT      = 2048
HOP_LENGTH = 512
FMIN, FMAX = 20, 8000

GENRES    = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
STEMS     = ['drums', 'vocals', 'bass']

SAMPLES_PER_GENRE_TRAIN = 2500   # 25k total
SAMPLES_PER_GENRE_VAL   = 250    # 2.5k total
STEM_WEIGHTS = {'drums': 0.45, 'vocals': 0.35, 'bass': 0.20}

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

print(f"Will generate {SAMPLES_PER_GENRE_TRAIN*10} train + {SAMPLES_PER_GENRE_VAL*10} val mashups")

# MEL TRANSFORM (GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX
).to(DEVICE)
db_transform = T.AmplitudeToDB(top_db=80).to(DEVICE)

def wav_to_mel(wav_tensor):
    """Compute mel spectrogram on GPU. Returns (n_mels, time) CPU tensor."""
    with torch.no_grad():
        wav_gpu = wav_tensor.unsqueeze(0).to(DEVICE)
        spec = mel_transform(wav_gpu)
        spec = db_transform(spec)
        # Instance normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.squeeze(0).cpu()

print(f"Mel transform on {DEVICE}")

# DATA INDEX
stem_index = {g: {st: [] for st in STEMS} for g in GENRES}

for genre in GENRES:
    gp = os.path.join(STEMS_DIR, genre)
    songs = sorted(s for s in os.listdir(gp) if os.path.isdir(os.path.join(gp, s)))
    for song in songs:
        for st in STEMS:
            fp = os.path.join(gp, song, f"{st}.wav")
            if os.path.exists(fp):
                stem_index[genre][st].append(fp)

noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))
print(f"Noise: {len(noise_files)} clips")

# AUDIO LOADING
def load_wav(path, sr=SR, target_len=TARGET_LEN):
    try:
        wav, orig_sr = torchaudio.load(path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if orig_sr != sr: wav = torchaudio.functional.resample(wav, orig_sr, sr)
        wav = wav.squeeze(0)
        if len(wav) < target_len:
            wav = F.pad(wav, (0, target_len - len(wav)))
        elif len(wav) > target_len:
            start = random.randint(0, len(wav) - target_len)
            wav = wav[start:start + target_len]
        return wav
    except:
        return torch.zeros(target_len)

# MASHUP GENERATION
def generate_mashup(genre, stem_index, noise_files, augment=True):
    """Generate one synthetic mashup waveform."""
    stems_wav = []
    for st in STEMS:
        available = stem_index[genre][st]
        if not available: continue
        wav = load_wav(random.choice(available))
        gain = random.uniform(0.5, 1.5) * (STEM_WEIGHTS[st] / 0.33)
        stems_wav.append(wav * gain)

    if not stems_wav:
        return torch.zeros(TARGET_LEN)

    mix = torch.stack(stems_wav).sum(0)

    if augment:
        # Time shift
        mix = torch.roll(mix, random.randint(-SR, SR))

        # ESC-50 noise (0-3 clips)
        for _ in range(random.randint(0, 3)):
            noise = load_wav(random.choice(noise_files))
            snr_db = random.uniform(3.0, 25.0)
            sig_pwr = mix.pow(2).mean() + 1e-10
            nse_pwr = noise.pow(2).mean() + 1e-10
            scale = (sig_pwr / (nse_pwr * 10 ** (snr_db / 10))).sqrt()
            mix = mix + noise * scale

        # Overdrive (30%)
        if random.random() < 0.3:
            mix = torch.clamp(mix * random.uniform(1.2, 3.0), -1, 1)

    # Normalize
    peak = mix.abs().max()
    if peak > 1e-6:
        mix = mix / peak * random.uniform(0.7, 1.0)
    return mix

# GENERATE TRAIN + VAL
t0 = time.time()

for split, n_per_genre in [('train', SAMPLES_PER_GENRE_TRAIN), ('val', SAMPLES_PER_GENRE_VAL)]:
    count = 0
    for genre in GENRES:
        label = GENRE2IDX[genre]
        for i in tqdm(range(n_per_genre), desc=f"{split}/{genre}", leave=False):
            wav = generate_mashup(genre, stem_index, noise_files, augment=(split == 'train'))
            mel = wav_to_mel(wav)  # (n_mels, time) — computed on GPU
            fname = f"{genre}_{i:04d}.pt"
            torch.save({'mel': mel, 'label': label}, os.path.join(OUTPUT_DIR, split, fname))
            count += 1
    print(f"{split}: {count} samples saved")

# PROCESS TEST SET
print("\nProcessing test set...")
test_df = pd.read_csv(TEST_CSV, dtype={'id': str})

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
    path = None
    for pat in [f"song{str(row['id']).zfill(4)}.wav", f"{row['id']}.wav", f"song{row['id']}.wav"]:
        p = os.path.join(TEST_DIR, pat)
        if os.path.exists(p): path = p; break

    if path:
        wav = load_wav(path)
    else:
        wav = torch.zeros(TARGET_LEN)

    mel = wav_to_mel(wav)
    torch.save({'mel': mel, 'id': str(row['id'])},
               os.path.join(OUTPUT_DIR, 'test', f"{str(row['id']).zfill(4)}.pt"))

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min")

# Verify
for split in ['train', 'val', 'test']:
    n = len(glob.glob(os.path.join(OUTPUT_DIR, split, '*.pt')))
    print(f"  {split}: {n} files")

sample = torch.load(os.path.join(OUTPUT_DIR, 'train', os.listdir(os.path.join(OUTPUT_DIR, 'train'))[0]))
print(f"  mel shape: {sample['mel'].shape}")
print(f"\nsave this notebook output as a Kaggle dataset (e.g. 'mashup-specs-25k')")
print(f"   Then use it as input for the training notebook.")
