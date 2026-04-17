# Shared configuration for Messy Mashup competition.
# Used across all experiments.

SEED = 42
SR = 22050
SR_DL = 16000
DURATION = 10.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

GENRES = sorted(['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock'])
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
IDX2GENRE = {i: g for g, i in GENRE2IDX.items()}

STEMS = ['drums', 'vocals', 'bass']  # others missing for all songs
STEM_WEIGHTS = {'drums': 0.45, 'vocals': 0.35, 'bass': 0.20}

WANDB_ENTITY = "23f3003225-indian-institute-of-technology-madras"
WANDB_PROJECT = "23f3003225-dl-genai-project"
