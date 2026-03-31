import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, librosa, timm, os, tempfile
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor, ASTForAudioClassification

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")

st.markdown("""<style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .stDeployButton {display:none;}
    .block-container {padding-top:2rem; max-width:720px;}
</style>""", unsafe_allow_html=True)

DEVICE = torch.device('cpu')
GENRES = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE_EMOJI = {'blues':'🎸','classical':'🎻','country':'🤠','disco':'🪩','hiphop':'🎤',
               'jazz':'🎷','metal':'🤘','pop':'🎵','reggae':'🌴','rock':'🎸'}
W_CNN, W_AST, W_RES = 0.10, 0.60, 0.30
SR_CNN, SR_AST, SR_RES = 22050, 16000, 22050
N_MELS, N_FFT, HOP = 128, 2048, 512
FMIN, FMAX, DUR = 20, 8000, 10.0

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2,-1)).pow(1.0/self.p)

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(sample_rate=SR_CNN, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=FMIN, f_max=FMAX)
        self.amp_to_db = T.AmplitudeToDB(top_db=80)
        self.inst_norm = nn.InstanceNorm2d(1)
        self.spec_aug_freq = T.FrequencyMasking(27)
        self.spec_aug_time = T.TimeMasking(80)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(nn.LayerNorm(nf), nn.Dropout(0.5), nn.Linear(nf, 10))
    def forward(self, x):
        with torch.no_grad():
            s = self.mel_spec(x)
            s = self.amp_to_db(s).unsqueeze(1)
            s = self.inst_norm(s)
        return self.head(self.gem(self.backbone(s)))

class ResnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(nn.LayerNorm(nf), nn.Dropout(0.4), nn.Linear(nf, 10))
    def forward(self, x):
        return self.head(self.gem(self.backbone(x)))

class AstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=10, ignore_mismatched_sizes=True)
    def forward(self, x):
        return self.ast(input_values=x).logits

@st.cache_resource
def load_models():
    cnn = CnnModel()
    cnn.load_state_dict(torch.load("best_cnn.pth", map_location=DEVICE, weights_only=True), strict=False)
    cnn.eval()
    res = ResnetModel()
    res.load_state_dict(torch.load("best_resnet50.pth", map_location=DEVICE, weights_only=True), strict=False)
    res.eval()
    ast_m = AstModel()
    ast_m.load_state_dict(torch.load("best_ast.pth", map_location=DEVICE, weights_only=True), strict=False)
    ast_m.eval()
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    return cnn, res, ast_m, fe

cnn_model, resnet_model, ast_model, ast_fe = load_models()

def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=DUR)
    t = int(sr * DUR)
    if len(y) < t: y = np.pad(y, (0, t - len(y)))
    elif len(y) > t: y = y[:t]
    return y

def get_cnn_probs(path):
    y = load_audio(path, SR_CNN)
    with torch.no_grad():
        logits = cnn_model(torch.from_numpy(y).float().unsqueeze(0))
    return F.softmax(logits, dim=1).numpy()[0]

def get_ast_probs(path):
    y = load_audio(path, SR_AST)
    inp = ast_fe([y], sampling_rate=SR_AST, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
    with torch.no_grad():
        logits = ast_model(inp["input_values"])
    return F.softmax(logits, dim=1).numpy()[0]

def get_res_probs(path):
    y = load_audio(path, SR_RES)
    S = librosa.feature.melspectrogram(y=y, sr=SR_RES, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    mel = torch.from_numpy(S_db).float()
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    with torch.no_grad():
        logits = resnet_model(mel.unsqueeze(0).unsqueeze(0))
    return F.softmax(logits, dim=1).numpy()[0]

# ═══════════ UI ═══════════

st.markdown("""
<div style="text-align:center; margin-bottom:8px;">
    <h1 style="font-size:2.2em; color:#1a1a2e; margin-bottom:2px;">🎵 Music Genre Classifier</h1>
    <p style="color:#666; font-size:0.95em; margin-top:0;">
        Classify music using an ensemble of <strong>EfficientNet-B0</strong>,
        <strong>Audio Spectrogram Transformer</strong> &amp; <strong>ResNet-50</strong>
    </p>
    <span style="background:#1a1a2e; color:white; padding:4px 16px; border-radius:16px;
                 font-size:0.8em; font-weight:600;">Kaggle Score: 0.9504 Macro F1</span>
</div>
""", unsafe_allow_html=True)

st.markdown("")

uploaded = st.file_uploader(
    "Upload a music clip (WAV, MP3, OGG, FLAC · up to 10 seconds)",
    type=["wav", "mp3", "ogg", "flac"],
    label_visibility="visible"
)

if uploaded:
    st.audio(uploaded, format="audio/wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing audio with 3 models..."):
        cp = get_cnn_probs(tmp_path)
        ap = get_ast_probs(tmp_path)
        rp = get_res_probs(tmp_path)
        ep = W_CNN * cp + W_AST * ap + W_RES * rp

    os.unlink(tmp_path)

    genre = GENRES[ep.argmax()]
    conf = float(ep.max()) * 100
    emoji = GENRE_EMOJI.get(genre, '🎵')
    cg, ag, rg = GENRES[cp.argmax()], GENRES[ap.argmax()], GENRES[rp.argmax()]
    agree = cg == ag == rg
    conf_color = "#16a34a" if conf >= 80 else "#ca8a04" if conf >= 50 else "#dc2626"

    st.markdown("---")

    st.markdown(f"""
    <div style="text-align:center; padding:24px; background:#1a1a2e; border-radius:14px;
                color:white; margin-bottom:16px;">
        <div style="font-size:2.8em; margin-bottom:2px;">{emoji}</div>
        <div style="font-size:1.6em; font-weight:700; text-transform:uppercase;
                    letter-spacing:2px;">{genre}</div>
        <div style="margin-top:8px;">
            <span style="background:rgba(255,255,255,0.15); padding:4px 14px; border-radius:16px;
                         font-size:0.88em;">{conf:.1f}% confidence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="font-size:0.85em; color:#555; font-weight:500;">Confidence</span>
            <span style="font-size:0.85em; color:{conf_color}; font-weight:700;">{conf:.1f}%</span>
        </div>
        <div style="background:#eee; border-radius:6px; height:6px; overflow:hidden;">
            <div style="background:{conf_color}; height:100%; width:{min(conf,100)}%;
                        border-radius:6px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Model Predictions")
    c1, c2, c3 = st.columns(3)
    for col, (name, w, g, probs) in zip(
        [c1, c2, c3],
        [("🧠 AST", W_AST, ag, ap),
         ("🏗️ ResNet-50", W_RES, rg, rp),
         ("⚡ EfficientNet", W_CNN, cg, cp)]
    ):
        mc = float(probs.max()) * 100
        match_icon = "✅" if g == genre else "❌"
        bg = "#f0fdf4" if g == genre else "#fef2f2"
        border = "#bbf7d0" if g == genre else "#fecaca"
        with col:
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:10px;
                        padding:12px; text-align:center;">
                <div style="font-size:0.75em; color:#888; font-weight:500;">{name} ({int(w*100)}%)</div>
                <div style="font-size:1.1em; font-weight:700; margin:4px 0; color:#333;">
                    {GENRE_EMOJI.get(g,'')} {g.title()} {match_icon}</div>
                <div style="font-size:0.78em; color:#888;">{mc:.1f}% confident</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    if agree:
        st.success("✅ All 3 models agree on the prediction")
    else:
        st.warning("⚠️ Models disagree — ensemble resolves the final prediction")

    st.markdown("")
    st.markdown("##### Genre Probabilities")
    import pandas as pd
    prob_df = pd.DataFrame({
        'Genre': [f"{GENRE_EMOJI.get(g,'')} {g.title()}" for g in GENRES],
        'Probability (%)': (ep * 100).round(2)
    }).sort_values('Probability (%)', ascending=True)
    st.bar_chart(prob_df.set_index('Genre'), horizontal=True, color="#1a1a2e")

    sp = sorted(enumerate(ep), key=lambda x: x[1], reverse=True)
    top3 = " → ".join([f"{GENRE_EMOJI.get(GENRES[i],'')} {GENRES[i].title()} ({p*100:.1f}%)" for i, p in sp[:3]])
    st.caption(f"**Top 3:** {top3}")

st.markdown("---")
st.caption("""
**How it works:** Your audio is processed by three models —
EfficientNet-B0 (local patterns), AST (global structure, pretrained on AudioSet),
and ResNet-50 (architectural diversity). Predictions are combined with optimized weights (10/60/30%).
· Built for the Messy Mashup competition — IIT Madras DL & GenAI Project.
""")