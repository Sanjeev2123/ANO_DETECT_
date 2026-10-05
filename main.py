import json, os, tempfile, time
import numpy as np, soundfile as sf, librosa, torch, torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---- load the model ONCE at startup ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
cfg = json.load(open(f"{MODEL_DIR}/config.json"))
MEAN = np.load(f"{MODEL_DIR}/mean.npy")
STD = np.load(f"{MODEL_DIR}/std.npy")

def blk(i, o): return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
D = cfg["input_dim"]
net = nn.Sequential(*blk(D,128), *blk(128,128), *blk(128,64), nn.Linear(64,8), nn.ReLU(),
                    *blk(8,64), *blk(64,128), *blk(128,128), nn.Linear(128,D))
net.load_state_dict(torch.load(f"{MODEL_DIR}/autoencoder.pt", map_location="cpu"))
net.eval()

# ---- app ----
app = FastAPI(title="ANO_DETECT API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_BYTES = 20 * 1024 * 1024
ALLOWED = (".wav", ".mp3", ".flac")

def load_audio(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1) if cfg["avg_channels"] else y[:, 0]
    if sr != cfg["sr"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg["sr"])
    return y

def score(y):
    m = librosa.feature.melspectrogram(y=y, sr=cfg["sr"], n_fft=cfg["n_fft"],
                                       hop_length=cfg["hop"], n_mels=cfg["n_mels"], power=2.0)
    m = 10 * np.log10(m + 1e-10)
    fr = cfg["frames"]
    T = m.shape[1] - fr + 1
    x = np.stack([m[:, i:i+fr].T.reshape(-1) for i in range(T)]).astype(np.float32)
    x = torch.tensor((x - MEAN) / STD)
    with torch.no_grad():
        return float(((net(x) - x) ** 2).mean())

@app.get("/health")
def health():
    return {"status": "ok", "model_version": cfg["model_version"], "machine_id": cfg["machine_id"]}

@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    t0 = time.time()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Unsupported file type. Use WAV, MP3 or FLAC.")
    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "File too large (max 20 MB).")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data); path = tmp.name
    try:
        y = load_audio(path)
    except Exception:
        raise HTTPException(400, "Audio file could not be read. It may be corrupted.")
    finally:
        os.remove(path)

    base = {"threshold": cfg["threshold"], "confidence": None,
            "model_version": cfg["model_version"], "machine_id": cfg["machine_id"]}
    # simple quality gate: too short or almost silent
    if len(y) < cfg["sr"] or float(np.sqrt(np.mean(y ** 2))) < 1e-4:
        return {**base, "result": "LOW_QUALITY", "anomaly_score": None, "signal_quality": "POOR",
                "processing_time_ms": int((time.time() - t0) * 1000)}
    s = score(y)
    return {**base, "result": "ANOMALY" if s > cfg["threshold"] else "NORMAL",
            "anomaly_score": round(s, 4), "signal_quality": "GOOD",
            "processing_time_ms": int((time.time() - t0) * 1000)}