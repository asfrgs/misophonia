# 🔇 Shush — Misophonia Trigger Reducer

Shush is a Streamlit app that detects and reduces common **misophonia triggers** (chewing, typing, coughing, etc.) in audio/video files or live microphone input — in real time.

It uses **YAMNet** (Google's audio classifier) to detect trigger sounds and **DeepFilterNet** for neural noise suppression.

---

## ✨ Features

- 📁 **Upload Tab** — Upload an audio or video file, process it, preview before/after, and download the cleaned audio
- 🎙️ **Live Tab** — Real-time mic input with trigger ducking via WebRTC
- 🧠 Powered by **YAMNet** (TensorFlow Hub) for sound classification
- 🔉 Enhanced with **DeepFilterNet** for neural noise reduction
- Supports formats: `mp3`, `wav`, `m4a`, `aac`, `flac`, `ogg`, `mp4`, `mkv`, `mov`, `avi`, `webm`

---

## 🚀 Getting Started

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd "Yam+DeepNet"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Note:** First run will download the YAMNet model from TensorFlow Hub (~25 MB) and DeepFilterNet weights. This is a one-time download.

### 4. Run the app

```bash
streamlit run python.py
```

> **Tip:** Renaming `python.py` → `shush_app.py` is recommended to avoid a TensorFlow double-import quirk (the app handles it either way via a `__name__` guard).

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `tensorflow` / `tensorflow-hub` | YAMNet audio classifier |
| `DeepFilterNet` | Neural noise suppression |
| `soundfile` | Audio I/O |
| `scipy` | Audio resampling |
| `numpy` | Numerical processing |
| `imageio-ffmpeg` | FFmpeg bindings for media extraction |
| `streamlit-webrtc` + `av` | Live mic streaming (Live tab) |

Full dependency list is in [`requirements.txt`](requirements.txt).

---

## ⚙️ How It Works

1. Audio is extracted at **16 kHz** and fed to **YAMNet** in sliding windows
2. YAMNet scores each window against ~521 sound classes — trigger classes are flagged
3. A gain envelope is computed: trigger regions are attenuated up to **–20 dB**
4. *(Optional)* **DeepFilterNet** applies neural noise suppression on top
5. Cleaned audio is written back at **48 kHz**

### Trigger sounds detected

Chewing · Biting · Crunch · Cough · Throat clearing · Sneeze · Sniff · Breathing · Typing · Clicking · Finger snapping · Cutlery · and more.

---

## 🛠️ Requirements

- Python **3.10 – 3.12**
- Windows / macOS / Linux
- FFmpeg (bundled via `imageio-ffmpeg` — no separate install needed)
- GPU optional but recommended for faster TensorFlow inference

