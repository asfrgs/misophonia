"""
Shush — Misophonia Trigger Reducer (Streamlit)
Tabs: Upload, Live. Plus optional nature-sound BACKGROUND mixing.

Background sounds: any audio file (mp3/wav/ogg/flac/m4a) placed in the same
folder as this script shows up as a background option with an icon + volume
slider. Set a slider above 0 to mix that sound into the cleaned output.

Run:
    streamlit run shush_app.py
"""

import base64
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import resample_poly

# ==========================================================================
# FIXED CONFIG
# ==========================================================================
TRIGGERS = {
    "Chewing, mastication", "Biting", "Crunch", "Cough", "Throat clearing",
    "Sneeze", "Sniff", "Breathing", "Gasp", "Burping, eructation",
    "Computer keyboard", "Typing", "Typewriter", "Writing",
    "Finger snapping", "Tap", "Clicking", "Slap, smack",
    "Wheeze", "Snoring", "Pant", "Snort", "Hiccup", "Whispering", "Grunt",
    "Cutlery, silverware", "Dishes, pots, and pans", "Chink, clink",
    "Tick", "Tick-tock", "Clickety-clack", "Scratch", "Crack",
}
THRESH_LO, THRESH_HI = 0.10, 0.30
MAX_REDUCTION_DB = 45
PAD_FRAMES = 1
USE_DEEPFILTERNET = True
SR_Y, SR_D = 16000, 48000
HOP, WIN = 0.48, 0.96

# folder to scan for background sounds (same folder as this script)
BG_DIR = Path(__file__).parent

BG_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
# emoji icon guessed from the file name
BG_ICONS = [
    (("rain", "drizzle"), "🌧️"),
    (("thunder", "storm"), "⛈️"),
    (("wave", "sea", "ocean"), "🌊"),
    (("stream", "river", "brook", "creek", "mountain"), "🏞️"),
    (("brown",), "🟤"),
    (("pink",), "🌸"),
    (("white",), "⚪"),
    (("forest", "tree", "jungle"), "🌲"),
    (("fire", "camp"), "🔥"),
    (("wind",), "🌬️"),
    (("bird",), "🐦"),
    (("night", "sleep"), "🌙"),
]
# ==========================================================================


@st.cache_resource(show_spinner="Loading YAMNet (one time)...")
def load_yamnet():
    import tensorflow_hub as hub
    m = hub.load("https://tfhub.dev/google/yamnet/1")
    with open(m.class_map_path().numpy().decode()) as f:
        names = [r["display_name"] for r in csv.DictReader(f)]
    return m, names


@st.cache_resource(show_spinner="Loading DeepFilterNet (one time)...")
def load_df():
    from df import init_df
    model, state, _ = init_df()
    return model, state


def icon_for(name):
    low = name.lower()
    for keys, emoji in BG_ICONS:
        if any(k in low for k in keys):
            return emoji
    return "🎵"


def list_backgrounds():
    """Find background sound files next to this script."""
    out = []
    for f in sorted(BG_DIR.iterdir()):
        if f.suffix.lower() in BG_EXT and "cleaned" not in f.stem.lower():
            pretty = f.stem.replace("-", " ").replace("_", " ").title()
            out.append((f, icon_for(f.name), pretty))
    return out


def extract(path, sr):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(tempfile.mkdtemp()) / f"a_{sr}.wav"
    subprocess.run([ff, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", str(out)], check=True, capture_output=True)
    return out


def dilate(x, k):
    p = np.pad(x, k, mode="edge")
    return np.max([p[i:i + len(x)] for i in range(2 * k + 1)], axis=0)


def mix_backgrounds(audio, sr, selections):
    """selections: list of (Path, volume 0..1). Loops each bg to length and mixes."""
    out = audio.astype(np.float32).copy()
    n = len(out)
    for path, vol in selections:
        try:
            bg = sf.read(str(extract(path, sr)), dtype="float32")[0]
        except Exception:
            continue
        if bg.ndim > 1:
            bg = bg.mean(axis=1)
        if len(bg) == 0:
            continue
        if len(bg) < n:
            bg = np.tile(bg, int(np.ceil(n / len(bg))))
        bg = bg[:n]
        out = out + bg * vol
    return np.clip(out, -1.0, 1.0)


def process_file(path, bg_selections):
    import tensorflow as tf
    yamnet, class_names = load_yamnet()
    trig_idx = [i for i, n in enumerate(class_names) if n in TRIGGERS]

    wav16, _ = sf.read(str(extract(path, SR_Y)), dtype="float32")
    scores, _, _ = yamnet(tf.constant(wav16))
    scores = scores.numpy()
    trig = scores[:, trig_idx].max(axis=1)

    centers = np.arange(len(trig)) * HOP + WIN / 2
    trig = dilate(trig, PAD_FRAMES)
    audio, sr = sf.read(str(extract(path, SR_D)), dtype="float32")
    t = np.arange(len(audio)) / sr
    env = np.interp(t, centers, trig)
    amount = np.clip((env - THRESH_LO) / (THRESH_HI - THRESH_LO + 1e-9), 0.0, 1.0)
    gain = 10 ** (-MAX_REDUCTION_DB * amount / 20)

    if USE_DEEPFILTERNET:
        import torch
        from df import enhance
        model, state = load_df()
        audio = enhance(model, state, torch.from_numpy(audio).unsqueeze(0)
                        ).squeeze(0).detach().cpu().numpy()

    n = min(len(audio), len(gain))
    out = np.clip(audio[:n] * gain[:n], -1.0, 1.0)

    # mix chosen background sounds on top of the cleaned audio
    if bg_selections:
        out = mix_backgrounds(out, sr, bg_selections)

    out_path = Path(tempfile.mkdtemp()) / "cleaned.wav"
    sf.write(str(out_path), out, sr)
    return out_path


@st.cache_data(show_spinner=False)
def load_bg_array(path_str, target_sr):
    """Decode a background file to mono float32 at target_sr (cached)."""
    wav = extract(Path(path_str), target_sr)
    data = sf.read(str(wav), dtype="float32")[0]
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def background_picker(key_prefix):
    """Grid of background sounds with icon + volume slider. Returns selections."""
    bgs = list_backgrounds()
    selections = []
    if not bgs:
        st.caption("No background sound files found in the app folder.")
        return selections
    st.write("Set a volume above 0 to mix that sound in.")
    cols = st.columns(3)
    for i, (path, emoji, name) in enumerate(bgs):
        with cols[i % 3]:
            st.markdown(
                f"<div style='font-size:46px;text-align:center;line-height:1'>{emoji}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div style='text-align:center;font-size:13px'>{name}</div>",
                        unsafe_allow_html=True)
            vol = st.slider("volume", 0, 100, 0, key=f"{key_prefix}_bg_{i}",
                            label_visibility="collapsed")
            if vol > 0:
                selections.append((path, vol / 100.0))
    return selections


def run_app():
    st.set_page_config(page_title="Misophonia Trigger Shield", page_icon="🛡️")

    # --- Logo ---
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        b64 = base64.b64encode(logo_path.read_bytes()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:480px;display:block;margin-bottom:8px">',
            unsafe_allow_html=True,
        )

    load_yamnet()
    if USE_DEEPFILTERNET:
        load_df()

    tab_up, tab_rt = st.tabs(["📁 Upload", "🎙️ Live"])

    # ---- Upload ----------------------------------------------------------
    with tab_up:
        up = st.file_uploader("Upload audio / video",
                              type=["mp3", "wav", "m4a", "aac", "flac", "ogg",
                                    "mp4", "mkv", "mov", "avi", "webm"],
                              key="uploader")

        process_clicked = st.button("Process", type="primary", key="process_btn")

        with st.expander("🎵 Add background sound (optional)"):
            bg_selections = background_picker("up")

        if up and process_clicked:
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getbuffer())
            with st.spinner("Processing..."):
                out_path = process_file(tmp, bg_selections)
            st.success("Done!")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Original")
                st.audio(str(tmp))
            with c2:
                st.caption("Cleaned")
                st.audio(str(out_path))
            st.download_button("⬇️ Download cleaned audio",
                               data=Path(out_path).read_bytes(),
                               file_name="cleaned.wav", mime="audio/wav",
                               key="download_btn")

    # ---- Live ------------------------------------------------------------
    with tab_rt:
        st.caption("Click START, allow mic. Speak / chew near the mic — "
                   "triggers get ducked and the audio comes back to you.")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av
            import tensorflow as tf

            with st.expander("🎵 Add background sound (optional)"):
                live_bg = background_picker("rt")

            # preload selected backgrounds (at 48k) for the live mixer
            BG_SR = 48000
            bg_live = [(load_bg_array(str(p), BG_SR), v) for p, v in live_bg]

            yamnet, class_names = load_yamnet()
            trig_idx = [i for i, n in enumerate(class_names) if n in TRIGGERS] or [0]

            class Reducer:
                def __init__(self, bg_src):
                    self.buf = np.zeros(SR_Y, dtype=np.float32)
                    self.gain = 1.0
                    self.bg_src = bg_src          # list of (float32 mono @48k, vol)
                    self.bg = None                # resampled to frame sr (lazy)
                    self.bg_pos = 0

                def _prep_bg(self, sr):
                    self.bg = []
                    for data, vol in self.bg_src:
                        if sr != BG_SR:
                            g = np.gcd(sr, BG_SR)
                            data = resample_poly(data, sr // g, BG_SR // g)
                        self.bg.append((data.astype(np.float32), vol))

                def recv(self, frame):
                    pcm = frame.to_ndarray()
                    sr = frame.sample_rate
                    mono = pcm.astype(np.float32)
                    if mono.ndim > 1:
                        mono = mono.mean(axis=0)
                    if np.max(np.abs(mono)) > 1.5:
                        mono = mono / 32768.0
                    g = np.gcd(SR_Y, sr)
                    m16 = resample_poly(mono, SR_Y // g, sr // g).astype(np.float32)
                    k = len(m16)
                    if k >= len(self.buf):
                        self.buf = m16[-len(self.buf):]
                    else:
                        self.buf = np.concatenate([self.buf[k:], m16])
                    scores, _, _ = yamnet(tf.constant(self.buf))
                    prob = float(scores.numpy()[:, trig_idx].max())
                    target = 10 ** (-MAX_REDUCTION_DB / 20) if prob >= THRESH_LO else 1.0
                    self.gain += np.clip(target - self.gain, -0.3, 0.3)

                    is_int = np.issubdtype(pcm.dtype, np.integer)
                    scale = 32768.0 if is_int else 1.0
                    out = pcm.astype(np.float32) * self.gain

                    # mix background sounds (looped) into the live output
                    if self.bg_src:
                        if self.bg is None:
                            self._prep_bg(sr)
                        N = out.shape[-1]
                        for data, vol in self.bg:
                            L = len(data)
                            if L == 0:
                                continue
                            idx = (self.bg_pos + np.arange(N)) % L
                            out = out + data[idx] * vol * scale
                        self.bg_pos += N

                    if is_int:
                        out = np.clip(out, -32768, 32767)
                    else:
                        out = np.clip(out, -1.0, 1.0)
                    out = out.astype(pcm.dtype)
                    new = av.AudioFrame.from_ndarray(out, format=frame.format.name,
                                                     layout=frame.layout.name)
                    new.sample_rate = sr
                    return new

            webrtc_streamer(
                key="shush-rt",
                mode=WebRtcMode.SENDRECV,
                audio_frame_callback=Reducer(bg_live).recv,
                media_stream_constraints={"audio": True, "video": False},
            )
        except ModuleNotFoundError:
            st.warning("Live needs extra packages:\n\n"
                       "`python -m pip install streamlit-webrtc av`")


if __name__ == "__main__":
    run_app()