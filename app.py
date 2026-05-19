"""
VisDrone Aerial Object Detection — Streamlit Interface
Course: AI447 Computer Vision
"""

import streamlit as st
import cv2.cv2 as cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisDrone Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #1e2130;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e8eaf0;
}

/* Header */
.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1.1;
}

.main-subtitle {
    font-size: 0.9rem;
    color: #5a6080;
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
    margin-bottom: 24px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #13161e;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #4fc3f7;
    line-height: 1;
}

.metric-label {
    font-size: 0.75rem;
    color: #5a6080;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Model card */
.model-card {
    background: #13161e;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}

.model-card.active {
    border-color: #4fc3f7;
    background: #0d1a26;
}

.model-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: #ffffff;
}

.model-meta {
    font-size: 0.75rem;
    color: #5a6080;
    margin-top: 2px;
}

/* Class tag */
.class-tag {
    display: inline-block;
    background: #1e2130;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #a0a8c0;
    margin: 2px;
    font-family: 'Space Mono', monospace;
}

/* Result box */
.result-box {
    background: #13161e;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px;
}

/* Status badge */
.badge-success {
    background: #0d2a1a;
    color: #4caf82;
    border: 1px solid #1a4a30;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
}

.badge-warn {
    background: #2a1a0d;
    color: #f0a040;
    border: 1px solid #4a3010;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 16px 0;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #13161e;
    border: 2px dashed #1e2130;
    border-radius: 12px;
    padding: 10px;
}

/* Selectbox, slider */
[data-testid="stSelectbox"] > div,
[data-baseweb="select"] {
    background-color: #1a1d27 !important;
    border-color: #2a2d3d !important;
    color: #e8eaf0 !important;
}

.stSlider [data-baseweb="slider"] {
    color: #4fc3f7;
}

/* Table */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2d3d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "YOLO11m  — Best Overall": {
        "path": r"C:\Users\user\runs\detect\yolo26_visdrone\yolo11m_run1\weights\best.pt",
        "params": "20.1M",
        "map50_test": "0.437",
        "fps": "67",
        "type": "CNN",
    },
    "YOLO26m  — CNN Baseline": {
        "path": r"C:\Users\user\runs\detect\yolo26_visdrone\yolom_run2_300ep\weights\best.pt",
        "params": "21.8M",
        "map50_test": "0.433",
        "fps": "69",
        "type": "CNN",
    },
    "YOLO26s — Lightweight": {
        "path": r"C:\Users\user\runs\detect\yolo26_visdrone\run4_copypaste2\weights\best.pt",
        "params": "9.9M",
        "map50_test": "0.474",
        "fps": "79",
        "type": "CNN",
    },
    "RT-DETR-L — Transformer": {
        "path": r"C:\Users\user\runs\detect\yolo26_visdrone\rtdetr_run4_300ep2\weights\best.pt",
        "params": "32.8M",
        "map50_test": "0.351",
        "fps": "28",
        "type": "Transformer",
    },
}

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

CLASS_COLORS = [
    (255, 99, 99), (255, 165, 60), (255, 220, 60), (60, 200, 100),
    (60, 180, 255), (140, 100, 255), (255, 100, 200), (100, 220, 220),
    (200, 140, 60), (180, 255, 100),
]

# ── Model cache ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)

# ── Helpers ────────────────────────────────────────────────────────────────────
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def run_detection(model: YOLO, img_bgr: np.ndarray, conf: float, imgsz: int = 960):
    t0 = time.time()
    results = model.predict(img_bgr, conf=conf, imgsz=imgsz, device=0, verbose=False)
    elapsed_ms = (time.time() - t0) * 1000
    return results[0], elapsed_ms

def build_class_summary(result) -> dict:
    counts = {c: 0 for c in CLASSES}
    if result.boxes is not None:
        for cls_id in result.boxes.cls.cpu().numpy().astype(int):
            if cls_id < len(CLASSES):
                counts[CLASSES[cls_id]] += 1
    return {k: v for k, v in counts.items() if v > 0}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚁 VisDrone System")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("### Model")
    selected_model_name = st.selectbox(
        "Select architecture",
        list(MODEL_REGISTRY.keys()),
        label_visibility="collapsed",
    )
    info = MODEL_REGISTRY[selected_model_name]
    st.markdown(f"""
    <div class="model-card active">
        <div class="model-name">{selected_model_name.split('—')[0].strip()}</div>
        <div class="model-meta">
            {info['type']} &nbsp;·&nbsp; {info['params']} params
            &nbsp;·&nbsp; {info['fps']} FPS &nbsp;·&nbsp;
            Test mAP50 {info['map50_test']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Settings")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    imgsz = st.select_slider("Inference resolution", options=[640, 960, 1280], value=960)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Dataset")
    st.markdown("**VisDrone2019-DET**")
    st.markdown(f"10 classes · 6,470 train images")
    tags_html = "".join(f'<span class="class-tag">{c}</span>' for c in CLASSES)
    st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.72rem;color:#3a4060;font-family:Space Mono,monospace;">'
        'AI447 Computer Vision · Spring 2025/2026</div>',
        unsafe_allow_html=True,
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Aerial Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">VisDrone2019 · CNN vs Transformer Benchmark</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_image, tab_video = st.tabs(["📷  Image Detection", "🎬  Video Tracking"])

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    uploaded_image = st.file_uploader(
        "Upload a drone image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_upload",
    )

    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        img_bgr = pil_to_cv2(pil_img)

        with st.spinner("Loading model..."):
            model = load_model(info["path"])

        with st.spinner("Running detection..."):
            result, elapsed_ms = run_detection(model, img_bgr, conf_thresh, imgsz)

        annotated_bgr = result.plot(line_width=2)
        annotated_pil = cv2_to_pil(annotated_bgr)

        n_objects = len(result.boxes) if result.boxes is not None else 0
        class_summary = build_class_summary(result)

        # ── Metrics row ───────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_objects}</div>
                <div class="metric-label">Objects Detected</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{elapsed_ms:.0f}<span style="font-size:1rem">ms</span></div>
                <div class="metric-label">Inference Time</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(class_summary)}</div>
                <div class="metric-label">Classes Found</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            fps_val = 1000 / elapsed_ms if elapsed_ms > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fps_val:.0f}</div>
                <div class="metric-label">FPS</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Images side by side ───────────────────────────────────────────────
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)
        with col_det:
            st.markdown("**Detections**")
            st.image(annotated_pil, use_container_width=True)

        # ── Per-class breakdown ───────────────────────────────────────────────
        if class_summary:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Detected Classes**")
            cols = st.columns(min(len(class_summary), 5))
            for i, (cls_name, count) in enumerate(
                sorted(class_summary.items(), key=lambda x: -x[1])
            ):
                with cols[i % 5]:
                    color = CLASS_COLORS[CLASSES.index(cls_name)]
                    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 3px solid {hex_color};">
                        <div class="metric-value" style="font-size:1.5rem;color:{hex_color};">{count}</div>
                        <div class="metric-label">{cls_name}</div>
                    </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3a4060;">
            <div style="font-size:3rem;margin-bottom:16px;">🛸</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;">
                Upload a drone image to begin detection
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a drone video (MP4 / AVI)",
        type=["mp4", "avi", "mov"],
        key="vid_upload",
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        with st.spinner("Loading model..."):
            model = load_model(info["path"])

        out_path = r"C:\Users\user\tracked_output.mp4"

        st.info("Running ByteTrack detection + tracking — this may take a few minutes for long videos.")

        progress_bar = st.progress(0, text="Processing frames...")

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_vid, (w, h))

        cap = cv2.VideoCapture(tfile.name)
        frame_idx = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                conf=conf_thresh,
                imgsz=imgsz,
                tracker="bytetrack.yaml",
                persist=True,
                device=0,
                verbose=False,
            )
            annotated = results[0].plot(line_width=2)
            out_writer.write(annotated)

            frame_idx += 1
            progress = int((frame_idx / max(total_frames, 1)) * 100)
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            progress_bar.progress(
                min(progress, 100),
                text=f"Frame {frame_idx}/{total_frames} · {fps_proc:.1f} FPS processing"
            )

        cap.release()
        out_writer.release()
        progress_bar.progress(100, text="Done!")

        # ── Stats ─────────────────────────────────────────────────────────────
        total_time = time.time() - t_start
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{frame_idx}</div>
                <div class="metric-label">Frames Processed</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_time:.0f}<span style="font-size:1rem">s</span></div>
                <div class="metric-label">Total Time</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_fps:.1f}</div>
                <div class="metric-label">Avg FPS</div>
            </div>""", unsafe_allow_html=True)

        st.success(f"Tracking complete. Output saved to: `{out_path}`")
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Tracked Video",
                data=f,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3a4060;">
            <div style="font-size:3rem;margin-bottom:16px;">🎬</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;">
                Upload a drone video for detection + ByteTrack tracking
            </div>
        </div>
        """, unsafe_allow_html=True)
