"""
===============================================================================
SPEECH EMOTION RECOGNITION (SER) FOR COMMUNICATION SYSTEMS
Streamlit Web Application & Inference Dashboard
===============================================================================
This file serves as the main web application interface for a Speech Emotion 
Recognition (SER) system designed for Electronic & Communication Engineering (ECE).
It includes digital signal processing (DSP) visualization, live audio recording/upload,
deep learning inference using CNN, RNN, or LSTM models, and full performance analysis.
===============================================================================
"""

from pathlib import Path
import tempfile
import time

import numpy as np
import streamlit as st

# Attempt to import signal processing and plotting libraries
# If librosa/matplotlib are missing, fallback gracefully to basic mode
try:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    HAS_SIGNAL_LIBS = True
except ImportError:
    HAS_SIGNAL_LIBS = False

# Import local machine learning inference module
from src.inference import predict_emotion


# =============================================================================
# SECTION 1: STREAMLIT PAGE CONFIGURATION
# =============================================================================
# Configures browser tab title, favicon icon, layout mode (wide screen),
# and default sidebar state when loaded.
st.set_page_config(
    page_title="Speech Emotion Recognition | Communication Systems",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SECTION 2: GLOBAL CSS DESIGN SYSTEM (CUSTOM STYLING)
# =============================================================================
# Injects custom CSS rules to enforce a dark navy and cyan accent theme across 
# all Streamlit elements, buttons, sidebar navigation, metric cards, and charts.
st.markdown(
    """
    <style>
/* CSS Color Variables for Design Consistency */
:root {
    --navy-dark: #041421;
    --navy-surface: #092238;
    --navy-card: #0d2b45;
    --accent-cyan: #0EA5E9;
    --accent-hover: #0284C7;
    --text-highlight: #38BDF8;
    --text-main: #ffffff;
    --text-muted: #cbd5e1;
    --border-color: #17384f;
}

/* Global Application Background and Text Color */
.stApp {
    background-color: var(--navy-dark) !important;
    color: var(--text-main) !important;
}

/* Content Container Dimensions and Padding */
.block-container {
    max-width: 1500px;
    padding: 1.5rem 2.2rem 3rem 2.2rem;
}

/* Hide Default Streamlit Menu and Footer */
#MainMenu, footer { visibility: hidden; }

/* Header Bar Background Reset */
header {
    visibility: visible !important;
    background: transparent !important;
}

/* Keep Streamlit sidebar toggle visible in all browser themes and Streamlit versions */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

/* Style the sidebar toggle button */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] {
    color: #0EA5E9 !important;
    background-color: #061525 !important;
    border: 1px solid #17384f !important;
    border-radius: 6px !important;
}

/* Keep the icon visible */
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="stSidebarCollapseButton"] svg {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapsedControl"] button svg *,
[data-testid="stSidebarCollapseButton"] svg * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    opacity: 1 !important;
}


/* Custom Top Navigation / Title Banner */
.topbar {
    background: var(--navy-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}

.top-title {
    font-size: 16px;
    font-weight: 800;
    color: var(--text-main);
}

.top-sub {
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 2px;
}

/* Summary / Metric Cards Styling */
.metric-card {
    background: var(--navy-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
}

.metric-label {
    color: var(--text-muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 800;
}

.metric-value {
    margin-top: 8px;
    color: var(--text-main);
    font-size: 24px;
    font-weight: 900;
}

.metric-note {
    margin-top: 4px;
    color: var(--text-muted);
    font-size: 10px;
}

/* Section Headings Formatting */
.section-heading {
    margin: 22px 0 12px 0;
}

.section-heading h2 {
    color: var(--text-main);
    font-size: 18px;
    font-weight: 850;
    margin: 0;
}

.section-heading p {
    color: var(--text-muted);
    font-size: 11px;
    margin: 3px 0 0 0;
}

/* Reusable Content Panel / Card Containers */
.panel {
    background: var(--navy-surface);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

.panel-title {
    color: var(--text-main);
    font-size: 14px;
    font-weight: 850;
}

.panel-sub {
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 4px;
    line-height: 1.5;
}

/* Horizontal Benchmark Bar Visualizer */
.bar-row {
    display: grid;
    grid-template-columns: 80px 1fr 55px;
    gap: 10px;
    align-items: center;
    margin: 12px 0;
}

.bar-label {
    color: var(--text-main);
    font-size: 11px;
    font-weight: 750;
}

.bar-track {
    height: 10px;
    background: #061525;
    border-radius: 99px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #0284C7, #38BDF8);
}

.bar-value {
    text-align: right;
    color: var(--text-main);
    font-size: 11px;
    font-weight: 800;
}

/* Color Badges for Recognized Emotions */
.emotion-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 850;
    margin-top: 8px;
}

.emotion-angry { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
.emotion-happy { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.4); }
.emotion-neutral { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.4); }
.emotion-sad { background: rgba(245, 197, 66, 0.15); color: #f5c542; border: 1px solid rgba(245, 197, 66, 0.4); }

.prob-fill.emotion-angry { background: #ef4444 !important; }
.prob-fill.emotion-happy { background: #22c55e !important; }
.prob-fill.emotion-neutral { background: #3b82f6 !important; }
.prob-fill.emotion-sad { background: #f5c542 !important; }

/* Model Architecture Info Cards */
.model-card {
    background: var(--navy-surface);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px;
    height: 100%;
}

.model-name {
    font-size: 20px;
    font-weight: 900;
    color: var(--text-main);
}

.model-score {
    font-size: 28px;
    font-weight: 900;
    color: var(--text-highlight);
    margin-top: 10px;
}

.tag-selected {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
    font-size: 10px;
    font-weight: 800;
}

.tag-evaluated {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-muted);
    border: 1px solid var(--border-color);
    font-size: 10px;
    font-weight: 800;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #061525 !important;
    border-right: 1px solid var(--border-color) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #cbd5e1 !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    font-weight: 650 !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--accent-cyan) !important;
    color: #ffffff !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.35) !important;
}

/* Primary Action Buttons */
.stButton > button[kind="primary"] {
    background: var(--accent-cyan) !important;
    border-color: var(--accent-cyan) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 3: SESSION STATE INITIALIZATION & SYSTEM CONSTANTS
# =============================================================================

# Track active view page across interactions
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Store inference classification results across reruns
if "result" not in st.session_state:
    st.session_state.result = None

# Evaluation Benchmark Results for deep learning architectures
MODEL_RESULTS = {
    "CNN": {"accuracy": 50.89, "f1": 46.60, "params": "101,636", "type": "2D Convolutional Net"},
    "LSTM": {"accuracy": 35.71, "f1": 32.40, "params": "227,652", "type": "Long Short-Term Memory"},
    "RNN": {"accuracy": 28.57, "f1": 11.11, "params": "63,300", "type": "Recurrent Neural Net"},
}

# Mapping of emotion keys to display icons and labels
EMOTION_META = {
    "happy": ("😊", "Happy"),
    "angry": ("😠", "Angry"),
    "sad": ("😢", "Sad"),
    "neutral": ("😐", "Neutral"),
}


# =============================================================================
# SECTION 4: HELPER & UTILITY FUNCTIONS
# =============================================================================

def navigate(page_name):
    """Updates the session state navigation page variable."""
    st.session_state.page = page_name


def save_audio(audio_file):
    """Saves uploaded or recorded audio buffer to a temporary WAV file on disk."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.getvalue())
        return Path(tmp.name)


def render_topbar():
    """Renders the top title and subtitle banner according to the current active page."""
    titles = {
        "Dashboard": ("Dashboard", "Speech emotion recognition overview & benchmark"),
        "Analyze": ("Speech Analysis", "Record/upload speech and inspect signal features"),
        "Models": ("Model Architecture Comparison", "Comparative evaluation of CNN, RNN, and LSTM"),
        "Team": ("Project Team", "ECE Final Year Project contributors and team members"),
        "About": ("System Architecture & Details", "Description, Core Components, and Operating Flow"),
    }
    title, subtitle = titles.get(st.session_state.page, titles["Dashboard"])

    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="top-title">{title}</div>
                <div class="top-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renders the left navigation sidebar with updated title, page routing buttons, and system status indicator."""
    with st.sidebar:
        # Sidebar Header Branding
        st.markdown(
            """
            <div style="padding: 8px 4px 20px 4px; border-bottom: 1px solid #17384f; margin-bottom: 15px;">
                <div style="display:flex; align-items:center; gap: 10px;">
                    <div style="width:38px; height:38px; border-radius:10px; background:#0EA5E9; 
                                color:#fff; display:flex; align-items:center; justify-content:center; 
                                font-weight:900; font-size:16px;">SE</div>
                    <div>
                        <div style="color:#ffffff; font-size:13px; font-weight:850;">Speech Emotion Recognition</div>
                        <div style="color:#8fa7bb; font-size:10px;">for communication systems</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="color:#71879c; font-size:9px; font-weight:800; text-transform:uppercase; margin-bottom:8px;">Main Menu</div>', unsafe_allow_html=True)

        # Navigation menu items
        pages = [
            ("Dashboard", "⌂"),
            ("Analyze", "◉"),
            ("Models", "▥"),
            ("Team", "👥"),
            ("About", "ℹ"),
        ]

        # Generate interactive navigation buttons
        for name, icon in pages:
            is_active = st.session_state.page == name
            if st.button(
                f"{icon}   {name}",
                key=f"nav_{name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state.page = name
                    st.rerun()

        # System Active Indicator Box
        st.markdown(
            """
            <div style="margin-top: 25px; padding: 12px 16px; border-radius: 12px; background: #092238; border: 1px solid #17384f; display:flex; align-items:center; gap:10px;">
                <span style="width:10px; height:10px; border-radius:50%; background:#22c55e; box-shadow: 0 0 8px #22c55e;"></span>
                <span style="color:#edf5fc; font-size:13px; font-weight:800;">System Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_signal_visuals(audio_path):
    """
    Computes and plots speech signal analysis charts using librosa and matplotlib:
    1. Time Domain Waveform s(t)
    2. Mel-Frequency Spectrogram Visualization
    """
    if not HAS_SIGNAL_LIBS:
        st.info("Install `librosa` and `matplotlib` to render live signal waveform and Mel-Spectrogram plots.")
        return None, 16000, 0.0

    # Load audio file with original sampling rate
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    # Create figure plot with dark theme background
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    fig.patch.set_facecolor("#092238")

    # Subplot 1: Amplitude Envelope Waveform s(t)
    ax1.set_facecolor("#061525")
    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#38BDF8", alpha=0.9)
    ax1.set_title("Speech Signal Waveform  s(t)", color="#edf5fc", fontsize=10, fontweight="bold", pad=6)
    ax1.set_ylabel("Amplitude", color="#8fa7bb", fontsize=8)
    ax1.tick_params(colors="#8fa7bb", labelsize=8)
    ax1.grid(color="#17384f", linestyle="--", linewidth=0.5, alpha=0.5)

    # Subplot 2: Mel Spectrogram Transformation
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_dB = librosa.power_to_db(S, ref=np.max)
    ax2.set_facecolor("#061525")
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, ax=ax2, cmap="magma")
    ax2.set_title("Mel-Frequency Spectrogram Feature Field", color="#edf5fc", fontsize=10, fontweight="bold", pad=6)
    ax2.set_xlabel("Time (s)", color="#8fa7bb", fontsize=8)
    ax2.set_ylabel("Freq (Hz)", color="#8fa7bb", fontsize=8)
    ax2.tick_params(colors="#8fa7bb", labelsize=8)

    fig.tight_layout()
    return fig, sr, duration


# =============================================================================
# SECTION 5: PAGE VIEW 1 - SYSTEM DASHBOARD
# =============================================================================
def render_dashboard():
    """Renders high-level summary overview cards and deep learning model benchmark comparison charts."""
    cols = st.columns(4)
    metrics = [
        ("Evaluated Models", "3 Architectures", "CNN • RNN • LSTM", "▥"),
        ("Target Emotions", "4 Classes", "Happy • Angry • Sad • Neutral", "◆"),
        ("Top Accuracy", "50.89%", "2D-CNN (RAVDESS benchmark)", "↑"),
        ("Feature Extraction", "40 MFCCs", "Time-frequency representation", "≈"),
    ]

    # Render summary metrics cards
    for col, (label, val, note, icon) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between;">
                        <div class="metric-label">{label}</div>
                        <div style="color:#38BDF8; font-weight:900;">{icon}</div>
                    </div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-heading">
            <h2>Deep Learning Benchmark Results</h2>
            <p>Comparative accuracy across all three candidate model architectures evaluated on speech data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 0.7], gap="large")

    # Left Panel: Accuracy progress bars
    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Test Accuracy Comparison</div>
                <div class="panel-sub">CNN achieved the highest test accuracy in this experiment.</div>
            """,
            unsafe_allow_html=True,
        )

        for name in ["CNN", "LSTM", "RNN"]:
            score = MODEL_RESULTS[name]["accuracy"]
            st.markdown(
                f"""
                <div class="bar-row">
                    <div class="bar-label">{name}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{score}%"></div>
                    </div>
                    <div class="bar-value">{score:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Right Panel: Output target classes
    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Target Classification Outputs</div>
                <div class="panel-sub">Emotion states recognized by the deployed network:</div>
                <div style="margin-top:16px; line-height:2.4; font-size:13px; font-weight:750;">
                    😊 &nbsp; <span style="color:#22c55e;">Happy</span><br>
                    😠 &nbsp; <span style="color:#ef4444;">Angry</span><br>
                    😢 &nbsp; <span style="color:#f5c542;">Sad</span><br>
                    😐 &nbsp; <span style="color:#3b82f6;">Neutral</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Quick action button to launch analysis view
    if st.button("Launch Speech Analysis Interface →", type="primary", use_container_width=False):
        navigate("Analyze")
        st.rerun()


# =============================================================================
# SECTION 6: PAGE VIEW 2 - SPEECH ANALYSIS & INFERENCE INTERFACE
# =============================================================================
def render_analyze():
    """Renders speech acquisition controls (mic/upload), inference executor, softmax outputs, and signal plots."""
    left, right = st.columns([1.05, 0.95], gap="large")

    # ---------------- LEFT COLUMN: INPUT & SELECTION ----------------
    with left:
        st.markdown(
            """
            <div class="section-heading" style="margin-top:0">
                <h2>1. Speech Signal Input</h2>
                <p>Provide speech input via live microphone stream or WAV file upload.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        input_type = st.radio(
            "Input Source",
            ["🎙 Live Microphone", "📁 Upload WAV File"],
            horizontal=True,
            label_visibility="collapsed",
        )

        audio_data = None

        # Handle Microphone Input stream
        if "Microphone" in input_type:
            st.markdown(
                """
                <div class="panel" style="padding: 14px; margin-bottom:12px;">
                    <div style="font-size:12px; font-weight:800; color:#edf5fc;">🎙 Microphone Stream</div>
                    <div style="font-size:10px; color:#8fa7bb; margin-top:3px;">
                        Speak clearly into your microphone for 2 to 4 seconds.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            audio_data = st.audio_input("Record Speech Input")
        # Handle File Upload
        else:
            st.markdown(
                """
                <div class="panel" style="padding: 14px; margin-bottom:12px;">
                    <div style="font-size:12px; font-weight:800; color:#edf5fc;">📁 Audio File Stream</div>
                    <div style="font-size:10px; color:#8fa7bb; margin-top:3px;">
                        Select a standard uncompressed WAV audio sample.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            audio_data = st.file_uploader("Upload WAV File", type=["wav"], label_visibility="collapsed")

        # The final HCI application uses the best-performing model selected
        # from the CNN/RNN/LSTM comparison experiment.
        st.markdown(
            """
            <div class="panel" style="padding:14px; margin-top:12px; margin-bottom:12px;">
                <div style="font-size:12px; font-weight:800; color:#edf5fc;">
                    🧠 Deployed Inference Model
                </div>
                <div style="font-size:16px; font-weight:900; color:#38BDF8; margin-top:5px;">
                    CNN
                </div>
                <div style="font-size:10px; color:#8fa7bb; margin-top:3px;">
                    Selected after comparative evaluation of CNN, RNN, and LSTM.
                    The Models page contains the full comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Trigger model execution pipeline
        if audio_data is not None:
            st.audio(audio_data)

            if st.button("Run Speech Analysis", type="primary", use_container_width=True):
                temp_path = None
                try:
                    with st.spinner("Processing speech signal & executing inference..."):
                        temp_path = save_audio(audio_data)
                        t_start = time.perf_counter()

                        # Final application inference uses the selected CNN model.
                        emotion, confidence, probabilities = predict_emotion(temp_path)

                        latency_ms = (time.perf_counter() - t_start) * 1000

                        # Save results to session state
                        st.session_state.result = {
                            "emotion": emotion.lower(),
                            "confidence": float(confidence),
                            "probabilities": probabilities,
                            "latency_ms": latency_ms,
                            "file_path": str(temp_path),
                            "arch_used": "CNN",
                        }
                except Exception as err:
                    st.error(f"Inference processing failed: {err}")

    # ---------------- RIGHT COLUMN: INFERENCE RESULTS ----------------
    with right:
        st.markdown(
            """
            <div class="section-heading" style="margin-top:0">
                <h2>2. Emotion Prediction & Probabilities</h2>
                <p>Classification outputs and softmax score distribution.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        res = st.session_state.result

        if res is None:
            st.markdown(
                """
                <div class="panel" style="text-align:center; padding: 40px 20px;">
                    <div style="font-size:40px; opacity:0.6;">🎙️</div>
                    <div style="font-size:15px; font-weight:800; margin-top:10px;">Awaiting Audio Input</div>
                    <div style="font-size:11px; color:#8fa7bb; margin-top:4px;">
                        Record or upload a speech sample, then click 'Run Speech Analysis'.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            emo = res["emotion"]
            conf = res["confidence"]
            probs = res["probabilities"]
            lat = res["latency_ms"]
            arch = res["arch_used"]

            icon, emo_name = EMOTION_META.get(emo, ("◉", emo.capitalize()))

            # Display main predicted emotion badge
            st.markdown(
                f"""
                <div class="panel" style="text-align:center; padding: 20px;">
                    <div style="font-size:10px; font-weight:800; color:#8fa7bb; text-transform:uppercase;">Predicted Emotion</div>
                    <div style="font-size:42px; margin: 8px 0 2px 0;">{icon}</div>
                    <div style="font-size:26px; font-weight:900;">{emo_name}</div>
                    <div class="emotion-badge emotion-{emo}">Confidence: {conf * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Performance metadata bar
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; background:#061525; padding:10px 14px; 
                            border-radius:10px; border:1px solid #17384f; margin-bottom:16px; font-size:10px; color:#8fa7bb;">
                    <div>Model: <b style="color:#edf5fc;">{arch}</b></div>
                    <div>Inference Latency: <b style="color:#22c55e;">{lat:.1f} ms</b></div>
                    <div>MFCC Features: <b style="color:#edf5fc;">40 Coefficients</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Softmax confidence score distribution
            st.markdown('<div style="font-size:12px; font-weight:800; margin-bottom:8px;">Softmax Probability Scores</div>', unsafe_allow_html=True)

            for emo_key, prob_val in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                pct = float(prob_val) * 100
                st.markdown(
                    f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:750;">
                            <span>{emo_key.capitalize()}</span>
                            <span>{pct:.1f}%</span>
                        </div>
                        <div class="bar-track" style="margin-top:4px;">
                            <div class="bar-fill prob-fill emotion-{emo_key}" style="width:{pct:.1f}%"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---------------- LOWER PANEL: DSP SIGNAL ANALYSIS ----------------
    if res is not None and "file_path" in res:
        st.markdown(
            """
            <div class="section-heading">
                <h2>3. Signal Processing Feature Field</h2>
                <p>Time-domain waveform and Mel-frequency spectrogram for signal visualization.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig, sr, duration = plot_signal_visuals(res["file_path"])

        if fig is not None:
            c1, c2 = st.columns([1.5, 0.5])
            with c1:
                st.pyplot(fig, use_container_width=True)
            with c2:
                # Digital Signal Processing metrics sidebar panel
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-title">Signal Diagnostics</div>
                        <div style="margin-top:10px; font-size:11px; line-height:1.8; color:#8fa7bb;">
                            • Sampling Rate (F<sub>s</sub>): <b style="color:#edf5fc;">{sr} Hz</b><br>
                            • Audio Duration (T): <b style="color:#edf5fc;">{duration:.2f} s</b><br>
                            • Total Samples (N): <b style="color:#edf5fc;">{int(sr * duration)}</b><br>
                            • Visualization: <b style="color:#edf5fc;">64 Mel Filters</b><br>
                            • Model Feature: <b style="color:#edf5fc;">40 MFCCs</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# SECTION 7: PAGE VIEW 3 - MODEL COMPARISON ARCHITECTURE
# =============================================================================
def render_models():
    """Renders architectural breakdown and engineering justifications for CNN, RNN, and LSTM."""
    cols = st.columns(3)

    descriptions = {
        "CNN": "2D Convolutional Layers learn local acoustic patterns from the MFCC time-frequency representation.",
        "RNN": "Standard Recurrent units model temporal sequential dependencies across speech frames.",
        "LSTM": "Gated Recurrent structures mitigate vanishing gradients across longer speech sequences.",
    }

    # Render card for each architecture
    for col, name in zip(cols, ["CNN", "RNN", "LSTM"]):
        d = MODEL_RESULTS[name]
        with col:
            tag_class = "tag-selected" if name == "CNN" else "tag-evaluated"
            tag_text = "SELECTED FOR DEPLOYMENT" if name == "CNN" else "EVALUATED MODEL"

            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name">{name}</div>
                    <div style="font-size:10px; color:#38BDF8; font-weight:800;">{d['type']}</div>
                    <div style="font-size:11px; color:#8fa7bb; margin-top:8px; min-height:45px;">{descriptions[name]}</div>
                    <div class="model-score">{d['accuracy']:.2f}%</div>
                    <div style="font-size:10px; color:#8fa7bb; margin-top:4px;">
                        Macro F1 Score: <b style="color:#edf5fc;">{d['f1']:.2f}%</b><br>
                        Parameters: <b style="color:#edf5fc;">{d['params']}</b>
                    </div>
                    <span class="{tag_class}">{tag_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Engineering Justification Note
    st.markdown(
        """
        <div class="panel" style="margin-top:20px;">
            <div class="panel-title">Engineering Justification for CNN Selection</div>
            <div class="panel-sub" style="margin-top:6px; line-height:1.6;">
                In this experiment, the CNN achieved the highest test accuracy and macro F1 among the three evaluated models. 
                Therefore, CNN was selected for final application inference, while RNN and LSTM remain important comparison 
                models required by the project.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 8: PAGE VIEW 4 - PROJECT TEAM MEMBERS
# =============================================================================
def render_team():
    """Renders ECE project group member profile cards."""
    st.markdown(
        """
        <div class="section-heading" style="margin-top:0">
            <h2>ECE Project Team Members</h2>
            <p>Electronic & Communication Engineering Final Year Project Group</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    team_members = [
        {"name": "Akpovero Godsent Oghenevwairhe", "matric": "U2021/3020031", "role": "Group Leader"},
        {"name": "Olakjo Daniel Kayode", "matric": "U2020/3020051", "role": "Project Group Member"},
        {"name": "Ugoagha Somtochukwu Victor", "matric": "U2021/3020069", "role": "Project Group Member"},
        {"name": "Akinlolu Olamide Dominion", "matric": "U2020/3020016", "role": "Project Group Member"},
        {"name": "Ebosetale Oselene Caleb", "matric": "U2021/3020063", "role": "Project Group Member"},
        {"name": "Shittu Oluwalayomi Mfon-Obong", "matric": "U2020/3020056", "role": "Project Group Member"},
        {"name": "Richard Isaiah", "matric": "U2020/3020027", "role": "Project Group Member"},
        {"name": "Ordu Thankgod Meyi", "matric": "U2021/3020045", "role": "Project Group Member"},
    ]

    cols = st.columns(2, gap="large")

    for idx, member in enumerate(team_members):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="panel" style="display:flex; align-items:center; gap:16px;">
                    <div style="width:48px; height:48px; border-radius:50%; background:rgba(14, 165, 233, 0.15); 
                                border:1px solid #38BDF8; color:#38BDF8; display:flex; align-items:center; 
                                justify-content:center; font-size:18px; font-weight:900;">
                        {member['name'][0]}
                    </div>
                    <div>
                        <div style="font-size:15px; font-weight:850; color:#edf5fc;">{member['name']}</div>
                        <div style="font-size:12px; font-weight:700; color:#38BDF8; margin-top:2px;">Matric No: {member['matric']}</div>
                        <div style="font-size:11px; color:#8fa7bb; margin-top:4px;">{member['role']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# SECTION 9: PAGE VIEW 5 - ABOUT & SYSTEM ARCHITECTURE
# =============================================================================
def render_about():
    """Renders detailed project documentation, system components, and operating flow steps."""
    # 1. System Description
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title" style="font-size:16px;">Description</div>
            <div class="panel-sub" style="margin-top:8px; line-height:1.6; font-size:12px;">
                This Speech Emotion Recognition (SER) system is an Electronic & Communication Engineering (ECE) 
                application developed for real-time Human-Computer Interaction (HCI). The system captures acoustic 
                speech signals, performs digital signal processing and feature extraction, and classifies continuous 
                speech into four primary target emotions: <b>Happy, Angry, Sad, and Neutral</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Core Components List
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title" style="font-size:16px;">Core Components</div>
            <div class="panel-sub" style="margin-top:8px; line-height:1.7; font-size:12px;">
                <div style="margin-bottom:8px;">• <b>Signal Acquisition & Conditioning:</b> Audio stream capture via live browser microphone or uploaded WAV files, resampled at F<sub>s</sub> = 16 kHz.</div>
                <div style="margin-bottom:8px;">• <b>DSP Feature Extractor:</b> Computation of 40 Mel-Frequency Cepstral Coefficients (MFCCs). A Mel spectrogram is also generated separately for signal visualization.</div>
                <div style="margin-bottom:8px;">• <b>Deep Neural Classifiers:</b> Evaluated 2D Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), and Long Short-Term Memory (LSTM) models.</div>
                <div>• <b>HCI Interface & Dashboard:</b> Real-time Streamlit dashboard rendering live spectral diagnostics, latency metrics, and confidence distributions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. How It Works - Sequential Flow Diagram
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title" style="font-size:16px; margin-bottom:12px;">How It Works</div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Speech Signal Acquisition", "Speech is recorded through the browser microphone or supplied as a WAV file."),
        ("02", "Digital Preprocessing", "Audio is converted to mono and standardized to 16 kHz, amplitude-normalized, then cropped or zero-padded to 4 seconds."),
        ("03", "MFCC Feature Extraction", "40 MFCC coefficients are extracted using the same preprocessing configuration used during CNN training (n_fft = 512, hop_length = 256)."),
        ("04", "Deep Learning Tensor Inference", "The resulting MFCC feature matrix is arranged as a time-by-feature sequence and passed to the trained CNN."),
        ("05", "Softmax Decision & Classification", "The CNN produces four class scores; softmax converts them into probabilities for happy, angry, sad, and neutral.")
    ]

    for num, title, desc in steps:
        st.markdown(
            f"""
            <div style="display:flex; gap:14px; align-items:center; background:#061525; border:1px solid #17384f; padding:12px 16px; border-radius:10px; margin-bottom:10px;">
                <div style="width:34px; height:34px; border-radius:8px; background:#0EA5E9; 
                            color:#fff; display:flex; align-items:center; justify-content:center; 
                            font-weight:900; font-size:13px; flex-shrink:0;">{num}</div>
                <div>
                    <div style="font-size:12px; font-weight:850; color:#edf5fc;">{title}</div>
                    <div style="font-size:11px; color:#8fa7bb; margin-top:2px;">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# SECTION 10: MAIN APPLICATION ROUTER & RENDER LOOP
# =============================================================================
# 1. Render persistent global sidebar and topbar
render_sidebar()
render_topbar()

# 2. Dynamically route and render requested page view based on active session state
if st.session_state.page == "Dashboard":
    render_dashboard()
elif st.session_state.page == "Analyze":
    render_analyze()
elif st.session_state.page == "Models":
    render_models()
elif st.session_state.page == "Team":
    render_team()
elif st.session_state.page == "About":
    render_about()